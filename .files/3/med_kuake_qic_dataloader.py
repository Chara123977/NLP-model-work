import os
import json

import warnings
import numpy as np
import paddle
from tqdm import tqdm
import random

# 忽略所有无关警告（包括ccache、distutils等）
warnings.filterwarnings("ignore")

# 仅导入低版本PaddleNLP支持的模块
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from paddle.optimizer.lr import CosineAnnealingDecay
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger
import jieba

class KUAKEQICDataLoader:
	def __init__(self, data_dir, model_name="ernie-3.0-medium-zh", max_seq_len=128):
		self.data_dir = data_dir
		self.model_name = model_name
		self.max_seq_len = max_seq_len

		# 匹配你的标签体系（疾病表述 替代 疾病描述）
		self.label_list = [
			"治疗方案", "其他", "疾病表述", "病因分析", "注意事项",
			"功效作用", "病情诊断", "就医建议", "医疗费用", "指标解读", "后果表述"
		]

		# 加载同义词词典（使用哈工大LTP同义词林简化版）
		self.SYNONYM_DICT = {
			"治疗": ["医治", "诊疗", "疗愈", "救治"],
			"方案": ["方法", "策略", "方案", "举措"],
			"病因": ["原因", "病根", "病源", "诱因"],
			"分析": ["剖析", "解析", "分析", "研判"],
			"诊断": ["确诊", "判定", "诊断", "检查"],
			"建议": ["提议", "劝告", "建议", "指导"],
			"费用": ["花销", "开支", "费用", "资费"],
			"注意": ["留意", "小心", "注意", "警惕"],
			"功效": ["效果", "功用", "功效", "作用"],
			"指标": ["指数", "数据", "指标", "参数"],
			"后果": ["结果", "恶果", "后果", "结局"],
			"恶化": ["加重", "变坏", "恶化", "加剧"],
			"方法": ["方式", "办法", "方法", "途径"],
			"原因": ["缘由", "起因", "原因", "根由"],
			"倾向": ["趋势", "偏向", "倾向", "态势"],
			"恶变": ["癌变", "恶化", "恶变", "病变"]
		}
		self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
		self.id2label = {idx: label for idx, label in enumerate(self.label_list)}
		self.num_classes = len(self.label_list)

		# 加载分词器（直接用ErnieTokenizer，避免AutoTokenizer兼容问题）
		self.tokenizer = ErnieTokenizer.from_pretrained(self.model_name)

		# 数据集缓存
		self.train_ds = None
		self.dev_ds = None
		self.test_ds = None

	# 新增：同义词替换数据增强函数
	def _synonym_replacement(self, text, replace_rate=0.2):
		"""
		同义词替换增强
		:param text: 原始文本
		:param replace_rate: 替换比例（0-1）
		:return: 增强后的文本
		"""
		# 分词
		words = list(jieba.cut(text))
		# 可替换的词的索引
		replaceable_indices = []
		for idx, word in enumerate(words):
			if word in self.SYNONYM_DICT and len(word) > 1:  # 仅替换长度>1的词，避免单字无意义替换
				replaceable_indices.append(idx)

		# 计算需要替换的词数
		replace_num = max(1, int(len(replaceable_indices) * replace_rate))
		if replace_num == 0 or len(replaceable_indices) == 0:
			return text  # 无可用替换词，返回原文本

		# 随机选择要替换的词
		selected_indices = random.sample(replaceable_indices, replace_num)
		# 替换
		augmented_words = words.copy()
		for idx in selected_indices:
			word = augmented_words[idx]
			synonyms = self.SYNONYM_DICT[word]
			augmented_words[idx] = random.choice(synonyms)

		# 拼接成新文本
		augmented_text = "".join(augmented_words)
		return augmented_text

	def read_json_data(self, file_path):
		"""读取JSON格式的KUAKE-QIC数据集"""
		data = []
		try:
			full_filename = os.path.basename(file_path)

			with open(file_path, "r", encoding="utf-8") as f:
				raw_data = json.load(f)

			# 校验JSON格式为数组
			if not isinstance(raw_data, list):
				logger.error(f"JSON文件{file_path}不是数组格式，请检查")
				return data

			for line_idx, sample in enumerate(raw_data):
				# 提取核心字段
				sample_id = sample.get("id", f"unknown_{line_idx}")
				query = sample.get("query")
				label = sample.get("label")

				if "test" in full_filename:
					if not query:
						logger.warning(f"样本{sample_id}缺少query/label字段，跳过")
						continue
				elif not query or not label:
					logger.warning(f"样本{sample_id}缺少query/label字段，跳过")
					continue
				# 校验标签有效性
				elif label not in self.label2id:
					logger.warning(f"样本{sample_id}标签{label}无效（非11类），跳过")
					continue
				else:
					data.append({
						"id": sample_id,
						"text": query.strip(),
						"label": self.label2id[label],
						"raw_label": label
					})
		except json.JSONDecodeError as e:
			logger.error(f"JSON文件{file_path}解析失败：{e}")
			raise e
		except Exception as e:
			logger.error(f"读取JSON文件失败：{e}")
			raise e

		logger.info(f"成功读取{file_path}，有效样本数：{len(data)}")
		return data

	# 新增：识别少数类
	def _identify_minority_classes(self, dataset):
		"""
		识别少数类（样本数低于均值的类别）
		:param dataset: 训练集数据
		:return: 少数类id列表
		"""
		# 统计各类别样本数
		label_counts = {}
		for sample in dataset:
			label = sample["label"]
			label_counts[label] = label_counts.get(label, 0) + 1

		# 计算均值，低于均值的为少数类
		mean_count = np.mean(list(label_counts.values()))
		minority_classes = [label for label, count in label_counts.items() if count < mean_count]

		# 打印少数类信息
		logger.info(f"类别样本数统计：{[(self.id2label[l], c) for l, c in label_counts.items()]}")
		logger.info(f"样本数均值：{mean_count:.1f}，少数类：{[self.id2label[l] for l in minority_classes]}")

		return minority_classes

	# 新增：对少数类进行数据增强
	def augment_minority_samples(self, augment_ratio=1.0, replace_rate=0.2):
		"""
		对少数类样本进行同义词替换增强
		:param augment_ratio: 增强比例（如1.0表示为每个少数类样本生成1个增强样本）
		:param replace_rate: 同义词替换比例
		"""
		if self.train_ds is None:
			logger.warning("训练集未加载，跳过数据增强")
			return

		# 识别少数类
		minority_classes = self._identify_minority_classes(self.train_ds)
		if not minority_classes:
			logger.info("无少数类，跳过数据增强")
			return

		# 筛选少数类样本
		minority_samples = [s for s in self.train_ds if s["label"] in minority_classes]
		logger.info(f"少数类样本数：{len(minority_samples)}，开始数据增强（增强比例：{augment_ratio}）")

		# 生成增强样本
		augmented_samples = []
		for idx, sample in enumerate(tqdm(minority_samples, desc="数据增强")):
			# 生成指定数量的增强样本
			for aug_idx in range(int(augment_ratio)):
				# 同义词替换
				aug_text = self._synonym_replacement(sample["text"], replace_rate=replace_rate)
				# 构造新样本（id加后缀区分）
				aug_sample = {
					"id": f"{sample['id']}_aug_{aug_idx}",
					"text": aug_text,
					"label": sample["label"],
					"raw_label": sample["raw_label"]
				}
				augmented_samples.append(aug_sample)

		# 合并增强样本到训练集
		self.train_ds += augmented_samples
		logger.info(f"数据增强完成！新增增强样本：{len(augmented_samples)}，训练集总数：{len(self.train_ds)}")

	def load_datasets(self):
		"""加载训练/验证/测试集 + 数据增强"""
		train_path = os.path.join(self.data_dir, "train.json")
		dev_path = os.path.join(self.data_dir, "dev.json")
		test_path = os.path.join(self.data_dir, "test.json")

		# 读取各数据集
		if os.path.exists(train_path):
			self.train_ds = self.read_json_data(train_path)
		else:
			raise FileNotFoundError(f"训练集文件不存在：{train_path}")

		if os.path.exists(dev_path):
			self.dev_ds = self.read_json_data(dev_path)
		else:
			raise FileNotFoundError(f"验证集文件不存在：{dev_path}")

		if os.path.exists(test_path):
			self.test_ds = self.read_json_data(test_path)
		else:
			logger.warning("测试集文件（test.json）不存在，仅加载训练/验证集")

		# 新增：对少数类进行数据增强
		self.augment_minority_samples(augment_ratio=1.0, replace_rate=0.2)

		# 打印数据统计
		logger.info(f"数据加载完成（含增强）：训练集{len(self.train_ds)}条，验证集{len(self.dev_ds)}条")
		if self.test_ds:
			logger.info(f"测试集{len(self.test_ds)}条")

	def convert_example(self, example):
		"""将样本转换为tuple格式（兼容低版本Paddle）"""
		# 核心修改：显式构造input_ids/token_type_ids（避免分词器返回格式问题）
		encoded_inputs = self.tokenizer(
			text=example["text"],
			max_length=self.max_seq_len,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			return_token_type_ids=True,
			return_tensors="np"  # 返回numpy数组，避免paddle tensor兼容问题
		)

		# 提取并展平（避免二维数组）
		input_ids = encoded_inputs["input_ids"].flatten().tolist()
		token_type_ids = encoded_inputs["token_type_ids"].flatten().tolist()
		attention_mask = encoded_inputs["attention_mask"].flatten().tolist()
		label = int(example["label"])

		return (
			input_ids,
			token_type_ids,
			attention_mask,
			label
		)

	# ========== 修复：返回numpy数组格式的权重 ==========
	def get_sample_weights(self, dataset):
		"""计算样本权重，解决类别不平衡（返回numpy数组）"""
		# 统计每个类别的样本数
		label_counts = {}
		for sample in dataset:
			label = sample["label"]
			label_counts[label] = label_counts.get(label, 0) + 1

		# 计算每个类别的权重（反比于样本数）
		total_samples = len(dataset)
		class_weights = {label: total_samples / count for label, count in label_counts.items()}

		# 为每个样本分配权重，并转换为numpy数组（关键修复）
		sample_weights = np.array([class_weights[sample["label"]] for sample in dataset], dtype=np.float32)
		return sample_weights

	def create_dataloader(self, dataset, batch_size=32, is_train=True):
		"""创建DataLoader（Tuple批处理 + 类别加权采样）"""
		converted_dataset = [self.convert_example(example) for example in dataset]

		batchify_fn = Tuple(
			Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
			Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
			Pad(axis=0, pad_val=0),  # attention_mask
			Stack(dtype="int64")  # label
		)

		# ========== 修复：使用numpy数组权重创建采样器 ==========
		if is_train:
			# 计算样本权重（numpy数组）
			sample_weights = self.get_sample_weights(dataset)
			# 创建加权随机采样器（兼容numpy数组）
			sampler = paddle.io.WeightedRandomSampler(
				weights=sample_weights,
				num_samples=len(dataset),
				replacement=True  # 允许重复采样少数类，保证每个epoch样本数一致
			)
			# 使用加权采样创建DataLoader
			dataloader = paddle.io.DataLoader(
				dataset=converted_dataset,
				batch_sampler=paddle.io.BatchSampler(sampler, batch_size=batch_size, drop_last=True),
				collate_fn=batchify_fn,
				num_workers=0
			)
		else:
			# 验证/测试集保持原有逻辑
			dataloader = paddle.io.DataLoader(
				dataset=converted_dataset,
				batch_size=batch_size,
				shuffle=is_train,
				collate_fn=batchify_fn,
				drop_last=is_train,
				num_workers=0
			)
		return dataloader

