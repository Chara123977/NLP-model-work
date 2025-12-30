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
import random
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

        # 扩展同义词词典，为EDA增强做准备
        self.SYNONYM_DICT = {
            "治疗": ["医治", "诊疗", "疗愈", "救治", "治疗方案", "治疗方法"],
            "方案": ["方法", "策略", "方案", "举措", "计划", "措施"],
            "病因": ["原因", "病根", "病源", "诱因", "因素", "成因"],
            "分析": ["剖析", "解析", "分析", "研判", "研究", "探究"],
            "诊断": ["确诊", "判定", "诊断", "检查", "判断", "识别"],
            "建议": ["提议", "劝告", "建议", "指导", "意见", "提议"],
            "费用": ["花销", "开支", "费用", "资费", "花费", "成本"],
            "注意": ["留意", "小心", "注意", "警惕", "当心", "关注"],
            "功效": ["效果", "功用", "功效", "作用", "疗效", "好处"],
            "指标": ["指数", "数据", "指标", "参数", "数值", "参考值"],
            "后果": ["结果", "恶果", "后果", "结局", "影响", "结果"],
            "恶化": ["加重", "变坏", "恶化", "加剧", "变严重", "发展"],
            "方法": ["方式", "办法", "方法", "途径", "手段", "技巧"],
            "原因": ["缘由", "起因", "原因", "根由", "理由", "因素"],
            "倾向": ["趋势", "偏向", "倾向", "态势", "方向", "走向"],
            "恶变": ["癌变", "恶化", "恶变", "病变", "恶化", "变坏"],
            "疾病": ["病症", "病", "疾患", "疾病", "病情", "病症"],
            "症状": ["表现", "症状", "征象", "现象", "体征", "临床表现"],
            "药物": ["药品", "药物", "药", "药品", "药物治疗", "药物"],
            "医院": ["医疗机构", "医院", "诊所", "医疗所", "医疗中心", "医馆"],
            "医生": ["医师", "医生", "大夫", "医师", "医者", "医务工作者"],
            "检查": ["检验", "检查", "检测", "化验", "检验", "诊断"],
            "预防": ["防范", "防止", "预防", "避免", "防备", "预防措施"],
            "好转": ["改善", "好转", "恢复", "康复", "缓解", "向好"],
            "康复": ["恢复", "康复", "痊愈", "复原", "康复治疗", "恢复健康"],
            "诊断": ["确诊", "诊断", "判定", "检查", "识别", "判断"],
            "手术": ["手术", "外科手术", "操作", "手术治疗", "外科治疗", "手术干预"]
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

    # EDA数据增强方法：同义词替换
    def _synonym_replacement_eda(self, words, n):
        """同义词替换 - EDA方法之一"""
        if len(words) < 2:
            return ''.join(words)

        words = words.copy()
        replaced_count = 0

        for _ in range(n):
            # 找到可替换的词
            replaceable_indices = []
            for idx, word in enumerate(words):
                if word in self.SYNONYM_DICT and len(word) > 1:
                    replaceable_indices.append(idx)

            if not replaceable_indices:
                break

            # 随机选择一个词进行替换
            idx = random.choice(replaceable_indices)
            synonyms = self.SYNONYM_DICT[words[idx]]
            if synonyms:
                words[idx] = random.choice(synonyms)
                replaced_count += 1

        return ''.join(words)

    # EDA数据增强方法：随机插入
    def _random_insertion(self, words, n):
        """随机插入 - EDA方法之一"""
        if len(words) < 2:
            return ''.join(words)

        words = words.copy()
        for _ in range(n):
            # 找到可以插入同义词的词
            insertable_indices = [i for i, word in enumerate(words)
                                  if word in self.SYNONYM_DICT and len(word) > 1]
            if not insertable_indices:
                break

            idx = random.choice(insertable_indices)
            synonyms = self.SYNONYM_DICT[words[idx]]
            if synonyms:
                synonym = random.choice(synonyms)
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, synonym)

        return ''.join(words)

    # EDA数据增强方法：随机交换
    def _random_swap(self, words, n):
        """随机交换 - EDA方法之一"""
        if len(words) < 2:
            return ''.join(words)

        words = words.copy()
        for _ in range(n):
            if len(words) < 2:
                break
            # 随机选择两个位置进行交换
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]

        return ''.join(words)

    # EDA数据增强方法：随机删除
    def _random_deletion(self, words, p):
        """随机删除 - EDA方法之一"""
        if len(words) < 2:
            return ''.join(words)

        # 以概率p删除每个词
        words = [word for word in words if random.random() > p]
        if len(words) == 0:
            # 如果全部被删除，返回原始词
            return ''.join(words)

        return ''.join(words)

    # EDA主函数：结合四种增强方法
    def _eda_augmentation(self, text, alpha_sr=0.3, alpha_ri=0.3, alpha_rs=0.3, p_rd=0.3):
        """
		EDA数据增强：结合同义词替换、随机插入、随机交换、随机删除
		:param text: 原始文本
		:param alpha_sr: 同义词替换比例
		:param alpha_ri: 随机插入比例
		:param alpha_rs: 随机交换比例
		:param p_rd: 随机删除概率
		:return: 增强后的文本列表
		"""
        words = list(jieba.cut(text))
        words = [word for word in words if word.strip()]
        if len(words) <= 1:
            return [text]

        augmented_texts = []

        # 1. 同义词替换
        if alpha_sr > 0:
            n_sr = max(1, int(alpha_sr * len(words)))
            augmented_text = self._synonym_replacement_eda(words, n_sr)
            if augmented_text != text:  # 避免生成与原句相同的文本
                augmented_texts.append(augmented_text)

        # 2. 随机插入
        if alpha_ri > 0:
            n_ri = max(1, int(alpha_ri * len(words)))
            augmented_text = self._random_insertion(words, n_ri)
            if augmented_text != text:
                augmented_texts.append(augmented_text)

        # 3. 随机交换
        if alpha_rs > 0:
            n_rs = max(1, int(alpha_rs * len(words)))
            augmented_text = self._random_swap(words, n_rs)
            if augmented_text != text:
                augmented_texts.append(augmented_text)

        # 4. 随机删除
        if p_rd > 0:
            augmented_text = self._random_deletion(words, p_rd)
            if augmented_text != text:
                augmented_texts.append(augmented_text)

        return augmented_texts

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

    # 识别少数类
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

    # 对少数类进行EDA数据增强
    def augment_minority_samples(self, augment_ratio=1.0, alpha_sr=0.2, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
        """
		对少数类样本进行EDA数据增强
		:param augment_ratio: 增强比例（如1.0表示为每个少数类样本生成1个增强样本）
		:param alpha_sr: 同义词替换比例
		:param alpha_ri: 随机插入比例
		:param alpha_rs: 随机交换比例
		:param p_rd: 随机删除概率
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
        logger.info(f"少数类样本数：{len(minority_samples)}，开始EDA数据增强（增强比例：{augment_ratio}）")

        # 生成增强样本
        augmented_samples = []
        for idx, sample in enumerate(tqdm(minority_samples, desc="EDA数据增强")):
            # 生成EDA增强版本
            augmented_texts = self._eda_augmentation(
                sample["text"],
                alpha_sr=alpha_sr,
                alpha_ri=alpha_ri,
                alpha_rs=alpha_rs,
                p_rd=p_rd
            )

            for aug_idx, aug_text in enumerate(augmented_texts):
                aug_sample = {
                    "id": f"{sample['id']}_eda_{aug_idx}",
                    "text": aug_text,
                    "label": sample["label"],
                    "raw_label": sample["raw_label"]
                }
                augmented_samples.append(aug_sample)

        # 合并增强样本到训练集
        self.train_ds += augmented_samples
        logger.info(f"EDA数据增强完成！新增增强样本：{len(augmented_samples)}，训练集总数：{len(self.train_ds)}")

    def load_datasets(self):
        """加载训练/验证/测试集 + EDA数据增强"""
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

        # 使用EDA增强
        self.augment_minority_samples(
            augment_ratio=1.0,
            alpha_sr=0.2,  # 同义词替换比例
            alpha_ri=0.1,  # 随机插入比例
            alpha_rs=0.1,  # 随机交换比例
            p_rd=0.1  # 随机删除概率
        )

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

    # 修复：返回numpy数组格式的权重
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

        # 修复：使用numpy数组权重创建采样器
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
