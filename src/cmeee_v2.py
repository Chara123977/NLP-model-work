import os
import json
import numpy as np
import paddle
import matplotlib.pyplot as plt
import seaborn as sns
from paddlenlp.transformers import ErnieForTokenClassification
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from config.cmeee_config import Config
import util.tools_function as tf
from data.med_cmee_data_loader import CMeEEDataLoader


class CMeEEEntityRecognizer:
	"""CMeEE-V2实体识别模型训练与推理类，新增可视化功能"""

	def __init__(self, config_path=None):
		"""
        初始化实体识别器
        :param config_path: 配置文件路径（可选，默认使用Config()）
        """
		# 初始化配置
		self.config = Config() if config_path is None else Config(config_path)
		paddle.set_device(self.config.device)

		# 初始化核心组件
		self.data_loader = None
		self.tokenizer = None
		self.model = None
		self.optimizer = None
		self.loss_fn = None
		self.best_f1 = 0.0

		# 保存测试集原始数据
		self.test_raw_data = None

		# 可视化相关数据记录
		self.train_losses = []  # 训练损失记录
		self.dev_losses = []  # 验证损失记录
		self.dev_macro_f1 = []  # 验证宏F1记录
		self.dev_precision = []  # 验证精度记录
		self.all_dev_labels = []  # 所有验证标签（用于混淆矩阵）
		self.all_dev_preds = []  # 所有验证预测（用于混淆矩阵）

		# 初始化数据加载器和分词器
		self._init_data_loader()
		self._init_tokenizer()
		self._load_test_raw_data()

	def _init_data_loader(self):
		"""内部方法：初始化数据加载器"""
		self.data_loader = CMeEEDataLoader(self.config)
		self.data_loader.load_datasets()
		self.data_loader.build_dataloaders()
		print("数据加载器初始化完成（训练/验证集）")

	def _load_test_raw_data(self):
		"""核心修复：直接从JSON文件加载测试集原始数据"""
		test_data_path = os.path.join(self.config.data_dir, 'CMeEE-V2_test.json')

		if not os.path.exists(test_data_path):
			raise FileNotFoundError(f"测试集文件不存在：{test_data_path}")

		try:
			with open(test_data_path, 'r', encoding='utf-8') as f:
				self.test_raw_data = json.load(f)

			valid_samples = 0
			for idx, sample in enumerate(self.test_raw_data):
				if 'text' in sample and 'entities' in sample:
					valid_samples += 1
				else:
					print(f"警告：测试集第{idx}条样本结构异常，缺少text/entities字段：{sample}")

			print(
				f"已直接加载测试集原始数据（绕过Dataset），共{len(self.test_raw_data)}条样本，有效结构样本{valid_samples}条")

		except Exception as e:
			raise ValueError(f"加载测试集原始数据失败：{e}")

	def _init_tokenizer(self):
		"""内部方法：初始化分词器"""
		self.tokenizer = self.data_loader.load_tokenizer()

	def _init_model(self):
		"""内部方法：初始化模型、优化器、损失函数"""
		self.model = ErnieForTokenClassification.from_pretrained(
			self.config.model_name,
			num_classes=self.config.num_classes
		)

		self.optimizer = paddle.optimizer.AdamW(
			learning_rate=self.config.learning_rate,
			parameters=self.model.parameters()
		)

		self.loss_fn = paddle.nn.CrossEntropyLoss(
			ignore_index=self.config.LABEL2ID['O']
		)

	def _train_one_epoch(self, loader):
		"""内部方法：训练一个epoch"""
		self.model.train()
		total_loss = 0.0
		step = 0

		for batch in tqdm(loader, desc="Training"):
			step += 1
			input_ids = paddle.to_tensor(batch[0]) if not isinstance(batch[0], paddle.Tensor) else batch[0]
			attention_mask = paddle.to_tensor(batch[1]) if not isinstance(batch[1], paddle.Tensor) else batch[1]
			label_ids = paddle.to_tensor(batch[2]) if not isinstance(batch[2], paddle.Tensor) else batch[2]

			input_ids = input_ids.astype('int64')
			attention_mask = attention_mask.astype('int64')
			label_ids = label_ids.astype('int64')

			outputs = self.model(input_ids, attention_mask=attention_mask)
			logits = outputs[0] if isinstance(outputs, tuple) else outputs

			if len(logits.shape) == 2:
				batch_size = label_ids.shape[0]
				seq_len = label_ids.shape[1]
				num_classes = logits.shape[-1]
				logits = logits.reshape([batch_size, seq_len, num_classes])

			logits_flat = logits.reshape([-1, self.config.num_classes])
			labels_flat = label_ids.reshape([-1])

			mask = (labels_flat != self.config.LABEL2ID['O']).astype('float32')
			if paddle.sum(mask) > 0:
				loss = self.loss_fn(logits_flat, labels_flat)
			else:
				loss = paddle.to_tensor(0.0)

			loss.backward()
			self.optimizer.step()
			self.optimizer.clear_grad()

			total_loss += loss.item()

			if step % 100 == 0:
				print(f"Step {step}, Loss: {loss.item():.4f}")

		avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
		return avg_loss

	def _evaluate(self, loader, is_dev=True):
		"""内部方法：验证/测试模型，新增宏F1和精度计算"""
		self.model.eval()
		all_preds = []
		all_labels = []
		total_loss = 0.0

		with paddle.no_grad():
			for batch in tqdm(loader, desc="Evaluating"):
				input_ids = paddle.to_tensor(batch[0]) if not isinstance(batch[0], paddle.Tensor) else batch[0]
				attention_mask = paddle.to_tensor(batch[1]) if not isinstance(batch[1], paddle.Tensor) else batch[1]
				label_ids = paddle.to_tensor(batch[2]) if not isinstance(batch[2], paddle.Tensor) else batch[2]

				input_ids = input_ids.astype('int64')
				attention_mask = attention_mask.astype('int64')
				label_ids = label_ids.astype('int64')

				outputs = self.model(input_ids, attention_mask=attention_mask)
				logits = outputs[0] if isinstance(outputs, tuple) else outputs

				if len(logits.shape) == 2:
					batch_size = label_ids.shape[0]
					seq_len = label_ids.shape[1]
					num_classes = logits.shape[-1]
					logits = logits.reshape([batch_size, seq_len, num_classes])

				# 计算验证损失
				logits_flat = logits.reshape([-1, self.config.num_classes])
				labels_flat = label_ids.reshape([-1])
				mask = (labels_flat != self.config.LABEL2ID['O']).astype('float32')
				if paddle.sum(mask) > 0:
					loss = self.loss_fn(logits_flat, labels_flat)
					total_loss += loss.item()

				# 获取预测结果
				preds = paddle.argmax(logits, axis=-1).numpy()
				labels = label_ids.numpy()

				# 收集有效标签
				batch_size, seq_len = preds.shape
				for b in range(batch_size):
					for i in range(seq_len):
						if labels[b][i] != self.config.LABEL2ID['O']:
							all_preds.append(preds[b][i])
							all_labels.append(labels[b][i])

		# 计算指标
		if not all_preds or not all_labels:
			print("警告：无有效预测结果用于计算指标")
			return 0.0, 0.0, 0.0, {}, all_labels, all_preds, 0.0

		try:
			# 计算micro F1、macro F1、精度
			micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
			macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

			# 生成详细报告获取精度
			unique_labels = sorted(list(set(all_labels + all_preds)))
			unique_target_names = [self.config.ID2LABEL.get(idx, f'unknown_{idx}') for idx in unique_labels]
			report = classification_report(
				all_labels,
				all_preds,
				labels=unique_labels,
				target_names=unique_target_names,
				output_dict=True,
				zero_division=0
			)
			precision = report['macro avg']['precision']
			avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

			# 记录验证数据（用于可视化）
			if is_dev:
				self.all_dev_labels = all_labels
				self.all_dev_preds = all_preds

		except Exception as e:
			print(f"计算指标失败：{e}")
			micro_f1 = 0.0
			macro_f1 = 0.0
			precision = 0.0
			report = {}
			avg_loss = 0.0

		return micro_f1, macro_f1, precision, report, all_labels, all_preds, avg_loss

	def _plot_training_curves(self):
		"""内部方法：绘制训练/验证损失、宏F1、精度曲线"""
		epochs = range(1, len(self.train_losses) + 1)

		# 创建2x2子图
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

		# 1. 训练损失曲线
		ax1.plot(epochs, self.train_losses, 'b-', linewidth=2, label='训练损失')
		ax1.plot(epochs, self.dev_losses, 'r-', linewidth=2, label='验证损失')
		ax1.set_title('训练损失 vs 验证损失', fontsize=14, fontweight='bold')
		ax1.set_xlabel('Epoch', fontsize=12)
		ax1.set_ylabel('损失值', fontsize=12)
		ax1.legend(fontsize=10)
		ax1.grid(True, alpha=0.3)

		# 2. 宏F1分数曲线
		ax2.plot(epochs, self.dev_macro_f1, 'g-', linewidth=2, marker='o', markersize=4)
		ax2.set_title('验证集宏F1分数曲线', fontsize=14, fontweight='bold')
		ax2.set_xlabel('Epoch', fontsize=12)
		ax2.set_ylabel('宏F1分数', fontsize=12)
		ax2.set_ylim(0, 1.0)
		ax2.grid(True, alpha=0.3)

		# 3. 精度曲线
		ax3.plot(epochs, self.dev_precision, 'orange', linewidth=2, marker='s', markersize=4)
		ax3.set_title('验证集精度曲线', fontsize=14, fontweight='bold')
		ax3.set_xlabel('Epoch', fontsize=12)
		ax3.set_ylabel('精度', fontsize=12)
		ax3.set_ylim(0, 1.0)
		ax3.grid(True, alpha=0.3)

		# 4. 综合曲线（宏F1 + 精度）
		ax4.plot(epochs, self.dev_macro_f1, 'g-', linewidth=2, label='宏F1分数', marker='o')
		ax4.plot(epochs, self.dev_precision, 'orange', linewidth=2, label='精度', marker='s')
		ax4.set_title('验证集宏F1分数 vs 精度', fontsize=14, fontweight='bold')
		ax4.set_xlabel('Epoch', fontsize=12)
		ax4.set_ylabel('分数', fontsize=12)
		ax4.set_ylim(0, 1.0)
		ax4.legend(fontsize=10)
		ax4.grid(True, alpha=0.3)

		plt.tight_layout()
		save_path = os.path.join(self.config.save_dir, 'training_curves.png')
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"\n训练曲线已保存至：{save_path}")
		plt.close()

	def _plot_confusion_matrix(self):
		"""内部方法：绘制归一化混淆矩阵"""
		if not self.all_dev_labels or not self.all_dev_preds:
			print("无验证数据，跳过混淆矩阵绘制")
			return

		# 获取有效类别
		unique_labels = sorted(list(set(self.all_dev_labels + self.all_dev_preds)))
		# 过滤O标签（如果存在）
		unique_labels = [label for label in unique_labels if label != self.config.LABEL2ID['O']]

		if len(unique_labels) == 0:
			print("无有效类别数据，跳过混淆矩阵绘制")
			return

		# 创建混淆矩阵
		cm = confusion_matrix(self.all_dev_labels, self.all_dev_preds, labels=unique_labels)
		# 归一化（按行）
		cm_normalized = normalize(cm, axis=1, norm='l1')

		# 获取类别名称
		class_names = [self.config.ID2LABEL.get(idx, f'unknown_{idx}') for idx in unique_labels]
		# 简化类别名称（去除B-/I-前缀）
		class_names = [name.split('-')[-1] if '-' in name else name for name in class_names]

		# 绘制混淆矩阵
		plt.figure(figsize=(12, 10))
		mask = np.triu(np.ones_like(cm_normalized, dtype=bool))  # 上三角掩码（可选）
		sns.heatmap(
			cm_normalized,
			annot=True,
			cmap='Blues',
			fmt='.2f',
			xticklabels=class_names,
			yticklabels=class_names,
			mask=None,  # 设为mask可只显示下三角
			annot_kws={"size": 8}
		)
		plt.title('归一化混淆矩阵（验证集）', fontsize=14, fontweight='bold', pad=20)
		plt.xlabel('预测标签', fontsize=12)
		plt.ylabel('真实标签', fontsize=12)
		plt.xticks(rotation=45, ha='right')
		plt.yticks(rotation=0)

		save_path = os.path.join(self.config.save_dir, 'confusion_matrix.png')
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"归一化混淆矩阵已保存至：{save_path}")
		plt.close()

	def train(self):
		"""
        【公开方法】训练模型，新增可视化数据记录
        :return: 最优验证集F1分数
        """
		self._init_model()

		train_loader = self.data_loader.get_dataloader('train')
		dev_loader = self.data_loader.get_dataloader('dev')

		os.makedirs(self.config.save_dir, exist_ok=True)

		print("\n开始训练...")
		for epoch in range(self.config.epochs):
			print(f"\n===== Epoch {epoch + 1}/{self.config.epochs} =====")

			# 训练
			train_loss = self._train_one_epoch(train_loader)
			self.train_losses.append(train_loss)
			print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

			# 验证
			dev_micro_f1, dev_macro_f1, dev_precision, dev_report, _, _, dev_loss = self._evaluate(dev_loader)
			self.dev_losses.append(dev_loss)
			self.dev_macro_f1.append(dev_macro_f1)
			self.dev_precision.append(dev_precision)

			print(f"Epoch {epoch + 1} Dev Micro F1: {dev_micro_f1:.4f}")
			print(f"Epoch {epoch + 1} Dev Macro F1: {dev_macro_f1:.4f}")
			print(f"Epoch {epoch + 1} Dev Precision: {dev_precision:.4f}")
			print(f"Epoch {epoch + 1} Dev Loss: {dev_loss:.4f}")

			# 保存最优模型
			if dev_macro_f1 > self.best_f1:  # 改用宏F1作为最优判断
				self.best_f1 = dev_macro_f1
				self.model.save_pretrained(self.config.save_dir)
				self.tokenizer.save_pretrained(self.config.save_dir)
				print(f"Best model saved (Macro F1: {self.best_f1:.4f})")

		# 训练结束后绘制可视化图表
		print("\n===== 生成训练可视化图表 =====")
		self._plot_training_curves()
		self._plot_confusion_matrix()

		# 保存训练日志
		training_log = {
			"train_losses": self.train_losses,
			"dev_losses": self.dev_losses,
			"dev_macro_f1": self.dev_macro_f1,
			"dev_precision": self.dev_precision,
			"best_macro_f1": self.best_f1,
			"total_epochs": self.config.epochs
		}
		log_path = os.path.join(self.config.save_dir, 'training_log.json')
		with open(log_path, 'w', encoding='utf-8') as f:
			json.dump(training_log, f, ensure_ascii=False, indent=2)
		print(f"训练日志已保存至：{log_path}")

		return self.best_f1

	def test(self, save_path=None):
		"""
        【公开方法】测试最优模型
        :param save_path: 预测结果保存路径
        :return: 测试集F1分数、详细报告、预测结果列表
        """
		if save_path is None:
			save_path = os.path.join(self.config.save_dir, "cmeee_test_predictions.json")
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		# 加载最优模型
		best_model = ErnieForTokenClassification.from_pretrained(
			self.config.save_dir,
			num_classes=self.config.num_classes
		)
		self.model = best_model

		# 预测测试集
		print("\n===== Predicting Test Dataset =====")
		test_predictions = []
		for idx, sample in enumerate(tqdm(self.test_raw_data, desc="Predicting Test Samples")):
			text = sample.get('text', '')
			if not text:
				print(f"警告：测试集第{idx}条样本无text字段，跳过")
				test_predictions.append({
					"text": "",
					"entities": [],
					"id": idx
				})
				continue

			pred_entities = self.predict(text)

			formatted_entities = []
			for ent in pred_entities:
				formatted_entities.append({
					"start_idx": ent["start"],
					"end_idx": ent["end"],
					"type": ent["type"],
					"entity": ent["text"]
				})

			pred_sample = {
				"text": text,
				"entities": formatted_entities,
				"id": idx
			}
			test_predictions.append(pred_sample)

		# 保存预测结果
		with open(save_path, 'w', encoding='utf-8') as f:
			json.dump(test_predictions, f, ensure_ascii=False, indent=2)
		print(f"\n测试集预测结果已保存至：{save_path}")

		# 计算测试集指标
		test_f1 = 0.0
		test_report = {}
		try:
			test_loader = self.data_loader.get_dataloader('test')
			print("\n===== Calculating Test Metrics =====")
			test_micro_f1, test_macro_f1, test_precision, test_report, _, _, _ = self._evaluate(test_loader,
																								is_dev=False)
			print(f"Test Micro F1: {test_micro_f1:.4f}")
			print(f"Test Macro F1: {test_macro_f1:.4f}")
			print(f"Test Precision: {test_precision:.4f}")
			test_f1 = test_macro_f1
		except Exception as e:
			print(f"\n测试集无标注数据/无法计算指标：{e}")

		return test_f1, test_report, test_predictions

	def predict(self, text):
		"""
        【公开方法】单条文本实体识别推理
        :param text: 输入医疗文本
        :return: 识别的实体列表
        """
		if self.model is None or self.best_f1 == 0.0:
			print("未检测到训练好的模型，自动加载最优模型...")
			self.model = ErnieForTokenClassification.from_pretrained(
				self.config.save_dir,
				num_classes=self.config.num_classes
			)

		self.model.eval()

		if hasattr(tf, 'clean_text'):
			text = tf.clean_text(text)
		if not text:
			return []

		tokenized = self.tokenizer(
			text,
			max_length=self.config.max_seq_len,
			padding='max_length',
			truncation=True,
			return_attention_mask=True,
			return_offsets_mapping=True
		)

		input_ids = paddle.to_tensor([tokenized['input_ids']], dtype='int64')
		attention_mask = paddle.to_tensor([tokenized['attention_mask']], dtype='int64')
		offset_mapping = tokenized['offset_mapping']

		with paddle.no_grad():
			outputs = self.model(input_ids, attention_mask=attention_mask)
			logits = outputs[0] if isinstance(outputs, tuple) else outputs

			if len(logits.shape) == 2:
				logits = logits.reshape([1, self.config.max_seq_len, self.config.num_classes])

			preds = paddle.argmax(logits, axis=-1).numpy()[0]

		results = []
		current_entity = ""
		current_type = ""
		current_start = -1
		filter_chars = ['（', '）', '。', '，', '、', '：', '；', ' ', '\t', '\n']

		for token_idx, (start, end) in enumerate(offset_mapping):
			if start == 0 and end == 0:
				continue
			if start >= len(text):
				break

			label = self.config.ID2LABEL[preds[token_idx]]
			current_char = text[start:end].strip()

			if all(c in filter_chars for c in current_char):
				if current_entity:
					results.append({
						'text': current_entity.strip(),
						'type': current_type,
						'start': current_start,
						'end': start
					})
					current_entity = ""
					current_type = ""
					current_start = -1
				continue

			if label.startswith('B-'):
				if current_entity:
					results.append({
						'text': current_entity.strip(),
						'type': current_type,
						'start': current_start,
						'end': start
					})
				current_type = label.split('-')[1]
				current_entity = current_char
				current_start = start
			elif label.startswith('I-') and current_entity:
				current_type_i = label.split('-')[1]
				if current_type_i == current_type:
					current_entity += current_char
			else:
				if current_entity:
					results.append({
						'text': current_entity.strip(),
						'type': current_type,
						'start': current_start,
						'end': start
					})
					current_entity = ""
					current_type = ""
					current_start = -1

		if current_entity:
			results.append({
				'text': current_entity.strip(),
				'type': current_type,
				'start': current_start,
				'end': len(text)
			})

		results = [r for r in results if r['text'] and len(r['text']) > 0]
		return results
