# 首先导入所有必需模块（补全warnings + 新增数据增强相关）
import os
import json
import random
import sys
import warnings
import numpy as np
import paddle
from tqdm import tqdm
from sklearn.metrics import recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

# 忽略所有无关警告（包括ccache、distutils等）
warnings.filterwarnings("ignore")

# 仅导入低版本PaddleNLP支持的模块
from paddlenlp.transformers import ErnieForSequenceClassification
from paddle.optimizer.lr import CosineAnnealingDecay
from paddlenlp.utils.log import logger

# 假设你的数据加载器在这个路径下，若路径不同请自行调整
from data.med_kuake_qic_dataloader import KUAKEQICDataLoader

# 通用中文字体配置，适配Windows/Mac/Linux
def setup_chinese_font():
	plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
	try:
		# Windows系统
		if os.name == 'nt':
			plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
		# MacOS系统
		elif sys.platform == 'darwin':
			plt.rcParams["font.family"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
		# Linux系统
		else:
			# 尝试常见的Linux中文字体
			font_list = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
			plt.rcParams["font.family"] = font_list

			# 静默过滤字体警告
			import matplotlib
			matplotlib.rcParams['font.sans-serif'] = font_list
	except:
		# 兜底方案：使用默认字体，关闭警告
		plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]

	# 关闭字体查找警告
	warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")


# 执行字体配置
setup_chinese_font()

# 忽略所有无关警告（包括ccache、distutils等）
warnings.filterwarnings("ignore")


# 设置随机种子
def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	paddle.seed(seed)


# ===================== 新增：独立的预测器类 =====================
class KUAKEQICPredictor:
	"""KUAKE-QIC意图分类预测器"""

	def __init__(self, config):
		self.config = config
		self.model = None
		self.tokenizer = None
		self.id2label = None
		self.label2id = None

		# 初始化预测器（加载模型和tokenizer）
		self._init_predictor()

	def _init_predictor(self):
		"""初始化预测器：加载最优模型、tokenizer和标签映射"""
		# 构建最优模型路径
		best_model_path = os.path.join(self.config.output_dir, "best_model")
		if not os.path.exists(best_model_path):
			raise FileNotFoundError(f"最优模型路径不存在：{best_model_path}")

		# 临时初始化数据加载器以获取tokenizer和标签映射
		temp_data_loader = KUAKEQICDataLoader(
			data_dir=self.config.data_dir,
			model_name=self.config.model_name,
			max_seq_len=self.config.max_seq_len
		)
		temp_data_loader.load_datasets()

		# 保存标签映射
		self.id2label = temp_data_loader.id2label
		self.label2id = temp_data_loader.label2id
		self.tokenizer = temp_data_loader.tokenizer

		# 加载预训练模型
		self.model = ErnieForSequenceClassification.from_pretrained(
			best_model_path,
			num_classes=temp_data_loader.num_classes
		)
		self.model.eval()
		logger.info(f"成功加载最优模型：{best_model_path}")

	@paddle.no_grad()
	def predict_single(self, text):
		"""单条文本意图预测"""
		# 预处理文本
		encoded_inputs = self.tokenizer(
			text=text,
			max_length=self.config.max_seq_len,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			return_token_type_ids=True,
			return_tensors="pd"
		)

		# 预测
		outputs = self.model(
			input_ids=encoded_inputs["input_ids"],
			token_type_ids=encoded_inputs["token_type_ids"],
			attention_mask=encoded_inputs["attention_mask"]
		)
		logits = outputs if len(outputs.shape) == 2 else outputs[0]

		# 解析结果
		pred_id = paddle.argmax(logits, axis=-1).numpy()[0]
		pred_label = self.id2label[pred_id]
		pred_probs = paddle.nn.functional.softmax(logits, axis=-1).numpy()[0]

		# 置信度字典
		prob_dict = {
			self.id2label[idx]: float(prob)
			for idx, prob in enumerate(pred_probs)
		}

		return {
			"text": text,
			"pred_label": pred_label,
			"pred_id": int(pred_id),
			"confidence": prob_dict
		}

	def predict_batch(self, texts):
		"""批量文本意图预测"""
		if not texts:
			return []

		# 批量预处理文本
		encoded_inputs = self.tokenizer(
			text=texts,
			max_length=self.config.max_seq_len,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			return_token_type_ids=True,
			return_tensors="pd"
		)

		# 批量预测
		outputs = self.model(
			input_ids=encoded_inputs["input_ids"],
			token_type_ids=encoded_inputs["token_type_ids"],
			attention_mask=encoded_inputs["attention_mask"]
		)
		logits = outputs if len(outputs.shape) == 2 else outputs[0]

		# 解析批量结果
		pred_ids = paddle.argmax(logits, axis=-1).numpy()
		pred_probs = paddle.nn.functional.softmax(logits, axis=-1).numpy()

		results = []
		for idx, text in enumerate(texts):
			pred_id = pred_ids[idx]
			pred_label = self.id2label[pred_id]
			prob_dict = {
				self.id2label[i]: float(pred_probs[idx][i])
				for i in range(len(self.id2label))
			}
			results.append({
				"text": text,
				"pred_label": pred_label,
				"pred_id": int(pred_id),
				"confidence": prob_dict
			})

		return results


# ===================== 训练器类（移除predict方法） =====================
class KUAKEQICTrainer:
	"""KUAKE-QIC意图分类训练器"""

	def __init__(self, config):
		self.config = config  # 保存配置对象
		set_seed(config.seed)

		# 初始化数据加载器（使用config中的参数）
		self.data_loader = KUAKEQICDataLoader(
			data_dir=config.data_dir,
			model_name=config.model_name,
			max_seq_len=config.max_seq_len
		)
		self.data_loader.load_datasets()

		# 创建DataLoader（使用config中的参数）
		self.train_loader = self.data_loader.create_dataloader(
			self.data_loader.train_ds,
			batch_size=config.batch_size,
			is_train=True
		)
		self.dev_loader = self.data_loader.create_dataloader(
			self.data_loader.dev_ds,
			batch_size=config.batch_size,
			is_train=False
		)
		if self.data_loader.test_ds:
			self.test_loader = self.data_loader.create_dataloader(
				self.data_loader.test_ds,
				batch_size=config.batch_size,
				is_train=False
			)

		# 初始化模型（使用config中的参数）
		self.model = ErnieForSequenceClassification.from_pretrained(
			config.model_name,
			num_classes=self.data_loader.num_classes
		)

		# 冻结层：减少训练参数，提升速度
		self._freeze_layers()

		# 初始化优化器和学习率调度器
		self.optimizer, self.lr_scheduler = self._init_optimizer()

		# ========== 加权交叉熵损失 ==========
		# 统计训练集类别分布
		label_counts = {}
		for sample in self.data_loader.train_ds:
			label = sample["label"]
			label_counts[label] = label_counts.get(label, 0) + 1
		# 计算类别权重（公式：total / (num_classes * count)）
		total_samples = sum(label_counts.values())
		class_weights = [total_samples / (self.data_loader.num_classes * label_counts.get(i, 1))
						 for i in range(self.data_loader.num_classes)]
		class_weights = paddle.to_tensor(class_weights, dtype="float32")
		# 使用加权交叉熵损失
		self.loss_fn = paddle.nn.CrossEntropyLoss(weight=class_weights)

		# 训练记录
		self.train_losses = []
		self.dev_losses = []  # 新增：验证损失记录
		self.dev_accs = []
		self.dev_f1s = []
		self.dev_recalls = []  # 新增：验证召回率记录
		self.best_f1 = 0.0

		# 用于混淆矩阵的变量
		self.best_dev_preds = []
		self.best_dev_labels = []

	def _freeze_layers(self):
		"""冻结模型层（embeddings + 前3层encoder）"""
		# 1. 冻结embeddings层
		for param in self.model.ernie.embeddings.parameters():
			param.requires_grad = False

		# 2. 冻结encoder前3层（ERNIE-3.0-medium共6层）
		freeze_layers = 3
		for layer_idx in range(freeze_layers):
			for param in self.model.ernie.encoder.layers[layer_idx].parameters():
				param.requires_grad = False

		# 打印可训练参数占比
		trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		total_params = sum(p.numel() for p in self.model.parameters())
		logger.info(f"可训练参数占比：{trainable_params / total_params:.2%}")

	def _init_optimizer(self):
		"""初始化优化器（使用config中的参数）"""
		# ========== 修复：手动计算训练步数（避免读取sampler的len） ==========
		# 原方式会触发sampler的len计算，改为直接用数据集长度计算
		num_batches_per_epoch = len(self.data_loader.train_ds) // self.config.batch_size
		if len(self.data_loader.train_ds) % self.config.batch_size != 0:
			num_batches_per_epoch += 1  # 向上取整
		num_training_steps = num_batches_per_epoch * self.config.epochs

		# 余弦退火学习率（使用config中的参数）
		lr_scheduler = CosineAnnealingDecay(
			learning_rate=self.config.learning_rate,
			T_max=num_training_steps,
			eta_min=1e-7
		)

		# AdamW优化器（仅优化可训练参数，使用config中的参数）
		optimizer = paddle.optimizer.AdamW(
			learning_rate=lr_scheduler,
			parameters=[p for p in self.model.parameters() if p.requires_grad],
			weight_decay=self.config.weight_decay
		)

		return optimizer, lr_scheduler

	def train_one_epoch(self, epoch):
		"""训练一个epoch（彻底修复维度问题）"""
		self.model.train()
		total_loss = 0.0
		progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

		for batch in progress_bar:
			# tuple解包
			input_ids, token_type_ids, attention_mask, labels = batch

			# 强制确保所有输入是paddle tensor且类型正确
			input_ids = paddle.cast(input_ids, dtype="int64")
			token_type_ids = paddle.cast(token_type_ids, dtype="int64")
			attention_mask = paddle.cast(attention_mask, dtype="int64")
			labels = paddle.cast(labels, dtype="int64")

			# 核心修复：确保labels是一维张量（最终保障）
			if len(labels.shape) > 1:
				labels = paddle.squeeze(labels, axis=-1)
			# 如果是0维，转为一维
			elif len(labels.shape) == 0:
				labels = paddle.unsqueeze(labels, axis=0)

			# 前向传播（显式传入所有参数，避免模型输出异常）
			outputs = self.model(
				input_ids=input_ids,
				token_type_ids=token_type_ids,
				attention_mask=attention_mask
			)
			# 确保outputs是logits（二维）
			logits = outputs if len(outputs.shape) == 2 else outputs[0]

			# 维度校验日志（方便排查）
			if epoch == 0 and progress_bar.n == 0:
				logger.info(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

			# 计算损失（此时axis=-1，维度匹配）
			loss = self.loss_fn(logits, labels)
			total_loss += loss.item()

			# 反向传播
			loss.backward()
			self.optimizer.step()
			self.lr_scheduler.step()
			self.optimizer.clear_grad()

			# 更新进度条
			progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

		avg_loss = total_loss / len(self.train_loader)
		self.train_losses.append(avg_loss)
		logger.info(f"Epoch {epoch + 1} 平均训练损失：{avg_loss:.4f}")
		return avg_loss

	@paddle.no_grad()
	def evaluate(self, dataloader, desc="Validation", save_preds=False):
		"""验证/测试模型（同步修复维度）"""
		self.model.eval()
		all_preds = []
		all_labels = []
		total_loss = 0.0

		for batch in tqdm(dataloader, desc=desc):
			# tuple解包
			input_ids, token_type_ids, attention_mask, labels = batch

			# 类型转换和维度修复
			input_ids = paddle.cast(input_ids, dtype="int64")
			token_type_ids = paddle.cast(token_type_ids, dtype="int64")
			attention_mask = paddle.cast(attention_mask, dtype="int64")
			labels = paddle.cast(labels, dtype="int64")

			if len(labels.shape) > 1:
				labels = paddle.squeeze(labels, axis=-1)

			# 前向传播
			outputs = self.model(
				input_ids=input_ids,
				token_type_ids=token_type_ids,
				attention_mask=attention_mask
			)
			logits = outputs if len(outputs.shape) == 2 else outputs[0]

			# 计算验证损失
			loss = self.loss_fn(logits, labels)
			total_loss += loss.item()

			# 预测（axis=-1，兼容所有版本）
			preds = paddle.argmax(logits, axis=-1).numpy()
			labels = labels.numpy()

			all_preds.extend(preds)
			all_labels.extend(labels)

		# 计算平均验证损失
		avg_loss = total_loss / len(dataloader)

		# 计算指标
		accuracy = accuracy_score(all_labels, all_preds)
		macro_f1 = f1_score(all_labels, all_preds, average="macro")
		micro_f1 = f1_score(all_labels, all_preds, average="micro")
		macro_recall = recall_score(all_labels, all_preds, average="macro")  # 计算召回率

		# 打印结果
		logger.info(f"{desc} 平均损失：{avg_loss:.4f}")
		logger.info(f"{desc} 准确率：{accuracy:.4f}")
		logger.info(f"{desc} 宏F1：{macro_f1:.4f}")
		logger.info(f"{desc} 微F1：{micro_f1:.4f}")
		logger.info(f"{desc} 宏召回率：{macro_recall:.4f}")

		# 如果需要保存预测结果（用于绘制最优模型的混淆矩阵）
		if save_preds:
			self.best_dev_preds = all_preds
			self.best_dev_labels = all_labels

		return accuracy, macro_f1, micro_f1, avg_loss, macro_recall

	def plot_training_curves(self, save_dir):
		"""绘制训练/验证曲线"""
		epochs = range(1, len(self.train_losses) + 1)

		# 创建2x2的子图布局
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

		# 1. 训练/验证损失曲线
		ax1.plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
		ax1.plot(epochs, self.dev_losses, 'r-', label='验证损失', linewidth=2)
		ax1.set_title('训练与验证损失曲线', fontsize=14, fontweight='bold')
		ax1.set_xlabel('Epoch', fontsize=12)
		ax1.set_ylabel('损失值', fontsize=12)
		ax1.legend(fontsize=10)
		ax1.grid(True, alpha=0.3)

		# 2. 验证准确率曲线
		ax2.plot(epochs, self.dev_accs, 'g-', label='验证准确率', linewidth=2)
		ax2.set_title('验证准确率曲线', fontsize=14, fontweight='bold')
		ax2.set_xlabel('Epoch', fontsize=12)
		ax2.set_ylabel('准确率', fontsize=12)
		ax2.legend(fontsize=10)
		ax2.grid(True, alpha=0.3)

		# 3. 验证F1曲线
		ax3.plot(epochs, self.dev_f1s, 'orange', label='验证宏F1', linewidth=2)
		ax3.set_title('验证宏F1曲线', fontsize=14, fontweight='bold')
		ax3.set_xlabel('Epoch', fontsize=12)
		ax3.set_ylabel('F1值', fontsize=12)
		ax3.legend(fontsize=10)
		ax3.grid(True, alpha=0.3)

		# 4. 验证召回率曲线
		ax4.plot(epochs, self.dev_recalls, 'purple', label='验证宏召回率', linewidth=2)
		ax4.set_title('验证宏召回率曲线', fontsize=14, fontweight='bold')
		ax4.set_xlabel('Epoch', fontsize=12)
		ax4.set_ylabel('召回率', fontsize=12)
		ax4.legend(fontsize=10)
		ax4.grid(True, alpha=0.3)

		plt.tight_layout()
		plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
		plt.close()
		logger.info(f"训练曲线已保存至：{os.path.join(save_dir, 'training_curves.png')}")

	def plot_confusion_matrix(self, save_dir):
		"""绘制归一化的混淆矩阵"""
		# 计算混淆矩阵
		cm = confusion_matrix(self.best_dev_labels, self.best_dev_preds)

		# 归一化（按行归一化，每行和为1）
		cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		# 创建混淆矩阵图
		plt.figure(figsize=(14, 12))
		sns.heatmap(
			cm_normalized,
			annot=True,
			fmt='.2f',
			cmap='Blues',
			xticklabels=self.data_loader.label_list,
			yticklabels=self.data_loader.label_list,
			cbar_kws={'label': '归一化概率'}
		)

		plt.title('验证集混淆矩阵（归一化）', fontsize=16, fontweight='bold', pad=20)
		plt.xlabel('预测标签', fontsize=14)
		plt.ylabel('真实标签', fontsize=14)
		plt.xticks(rotation=45, ha='right')
		plt.yticks(rotation=0)

		# 调整布局并保存
		plt.tight_layout()
		save_path = os.path.join(save_dir, 'confusion_matrix.png')
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close()
		logger.info(f"混淆矩阵已保存至：{save_path}")

	def train(self):
		"""主训练流程（使用config中的参数）"""
		logger.info("开始训练KUAKE-QIC意图分类模型...")
		# 打印配置信息
		logger.info(self.config)

		for epoch in range(self.config.epochs):
			# 训练
			self.train_one_epoch(epoch)

			# 验证
			dev_acc, dev_macro_f1, dev_micro_f1, dev_loss, dev_recall = self.evaluate(self.dev_loader)
			self.dev_losses.append(dev_loss)  # 记录验证损失
			self.dev_accs.append(dev_acc)
			self.dev_f1s.append(dev_macro_f1)
			self.dev_recalls.append(dev_recall)  # 记录验证召回率

			# 保存最优模型（使用config中的参数）
			if dev_macro_f1 > self.best_f1:
				self.best_f1 = dev_macro_f1
				save_dir = os.path.join(self.config.output_dir, "best_model")
				os.makedirs(save_dir, exist_ok=True)
				self.model.save_pretrained(save_dir)
				self.data_loader.tokenizer.save_pretrained(save_dir)
				self.save_training_log(save_dir)
				# 保存最优模型的预测结果（用于混淆矩阵）
				self.evaluate(self.dev_loader, desc="Best Validation", save_preds=True)
				logger.info(f"最优模型已保存（宏F1：{self.best_f1:.4f}）")

		# 绘制并保存训练曲线（使用config中的参数）
		self.plot_training_curves(self.config.output_dir)

		# 绘制并保存混淆矩阵（使用config中的参数）
		if self.best_dev_preds and self.best_dev_labels:
			self.plot_confusion_matrix(self.config.output_dir)

		# 测试集评估（如有，使用config中的参数）
		if self.data_loader.test_ds:
			logger.info("开始测试最优模型...")
			self.model = ErnieForSequenceClassification.from_pretrained(
				os.path.join(self.config.output_dir, "best_model"),
				num_classes=self.data_loader.num_classes
			)
			test_acc, test_macro_f1, test_micro_f1, test_loss, test_recall = self.evaluate(self.test_loader,
																						   desc="Test")
			logger.info(f"测试集 准确率：{test_acc:.4f}，宏F1：{test_macro_f1:.4f}，宏召回率：{test_recall:.4f}")

		# 保存最终日志（使用config中的参数）
		self.save_training_log(self.config.output_dir)
		logger.info(f"训练完成！最优验证集宏F1：{self.best_f1:.4f}")

	def save_training_log(self, save_dir):
		"""保存训练日志（包含配置信息）"""
		log_data = {
			"config": self.config.__dict__,  # 保存配置参数
			"train_losses": self.train_losses,
			"dev_losses": self.dev_losses,  # 新增：保存验证损失
			"dev_accs": self.dev_accs,
			"dev_f1s": self.dev_f1s,
			"dev_recalls": self.dev_recalls,  # 新增：保存验证召回率
			"best_f1": self.best_f1,
			"label2id": self.data_loader.label2id,
			"id2label": self.data_loader.id2label
		}
		os.makedirs(save_dir, exist_ok=True)
		with open(os.path.join(save_dir, "training_log.json"), "w", encoding="utf-8") as f:
			json.dump(log_data, f, ensure_ascii=False, indent=2)

