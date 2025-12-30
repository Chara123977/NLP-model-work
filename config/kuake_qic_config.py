class Config:
	"""KUAKE-QIC模型训练配置类"""

	def __init__(self):
		# 数据相关配置
		self.data_dir = "datasets/KUAKE_QIC"  # JSON数据集目录
		self.output_dir = "output/kuake_qic"  # 模型输出目录

		# 模型相关配置
		self.model_name = "ernie-3.0-medium-zh"  # 预训练模型名称
		self.max_seq_len = 128  # 最大序列长度

		# 训练相关配置
		self.batch_size = 16  # 批次大小（低显存推荐8）
		self.epochs = 10  # 训练轮数
		self.learning_rate = 1e-5  # 学习率
		self.weight_decay = 0.001  # 权重衰减
		self.seed = 42  # 随机种子
