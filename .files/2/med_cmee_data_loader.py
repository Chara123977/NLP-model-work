import os
from paddlenlp.transformers import ErnieTokenizer
from paddlenlp.data import Pad
from paddle.io import DataLoader
from typing import List, Tuple

from data.med_cmeee_datasets import CMeEEV2Dataset


# 假设你的CMeEEV2Dataset和config已经定义，这里保留原有逻辑
# 如果还未定义CMeEEV2Dataset，文末会补充基础实现示例

class CMeEEDataLoader:
	"""CMeEE-V2数据集统一加载器，整合数据加载、批处理、DataLoader构建"""

	def __init__(self, config):
		"""
		初始化数据加载器
		:param config: 配置对象（包含model_name、data_dir、batch_size、LABEL2ID等）
		"""
		self.config = config
		self.tokenizer = self.load_tokenizer()  # 加载tokenizer
		self.dataset_dict = {}  # 存储训练/验证/测试数据集
		self.dataloader_dict = {}  # 存储训练/验证/测试DataLoader

	def load_tokenizer(self) -> ErnieTokenizer:
		"""内部方法：加载ErnieTokenizer，处理兼容性"""
		tokenizer = ErnieTokenizer.from_pretrained(
			self.config.model_name,
			use_fast=False  # 禁用fast tokenizer，避免兼容性问题
		)
		return tokenizer

	def load_datasets(self) -> None:
		"""加载训练、验证、测试数据集，并做空值检查"""
		# 定义数据集路径映射
		dataset_paths = {
			'train': os.path.join(self.config.data_dir, 'CMeEE-V2_train.json'),
			'dev': os.path.join(self.config.data_dir, 'CMeEE-V2_dev.json'),
			'test': os.path.join(self.config.data_dir, 'CMeEE-V2_test.json')
		}

		# 加载每个数据集
		for ds_type, ds_path in dataset_paths.items():
			self.dataset_dict[ds_type] = CMeEEV2Dataset(
				ds_path,
				self.tokenizer,
				self.config
			)
			# 空数据集检查
			if len(self.dataset_dict[ds_type]) == 0:
				raise ValueError(f"{ds_type}数据集为空！请检查路径：{ds_path}")
			print(f"{ds_type}数据集大小：{len(self.dataset_dict[ds_type])}")

	def _batchify_fn(self, batch: List[Tuple]) -> Tuple:
		"""内部批处理函数：对batch数据进行padding"""
		input_ids, attention_mask, label_ids = list(zip(*batch))
		return (
			Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype='int64')(input_ids),
			Pad(axis=0, pad_val=0, dtype='int64')(attention_mask),
			Pad(axis=0, pad_val=self.config.LABEL2ID['O'], dtype='int64')(label_ids)
		)

	def build_dataloaders(self) -> None:
		"""构建训练/验证/测试DataLoader"""
		# 定义每个DataLoader的配置
		dataloader_configs = {
			'train': {'shuffle': True, 'drop_last': False},
			'dev': {'shuffle': False, 'drop_last': False},
			'test': {'shuffle': False, 'drop_last': False}
		}

		# 构建每个DataLoader
		for ds_type, cfg in dataloader_configs.items():
			self.dataloader_dict[ds_type] = DataLoader(
				self.dataset_dict[ds_type],
				batch_size=self.config.batch_size,
				shuffle=cfg['shuffle'],
				collate_fn=self._batchify_fn,
				num_workers=0,
				drop_last=cfg['drop_last']
			)

	def get_dataloader(self, ds_type: str) -> DataLoader:
		"""
		获取指定类型的DataLoader（train/dev/test）
		:param ds_type: 数据集类型，可选 'train'/'dev'/'test'
		:return: 对应的DataLoader
		"""
		if ds_type not in self.dataloader_dict:
			raise ValueError(f"未找到{ds_type}类型的DataLoader，请先调用build_dataloaders()")
		return self.dataloader_dict[ds_type]