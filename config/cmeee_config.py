# ===================== 配置参数 =====================
import os
import paddle
import yaml

# class Config:
#     def __init__(self):
#         # 数据路径（请替换为你的CMeEE-V2数据集路径）
#         self.data_dir = "datasets/CMeEE-V2"
#         # 预训练模型名称
#         self.model_name = "ernie-3.0-medium-zh"
#         # 训练参数
#         self.batch_size = 16
#         self.epochs = 2
#         self.learning_rate = 2e-5
#         self.max_seq_len = 128
#         # 保存路径
#         self.save_dir = "cmeee_model"
#         # 设备（自动识别GPU/CPU）
#         self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
# 
#         # ========== 标签映射（匹配数据中的type字段） ==========
#         self.LABEL_LIST = [
#             'O', 'B-dis', 'I-dis', 'B-sym', 'I-sym',
#             'B-pro', 'I-pro', 'B-drug', 'I-drug',
#             'B-body', 'I-body', 'B-exam', 'I-exam',
#             'B-val', 'I-val', 'B-per', 'I-per'
#         ]
# 
#         self.LABEL2ID = {label: idx for idx, label in enumerate(self.LABEL_LIST)}
#         self.ID2LABEL = {idx: label for idx, label in enumerate(self.LABEL_LIST)}
#         self.num_classes = len(self.LABEL_LIST)  # 显式定义类别数


import os
import yaml


class Config:
    def __init__(self, path=None):
        """
        初始化配置类。

        Args:
             path (str, optional): YAML 配置文件的路径。如果为 None，则使用默认配置。
        """
        # --- 1. 设置默认参数 ---
        self.data_dir = "datasets/CMeEE-V2"
        self.save_dir = "cmeee_model"
        self.model_name = "ernie-3.0-medium-zh"
        self.num_classes = 17
        self.batch_size = 16
        self.epochs = 2
        self.learning_rate = 2e-5
        self.max_seq_len = 128
        self.device = "auto"

        # 标签列表 (BIO scheme for 8 entity types)
        self.LABEL_LIST = [
            "O",
            "B-dis", "I-dis",
            "B-sym", "I-sym",
            "B-pro", "I-pro",
            "B-drug", "I-drug",
            "B-body", "I-body",
            "B-exam", "I-exam",
            "B-val", "I-val",
            "B-per", "I-per"
        ]
        self.LABEL2ID = {label: idx for idx, label in enumerate(self.LABEL_LIST)}
        self.ID2LABEL = {idx: label for idx, label in enumerate(self.LABEL_LIST)}

        # --- 2. 如果提供了  path，则加载并覆盖默认参数 ---
        if path is not None:
            self._load_from_yaml(path)

    def _load_from_yaml(self, path):
        """
        从 YAML 文件加载配置并更新当前实例的属性。

        Args:
             path (str): YAML 配置文件的路径。

        Raises:
            FileNotFoundError: 如果指定的 YAML 文件不存在。
            yaml.YAMLError: 如果 YAML 文件格式不正确。
        """
        if not os.path.exists( path):
            raise FileNotFoundError(f"配置文件未找到: { path}")

        with open( path, 'r', encoding='utf-8') as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"YAML 文件解析错误: {e}")

        # 使用 YAML 文件中的值覆盖现有配置
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"警告: 配置文件中包含未知参数 '{key}'，已忽略。")