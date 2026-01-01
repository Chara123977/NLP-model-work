# ===================== 配置参数 =====================
import paddle


class Config:
    def __init__(self):
        # 数据路径（请替换为你的CMeEE-V2数据集路径）
        self.data_dir = "datasets/CMeEE-V2"
        # 预训练模型名称
        self.model_name = "ernie-3.0-medium-zh"
        # 训练参数
        self.batch_size = 16
        self.epochs = 2
        self.learning_rate = 2e-5
        self.max_seq_len = 128
        # 保存路径
        self.save_dir = "cmeee_model"
        # 设备（自动识别GPU/CPU）
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"

        # ========== 标签映射（匹配数据中的type字段） ==========
        self.LABEL_LIST = [
            'O', 'B-dis', 'I-dis', 'B-sym', 'I-sym',
            'B-pro', 'I-pro', 'B-drug', 'I-drug',
            'B-body', 'I-body', 'B-exam', 'I-exam',
            'B-val', 'I-val', 'B-per', 'I-per'
        ]

        self.LABEL2ID = {label: idx for idx, label in enumerate(self.LABEL_LIST)}
        self.ID2LABEL = {idx: label for idx, label in enumerate(self.LABEL_LIST)}
        self.num_classes = len(self.LABEL_LIST)  # 显式定义类别数
