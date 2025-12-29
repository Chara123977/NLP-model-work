import paddle

# ===================== 训练参数配置 =====================
class Config:
    def __init__(self):
        self.data_dir = "datasets/CHIP-CTC"  # 你的数据路径
        self.model_name = "ernie-3.0-base-zh"
        self.batch_size = 32
        self.epochs = 15
        self.init_lr = 1e-5          # 学习率略微降低
        self.patience = 3            # 早停耐心值
        self.min_delta = 0.001
        self.warmup_proportion = 0.1
        self.max_seq_len = 128
        self.lr_step_size = 2
        self.lr_gamma = 0.8
        self.min_delta = 0.001
        self.save_dir = "runs"
        self.device = paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
        self.test_pred_save_path = "./chip_ctc_test_predictions.json"  # 新增：测试集预测结果保存路径