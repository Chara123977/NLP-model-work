import paddle
from paddlenlp.transformers import ErnieForSequenceClassification

class CHIPCTCModel:
    """CHIP-CTC 文本分类模型类，封装模型构建、优化器、损失函数等逻辑"""

    def __init__(self, config, num_classes=44):
        """
        初始化模型类
        Args:
            config: 配置对象，需包含以下属性：
                - model_name: 预训练模型名称（如 "ernie-3.0-base-zh"）
                - init_lr: 初始学习率
                - lr_step_size: 学习率衰减步长
                - lr_gamma: 学习率衰减系数
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.lr_scheduler = None
        self.num_classes = None

    def build(self, num_classes):
        """
        构建完整的模型、优化器、损失函数和学习率调度器
        Args:
            num_classes: 分类任务的类别数量
        Returns:
            tuple: (model, optimizer, loss_fn, lr_scheduler)
        """
        self.num_classes = num_classes

        # 1. 构建ERNIE序列分类模型
        print(f"\n📌 开始构建模型：{self.config.model_name}")
        print(f"📌 分类类别数：{num_classes}")

        self.model = ErnieForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_classes=num_classes,
            attention_probs_dropout_prob=0.1,  # 注意力层dropout
            hidden_dropout_prob=0.1  # 隐藏层dropout
        )

        # 2. 构建学习率调度器（StepDecay）
        self.lr_scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=self.config.init_lr,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma
        )

        # 3. 构建AdamW优化器
        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=0.01,  # L2正则化系数
            # 对bias和layer_norm.weight不应用权重衰减
            apply_decay_param_fun=lambda x: x not in ["bias", "layer_norm.weight"]
        )

        # 4. 构建损失函数（交叉熵损失）
        self.loss_fn = paddle.nn.CrossEntropyLoss()

        print("✅ 模型、优化器、损失函数构建完成！")
        return self.model, self.optimizer, self.loss_fn, self.lr_scheduler

    def get_model(self):
        """获取构建好的模型实例"""
        if self.model is None:
            raise RuntimeError("请先调用 build(num_classes) 方法构建模型")
        return self.model

    def get_optimizer(self):
        """获取构建好的优化器实例"""
        if self.optimizer is None:
            raise RuntimeError("请先调用 build(num_classes) 方法构建模型")
        return self.optimizer

    def get_loss_fn(self):
        """获取构建好的损失函数实例"""
        if self.loss_fn is None:
            raise RuntimeError("请先调用 build(num_classes) 方法构建模型")
        return self.loss_fn

    def get_lr_scheduler(self):
        """获取构建好的学习率调度器实例"""
        if self.lr_scheduler is None:
            raise RuntimeError("请先调用 build(num_classes) 方法构建模型")
        return self.lr_scheduler

    def update_learning_rate(self):
        """更新学习率（训练时每个epoch调用）"""
        if self.lr_scheduler is None:
            raise RuntimeError("请先调用 build(num_classes) 方法构建模型")
        self.lr_scheduler.step()
        current_lr = self.lr_scheduler.get_lr()
        print(f"🔄 学习率已更新：{current_lr:.6f}")
        return current_lr