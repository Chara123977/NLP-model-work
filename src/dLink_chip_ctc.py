import os
import json
import numpy as np

import paddle
import matplotlib.pyplot as plt
from tqdm import tqdm
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 忽略DeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 导入自定义模块（请确保路径正确）
from config.chip_config import Config
from models.med_chip_mode import CHIPCTCModel
from data.med_chip_data_loader import CHIPCTCDataLoader


class CHIPCTCClassifier:
    """CHIP-CTC 医疗文本分类器"""

    def __init__(self):
        """初始化分类器：加载配置、设置matplotlib中文"""
        # 加载配置
        self.config = Config()
        # 初始化核心属性
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None

        # matplotlib 中文配置
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]

    def train(self, load_from_path=None):
        """
        对外暴露的训练方法：启动完整的模型训练流程
        Args:
            load_from_path: 从指定路径加载已保存的模型继续训练（可选）
        """
        # 创建保存目录
        os.makedirs(self.config.save_dir, exist_ok=True)

        try:
            # 1. 加载数据
            data_loader_instance = CHIPCTCDataLoader(self.config)
            train_loader, dev_loader, test_loader = data_loader_instance.build_dataloaders()
            self.tokenizer, self.label2id, self.test_dataset = data_loader_instance.get_additional_info()
            self.id2label = {v: k for k, v in self.label2id.items()}

            # 2. 构建模型
            num_classes = len(self.label2id)
            model_components = CHIPCTCModel(self.config, num_classes).build(num_classes, load_from_path)
            self.model, optimizer, loss_fn, lr_scheduler = model_components
            print(f"\n✅ 成功加载模型：{self.config.model_name}（分类数：{num_classes}）")
            if load_from_path:
                print(f"✅ 从路径继续训练：{load_from_path}")

        except Exception as e:
            print(f"\n❌ 数据/模型加载失败：{e}")
            return

        # 3. 开始训练
        print("\n========== 开始训练 ==========")
        best_macro_f1 = 0.0
        patience_counter = 0
        train_losses = []
        dev_losses = []
        dev_macro_f1s = []

        for epoch in range(self.config.epochs):
            print(f"\n===== Epoch {epoch + 1}/{self.config.epochs} =====")

            # 训练一轮
            train_loss = self._train_one_epoch(self.model, train_loader, loss_fn, optimizer)
            train_losses.append(train_loss)

            # 更新学习率
            lr_scheduler.step()
            current_lr = lr_scheduler.get_lr()

            # 验证评估
            dev_loss, dev_acc, dev_macro_f1, dev_micro_f1 = self._evaluate_model(self.model, dev_loader, loss_fn)
            dev_losses.append(dev_loss)
            dev_macro_f1s.append(dev_macro_f1)

            # 打印本轮结果
            print(f"训练损失：{train_loss:.4f} | 当前学习率：{current_lr:.6f}")
            print(f"验证损失：{dev_loss:.4f} | 验证准确率：{dev_acc:.4f}")
            print(f"验证宏F1：{dev_macro_f1:.4f} | 验证微F1：{dev_micro_f1:.4f}")

            # 保存最优模型
            if dev_macro_f1 > best_macro_f1 + self.config.min_delta:
                best_macro_f1 = dev_macro_f1
                patience_counter = 0
                model_save_path = os.path.join(self.config.save_dir, "best_model")
                self.model.save_pretrained(model_save_path)
                self.tokenizer.save_pretrained(model_save_path)
                with open(os.path.join(model_save_path, "label2id.json"), "w", encoding="utf-8") as f:
                    json.dump(self.label2id, f, ensure_ascii=False, indent=2)
                print(f"✅ 保存最优模型（宏F1：{best_macro_f1:.4f}）")
            else:
                patience_counter += 1
                print(f"⚠️  早停计数器：{patience_counter}/{self.config.patience}")
                if patience_counter >= self.config.patience:
                    print("✅ 早停触发，停止训练")
                    break

        # 4. 评估最优模型
        print("\n========== 评估最优模型（验证集） ==========")
        best_model_path = os.path.join(self.config.save_dir, "best_model")
        best_model = ErnieForSequenceClassification.from_pretrained(best_model_path)
        dev_acc, dev_macro_f1, dev_micro_f1 = self._evaluate_with_report(best_model, dev_loader, loss_fn)
        print(f"\n✅ 验证集最终结果：")
        print(f"准确率：{dev_acc:.4f} | 宏F1：{dev_macro_f1:.4f} | 微F1：{dev_micro_f1:.4f}")

        # 5. 生成测试集预测
        print("\n========== 生成测试集预测结果 ==========")
        self._predict_test_set(best_model, test_loader, self.test_dataset)

        # 6. 绘制训练曲线
        self._plot_training_curves(train_losses, dev_losses, dev_macro_f1s)

        print("\n🎉 训练流程全部完成！")

    def predict(self, text: str or list, model_path: str = None):
        """
        对外暴露的预测方法：加载模型并预测给定文本
        Args:
            text: 单个文本字符串 或 文本列表
            model_path: 模型路径，默认使用配置中的最优模型路径
        Returns:
            预测结果（单个文本返回str，列表返回list[str]）
        """
        # 1. 设置默认模型路径
        if model_path is None:
            model_path = os.path.join(self.config.save_dir, "best_model")

        # 2. 加载模型和tokenizer（首次预测时加载）
        if self.model is None or self.tokenizer is None:
            try:
                # 加载tokenizer
                self.tokenizer = ErnieTokenizer.from_pretrained(model_path)
                # 加载标签映射
                label2id_path = os.path.join(model_path, "label2id.json")
                with open(label2id_path, "r", encoding="utf-8") as f:
                    self.label2id = json.load(f)
                self.id2label = {v: k for k, v in self.label2id.items()}
                # 加载模型
                num_classes = len(self.label2id)
                self.model = ErnieForSequenceClassification.from_pretrained(model_path, num_classes=num_classes)
                self.model.eval()
                print(f"✅ 成功加载预测模型：{model_path}")
            except Exception as e:
                print(f"\n❌ 模型加载失败：{e}")
                return None

        # 3. 处理输入文本（统一转为列表）
        is_single = False
        if isinstance(text, str):
            is_single = True
            text = [text]

        # 4. 文本预处理 + 预测
        predictions = []
        with paddle.no_grad():
            for t in text:
                # 文本编码：显式添加return_attention_mask=True
                encoded_input = self.tokenizer(
                    t,
                    max_len=self.config.max_seq_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pd',
                    return_attention_mask=True  # 关键修复：强制返回attention_mask
                )
                # 模型预测
                logits = self.model(
                    input_ids=encoded_input['input_ids'],
                    token_type_ids=encoded_input['token_type_ids'],
                    attention_mask=encoded_input['attention_mask']
                )
                # 获取预测标签
                pred_id = paddle.argmax(logits, axis=1).item()
                pred_label = self.id2label[pred_id]
                predictions.append(pred_label)

        # 5. 返回结果（单个文本返回字符串，列表返回列表）
        if is_single:
            return predictions[0]
        return predictions

    # ===================== 内部私有方法（不对外暴露） =====================
    def _train_one_epoch(self, model, train_loader, loss_fn, optimizer):
        """单轮训练（内部方法）"""
        model.train()
        total_loss = 0.0
        total_samples = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="训练", ncols=100)

        for batch_idx, (input_ids, token_type_ids, attention_mask, labels) in pbar:
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            avg_loss = total_loss / total_samples
            pbar.set_postfix({"训练损失": f"{avg_loss:.4f}"})

        pbar.close()
        return total_loss / total_samples

    @paddle.no_grad()
    def _evaluate_model(self, model, dev_loader, loss_fn):
        """验证模型（内部方法）"""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        pbar = tqdm(enumerate(dev_loader), total=len(dev_loader), desc="验证", ncols=100)

        for batch_idx, batch in pbar:
            input_ids, token_type_ids, attention_mask, labels = batch
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = paddle.argmax(logits, axis=1)
            total_correct += paddle.sum(preds == labels).item()
            total_samples += len(labels)

            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            pbar.set_postfix({"验证损失": f"{avg_loss:.4f}", "准确率": f"{accuracy:.4f}"})

        pbar.close()
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)

        return avg_loss, accuracy, macro_f1, micro_f1

    @paddle.no_grad()
    def _evaluate_with_report(self, model, dev_loader, loss_fn):
        """详细评估（内部方法）"""
        model.eval()
        all_preds = []
        all_labels = []
        pbar = tqdm(enumerate(dev_loader), total=len(dev_loader), desc="详细评估", ncols=100)

        for batch_idx, batch in pbar:
            input_ids, token_type_ids, attention_mask, labels = batch
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            preds = paddle.argmax(logits, axis=1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            pbar.set_postfix({"已处理批次": f"{batch_idx + 1}/{len(dev_loader)}"})

        pbar.close()
        all_preds_names = [self.id2label[p] for p in all_preds]
        all_labels_names = [self.id2label[l] for l in all_labels]

        print("\n==================== 验证集评估报告 ====================")
        print(classification_report(
            all_labels_names,
            all_preds_names,
            target_names=sorted(self.label2id.keys()),
            zero_division=0
        ))

        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_labels_names, all_preds_names, labels=sorted(self.label2id.keys()))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("CHIP-CTC多分类混淆矩阵（验证集）")
        plt.colorbar()
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        tick_marks = np.arange(len(self.label2id))
        plt.xticks(tick_marks, sorted(self.label2id.keys()), rotation=45)
        plt.yticks(tick_marks, sorted(self.label2id.keys()))

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig("./chip_ctc_confusion_matrix.png")
        plt.show()

        accuracy = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)

        return accuracy, macro_f1, micro_f1

    @paddle.no_grad()
    def _predict_test_set(self, model, test_loader, test_dataset):
        """测试集批量预测（内部方法）"""
        model.eval()
        predictions = []
        sample_idx = 0
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="测试集预测", ncols=100)

        for batch_idx, batch in pbar:
            input_ids, token_type_ids, attention_mask = batch
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            preds = paddle.argmax(logits, axis=1).numpy().tolist()

            batch_size = len(preds)
            batch_sample_ids = test_dataset.test_sample_ids[sample_idx: sample_idx + batch_size]
            sample_idx += batch_size

            for sample_id, pred_idx in zip(batch_sample_ids, preds):
                predictions.append({
                    "id": sample_id,
                    "label": self.id2label[pred_idx]
                })

            pbar.set_postfix({"已预测样本": f"{len(predictions)}/{len(test_dataset)}"})

        pbar.close()
        # 保存预测结果
        with open(self.config.test_pred_save_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
        print(f"\n✅ 测试集预测结果已保存到：{self.config.test_pred_save_path}")
        print(f"✅ 共预测 {len(predictions)} 条无标注样本")

    def _plot_training_curves(self, train_losses, dev_losses, dev_macro_f1s):
        """绘制训练曲线（内部方法）"""
        plt.switch_backend('Agg')
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="训练损失", marker='o', markersize=4)
        plt.plot(dev_losses, label="验证损失", marker='s', markersize=4)
        plt.title("训练/验证损失曲线")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)

        # 宏F1曲线
        plt.subplot(1, 2, 2)
        plt.plot(dev_macro_f1s, label="验证宏F1", color="orange", marker='^', markersize=4)
        plt.title("验证集宏F1曲线")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1 Score")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        curve_save_path = "./chip_ctc_training_curve.png"
        plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 训练曲线已保存至：{curve_save_path}")


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 1. 初始化分类器
    classifier = CHIPCTCClassifier()

    # 2. 启动训练（调用对外的train方法）
    # classifier.train()

    # 3. 从已保存模型继续训练（示例）
    # classifier.train(load_from_path="runs/best_model")

    # 4. 预测示例（训练完成后调用）
    # 单文本预测
    text = "4）既往患有骨髓增生异常综合征（Myelodysplastic syndrome，MDS）的患者，或者未定型的急性白血病患者。"#"患者出现发热、咳嗽、胸闷等症状，持续3天"
    pred_label = classifier.predict(text)
    print(f"\n文本：{text}")
    print(f"预测标签：{pred_label}")

    # 多文本预测
    texts = [
        "血压升高，头晕乏力，伴有心悸",
        "腹痛、腹泻，每日5-6次，大便稀水样"
    ]
    pred_labels = classifier.predict(texts)
    print(f"\n多文本预测结果：")
    for t, l in zip(texts, pred_labels):
        print(f"文本：{t} | 预测标签：{l}")