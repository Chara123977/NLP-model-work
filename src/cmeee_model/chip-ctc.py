import os
import json
import numpy as np

import paddle
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm进度条库
from paddlenlp.transformers import ErnieForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, f1_score
# 注意：如果你的 config.params 和 data.med_datasets 模块路径不同，请自行调整
from config.chip_config import Config
from models import CHIPCTCModel
from models import CHIPCTCDataLoader

import warnings

# 忽略DeprecationWarning（仅隐藏警告，不影响功能）
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================== matplotlib 中文配置 =====================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]  # 优先使用中文字体
plt.rcParams["font.size"] = 10  # 设置默认字体大小
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]  # 指定默认中文字体

config = Config()

# ===================== 训练/验证/测试函数 =====================
def train_one_epoch(model, train_loader, loss_fn, optimizer):
    """单轮训练（添加tqdm进度条）"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    # 初始化训练进度条：指定总批次数，设置描述和宽度
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="训练", ncols=100)

    for batch_idx, (input_ids, token_type_ids, attention_mask, labels) in pbar:
        logits = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
        avg_loss = total_loss / total_samples

        # 更新进度条显示：实时展示当前平均损失
        pbar.set_postfix({"训练损失": f"{avg_loss:.4f}"})

    pbar.close()
    return total_loss / total_samples


@paddle.no_grad()
def evaluate_model(model, dev_loader, loss_fn):
    """验证模型（添加tqdm进度条）"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    # 初始化验证进度条
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

        # 实时计算并显示验证损失和准确率
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
def evaluate_with_report(model, dev_loader, loss_fn, label2id):
    """对有标注数据生成评估报告（添加进度条）"""
    model.eval()
    all_preds = []
    all_labels = []
    id2label = {v: k for k, v in label2id.items()}

    # 初始化评估进度条
    pbar = tqdm(enumerate(dev_loader), total=len(dev_loader), desc="详细评估", ncols=100)

    for batch_idx, batch in pbar:
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        preds = paddle.argmax(logits, axis=1)

        all_preds.extend(preds.numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

        # 实时显示进度
        pbar.set_postfix({"已处理批次": f"{batch_idx + 1}/{len(dev_loader)}"})

    pbar.close()
    all_preds_names = [id2label[p] for p in all_preds]
    all_labels_names = [id2label[l] for l in all_labels]

    print("\n==================== 验证集评估报告 ====================")
    print(classification_report(
        all_labels_names,
        all_preds_names,
        target_names=sorted(label2id.keys()),
        zero_division=0
    ))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels_names, all_preds_names, labels=sorted(label2id.keys()))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("CHIP-CTC多分类混淆矩阵（验证集）")
    plt.colorbar()
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    tick_marks = np.arange(len(label2id))
    plt.xticks(tick_marks, sorted(label2id.keys()), rotation=45)
    plt.yticks(tick_marks, sorted(label2id.keys()))

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
def predict_test_set(model, test_loader, label2id, save_path, test_dataset):
    """对无标注测试集生成预测结果（添加进度条）"""
    model.eval()
    id2label = {v: k for k, v in label2id.items()}
    predictions = []
    sample_idx = 0  # 用于匹配test_dataset中的sample_id

    print("\n开始对无标注测试集生成预测结果...")
    # 初始化预测进度条
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="测试集预测", ncols=100)

    for batch_idx, batch in pbar:
        input_ids, token_type_ids, attention_mask = batch
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        preds = paddle.argmax(logits, axis=1).numpy().tolist()

        # 通过索引匹配sample_id
        batch_size = len(preds)
        batch_sample_ids = test_dataset.test_sample_ids[sample_idx: sample_idx + batch_size]
        sample_idx += batch_size

        # 匹配样本ID和预测标签
        for sample_id, pred_idx in zip(batch_sample_ids, preds):
            predictions.append({
                "id": sample_id,
                "label": id2label[pred_idx]
            })

        # 实时显示预测进度
        pbar.set_postfix({"已预测样本": f"{len(predictions)}/{len(test_dataset)}"})

    pbar.close()
    # 保存预测结果（按行JSON，符合CHIP-CTC提交格式）
    with open(save_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n✅ 测试集预测结果已保存到：{save_path}")
    print(f"✅ 共预测 {len(predictions)} 条无标注样本")
    return predictions


# ===================== 主训练流程 =====================
def main():
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)

    try:
        # ========== 关键修改1：正确调用CHIPCTCDataLoader ==========
        # 1. 先实例化数据加载类
        data_loader_instance = CHIPCTCDataLoader(config)
        # 2. 调用build_dataloaders()获取3个DataLoader（仅返回train/dev/test loader）
        train_loader, dev_loader, test_loader = data_loader_instance.build_dataloaders()
        # 3. 调用get_additional_info()获取tokenizer/label2id/test_dataset（补充需要的参数）
        tokenizer, label2id, test_dataset = data_loader_instance.get_additional_info()

    except Exception as e:
        print(f"\n❌ 数据加载失败：{e}")
        return

    # 获取分类数
    num_classes = len(label2id)
    # 构建模型、优化器、损失函数、学习率调度器
    model, optimizer, loss_fn, lr_scheduler = CHIPCTCModel(config, num_classes).build(num_classes)
    print(f"\n✅ 成功加载模型：{config.model_name}（分类数：{num_classes}）")

    print("\n========== 开始训练 ==========")
    best_macro_f1 = 0.0
    patience_counter = 0
    train_losses = []
    dev_losses = []
    dev_macro_f1s = []

    for epoch in range(config.epochs):
        print(f"\n===== Epoch {epoch + 1}/{config.epochs} =====")

        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        train_losses.append(train_loss)

        # 更新学习率
        lr_scheduler.step()
        current_lr = lr_scheduler.get_lr()

        # 验证集评估
        dev_loss, dev_acc, dev_macro_f1, dev_micro_f1 = evaluate_model(model, dev_loader, loss_fn)
        dev_losses.append(dev_loss)
        dev_macro_f1s.append(dev_macro_f1)

        # 打印epoch结果
        print(f"训练损失：{train_loss:.4f} | 当前学习率：{current_lr:.6f}")
        print(f"验证损失：{dev_loss:.4f} | 验证准确率：{dev_acc:.4f}")
        print(f"验证宏F1：{dev_macro_f1:.4f} | 验证微F1：{dev_micro_f1:.4f}")

        # 最优模型保存（带最小提升阈值）
        if dev_macro_f1 > best_macro_f1 + config.min_delta:
            best_macro_f1 = dev_macro_f1
            patience_counter = 0
            # 保存模型、tokenizer、标签映射
            model_save_path = os.path.join(config.save_dir, "best_model")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            with open(os.path.join(model_save_path, "label2id.json"), "w", encoding="utf-8") as f:
                json.dump(label2id, f, ensure_ascii=False, indent=2)
            print(f"✅ 保存最优模型（宏F1：{best_macro_f1:.4f}）")
        else:
            # 早停计数器
            patience_counter += 1
            print(f"⚠️  早停计数器：{patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                print("✅ 早停触发，停止训练")
                break

    print("\n========== 评估最优模型（验证集） ==========")
    # 加载最优模型
    best_model_path = os.path.join(config.save_dir, "best_model")
    best_model = ErnieForSequenceClassification.from_pretrained(best_model_path)
    # 用验证集做详细评估（生成报告和混淆矩阵）
    dev_acc, dev_macro_f1, dev_micro_f1 = evaluate_with_report(best_model, dev_loader, loss_fn, label2id)

    print(f"\n✅ 验证集最终结果：")
    print(f"准确率：{dev_acc:.4f} | 宏F1：{dev_macro_f1:.4f} | 微F1：{dev_micro_f1:.4f}")

    print("\n========== 生成测试集预测结果 ==========")
    # 对无标注测试集预测并保存结果
    predict_test_set(best_model, test_loader, label2id, config.test_pred_save_path, test_dataset)

    # ========== 关键修改2：修复matplotlib后端警告（可选） ==========
    # 避免tkagg交互后端警告，设置为非交互后端
    plt.switch_backend('Agg')  # 若需要显示，可注释这行；若仅保存图片，建议保留

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失", marker='o', markersize=4)
    plt.plot(dev_losses, label="验证损失", marker='s', markersize=4)
    plt.title("训练/验证损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图2：宏F1曲线
    plt.subplot(1, 2, 2)
    plt.plot(dev_macro_f1s, label="验证宏F1", color="orange", marker='^', markersize=4)
    plt.title("验证集宏F1曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1 Score")
    plt.legend()
    plt.grid(alpha=0.3)

    # 保存曲线（避免show()阻塞）
    plt.tight_layout()
    curve_save_path = "./chip_ctc_training_curve.png"
    plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 训练曲线已保存至：{curve_save_path}")
    # plt.show()  # 若需要交互式显示，取消注释（注意：服务器环境可能无法显示）


if __name__ == "__main__":
    main()