
from src.dLink_chip_ctc import CHIPCTCClassifier

# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 1. 初始化分类器
    classifier = CHIPCTCClassifier()

    # 2. 启动训练（调用对外的train方法）
    classifier.train()

    # 3. 预测示例（训练完成后调用）
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