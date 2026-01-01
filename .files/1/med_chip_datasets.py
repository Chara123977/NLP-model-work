from paddle.io import Dataset, DataLoader

#===================== 用于读取NLP数据集 =====================
class CHIPCTCDataset(Dataset):
    """适配CHIP-CTC单文本多分类格式：id/label/text（兼容无标注test数据）"""
    def __init__(self, data_list, tokenizer, max_seq_len, label2id=None, is_test=False):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_test = is_test  # 标记是否为无标注测试集
        # 新增：单独存储测试集的sample_id列表（避免字符串进入张量）
        self.test_sample_ids = [item.get("id", "") for item in data_list] if is_test else None

        if label2id is None:
            self.label2id = self._build_label_map()
        else:
            self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)

    def _build_label_map(self):
        if self.is_test:
            raise ValueError("无标注测试集不能自动构建标签映射！请传入训练集的label2id")
        labels = set()
        for item in self.data:
            labels.add(item.get("label", "Unknown"))
        sorted_labels = sorted(list(labels))
        return {label: idx for idx, label in enumerate(sorted_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample.get("text", "").strip()

        # 无标注测试集：仅返回数值类型的张量（剥离字符串ID）
        if self.is_test:
            encoded_input = self.tokenizer(
                text=text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            # 只返回数值类型的张量，字符串ID通过self.test_sample_ids后续匹配
            return (
                encoded_input["input_ids"],
                encoded_input["token_type_ids"],
                encoded_input["attention_mask"]
            )
        else:
            label = sample.get("label", "Unknown")
            encoded_input = self.tokenizer(
                text=text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            return (
                encoded_input["input_ids"],
                encoded_input["token_type_ids"],
                encoded_input["attention_mask"],
                self.label2id.get(label, 0)
            )