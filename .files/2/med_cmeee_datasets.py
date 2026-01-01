from paddle.io import Dataset
import json
import numpy as np
import chardet
import util.tools_function as tf

from tqdm import tqdm
# ======================== 3. 数据加载与预处理 ========================
# 自定义数据集类
class CMeEEV2Dataset(Dataset):
    def __init__(self, data_path, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        # 提前预处理所有数据（避免在__getitem__中做复杂操作）
        self.processed_data = self._load_and_process_data(data_path)

    def _load_and_process_data(self, path):
        """加载并预处理数据，提前完成tokenization和标签映射"""
        processed_data = []

        # 1. 加载原始数据
        raw_data = []
        try:
            # 检测文件编码
            with open(path, 'rb') as f:
                raw_bytes = f.read()
                encoding = chardet.detect(raw_bytes)['encoding'] or 'utf-8-sig'

            # 解析JSON数组
            with open(path, 'r', encoding=encoding) as f:
                content = f.read().strip()
                if not content:
                    print(f"警告：文件 {path} 内容为空！")
                    return processed_data
                json_data = json.loads(content)

                if not isinstance(json_data, list):
                    print(f"警告：文件 {path} 解析结果不是JSON数组")
                    return processed_data

            # 处理原始数据
            for item_idx, item in enumerate(json_data, 1):
                try:
                    # 清理文本
                    text = tf.clean_text(item.get('text', ''))
                    if not text:
                        print(f"警告：第{item_idx}条数据文本为空，跳过")
                        continue

                    # 初始化标签
                    char_labels = ['O'] * len(text)

                    # 处理实体标注
                    for entity in item.get('entities', []):
                        start = entity.get('start_idx', -1)
                        end = entity.get('end_idx', -1)
                        type_ = entity.get('type', '')

                        # 边界和合法性检查
                        if (start < 0 or end <= start or end > len(text) or
                                not type_ or type_ not in ['dis', 'sym', 'pro', 'drug', 'body', 'exam', 'val', 'per']):
                            print(f"警告：第{item_idx}条数据实体无效，跳过 | start={start}, end={end}, type={type_}")
                            continue

                        # BIO标注
                        char_labels[start] = f'B-{type_}'
                        for i in range(start + 1, end):
                            char_labels[i] = f'I-{type_}'

                    raw_data.append((text, char_labels))

                except Exception as e:
                    print(f"警告：第{item_idx}条数据处理失败，跳过 | 错误：{str(e)[:100]}")
                    continue

            print(f"成功加载{len(raw_data)}条原始数据（总数据项数：{len(json_data)}）")

            # 2. 提前完成tokenization（避免在DataLoader中处理）
            print("开始预处理数据（tokenization）...")
            for idx, (text, char_labels) in enumerate(tqdm(raw_data)):
                try:
                    # 关键修复：传入字符串而非列表
                    tokenized = self.tokenizer(
                        text,  # 直接传字符串，不是list(text)
                        max_length=self.config.max_seq_len,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_offsets_mapping=True,
                        return_token_type_ids=False
                    )

                    input_ids = tokenized['input_ids']
                    attention_mask = tokenized['attention_mask']
                    offset_mapping = tokenized['offset_mapping']

                    # 映射字符标签到token标签
                    token_labels = ['O'] * self.config.max_seq_len
                    for token_idx, (start, end) in enumerate(offset_mapping):
                        # 跳过特殊token和padding
                        if start == 0 and end == 0:
                            continue
                        # 找到对应的字符位置
                        char_idx = start
                        if 0 <= char_idx < len(char_labels):
                            token_labels[token_idx] = char_labels[char_idx]

                    # 转换标签为ID
                    label_ids = [
                        self.config.LABEL2ID.get(label, self.config.LABEL2ID['O'])
                        for label in token_labels
                    ]

                    processed_data.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'label_ids': label_ids
                    })

                except Exception as e:
                    print(f"警告：第{idx}条数据tokenization失败，跳过 | 错误：{str(e)[:100]}")
                    continue

            print(f"数据预处理完成，有效数据数：{len(processed_data)}")

        except Exception as e:
            print(f"错误：加载数据失败 | {str(e)}")

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        """仅返回预处理好的数据，避免运行时处理"""
        data = self.processed_data[idx]
        return (
            np.array(data['input_ids'], dtype=np.int64),
            np.array(data['attention_mask'], dtype=np.int64),
            np.array(data['label_ids'], dtype=np.int64)
        )