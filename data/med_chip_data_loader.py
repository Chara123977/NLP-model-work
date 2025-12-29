import os
import json
from paddlenlp.transformers import ErnieTokenizer
from paddlenlp.data import Pad, Stack, Tuple
from data.med_chip_datasets import CHIPCTCDataset
from paddle.io import Dataset, DataLoader


class CHIPCTCDataLoader:
    """CHIP-CTC 数据集加载类，封装所有数据加载相关功能"""

    def __init__(self, config):
        """
        初始化数据加载器
        Args:
            config: 配置对象，需包含 model_name, data_dir, batch_size, max_seq_len 等属性
        """
        self.config = config
        self.tokenizer = None
        self.label2id = None
        self.test_dataset = None

        # 初始化所有DataLoader
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None

    def _load_json_file(self, file_path, is_test=False):
        """
        兼容两种JSON格式：
        1. 每行一个JSON对象（推荐）
        2. 整个文件是一个JSON数组
        is_test=True 时加载无标注数据（仅id/text）
        """
        data = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        print(f"正在加载文件：{file_path}")
        try:
            # 尝试1：按JSON数组加载
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 校验是否是列表/数组
            if isinstance(data, list) and len(data) > 0:
                valid_data = []
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # 无标注数据仅校验id和text
                        if is_test:
                            if all(k in item for k in ["id", "text"]):
                                valid_data.append(item)
                            else:
                                print(f"警告：数组第{idx + 1}项缺少id/text字段，跳过")
                        else:
                            if all(k in item for k in ["id", "label", "text"]):
                                valid_data.append(item)
                            else:
                                print(f"警告：数组第{idx + 1}项不是有效字典，跳过")
                data = valid_data
                print(f"✅ 以JSON数组格式加载：{len(data)} 条有效数据")
                return data
        except json.JSONDecodeError:
            # 尝试2：按每行一个JSON对象加载
            print("JSON数组格式加载失败，尝试按行加载...")
            valid_data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if is_test:
                            if all(k in item for k in ["id", "text"]):
                                valid_data.append(item)
                            else:
                                print(f"警告：第{line_num + 1}行缺少id/text，跳过：{line[:50]}...")
                        else:
                            if all(k in item for k in ["id", "label", "text"]):
                                valid_data.append(item)
                            else:
                                print(f"警告：第{line_num + 1}行缺少字段，跳过：{line[:50]}...")
                    except json.JSONDecodeError:
                        print(f"警告：第{line_num + 1}行JSON格式错误，跳过：{line[:50]}...")
            data = valid_data

        if len(data) == 0:
            raise ValueError(f"文件 {file_path} 中无有效数据！")
        print(f"✅ 以逐行格式加载：{len(data)} 条有效数据")
        return data

    def _load_chip_ctc_data(self):
        """加载已划分的train/dev/test.json（兼容无标注test）"""
        train_path = os.path.join(self.config.data_dir, "train.json")
        dev_path = os.path.join(self.config.data_dir, "dev.json")
        test_path = os.path.join(self.config.data_dir, "test.json")

        missing_files = []
        for path, name in [(train_path, "train.json"), (dev_path, "dev.json"), (test_path, "test.json")]:
            if not os.path.exists(path):
                missing_files.append(name)

        if missing_files:
            raise FileNotFoundError(
                f"缺少已划分的数据集文件：{missing_files}\n"
                f"请确认 {self.config.data_dir} 下有 train.json/dev.json/test.json"
            )

        # 加载有标注数据
        train_data = self._load_json_file(train_path)
        dev_data = self._load_json_file(dev_path)
        # 加载无标注测试集
        test_data = self._load_json_file(test_path, is_test=True)

        return train_data, dev_data, test_data

    def build_dataloaders(self):
        """构建所有数据加载器（主入口方法）"""
        # 1. 加载Tokenizer
        self.tokenizer = ErnieTokenizer.from_pretrained(self.config.model_name)
        print(f"\n✅ 成功加载模型Tokenizer：{self.config.model_name}")

        # 2. 加载原始数据集
        print("\n开始加载已划分的数据集...")
        train_data, dev_data, test_data = self._load_chip_ctc_data()

        # 3. 构建Dataset实例
        train_dataset = CHIPCTCDataset(train_data, self.tokenizer, self.config.max_seq_len)
        self.label2id = train_dataset.label2id
        dev_dataset = CHIPCTCDataset(dev_data, self.tokenizer, self.config.max_seq_len, label2id=self.label2id)
        self.test_dataset = CHIPCTCDataset(
            test_data, self.tokenizer, self.config.max_seq_len,
            label2id=self.label2id, is_test=True
        )

        print(f"\n✅ 标签映射（自动生成）：{self.label2id}")
        print(f"✅ 分类数量：{len(self.label2id)}")

        # 4. 定义批量处理函数
        # 有标注数据的批量处理函数
        batchify_fn = Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
            Pad(axis=0, pad_val=0),  # attention_mask
            Stack(dtype="int64")  # label
        )

        # 无标注测试集的批量处理函数
        test_batchify_fn = Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
            Pad(axis=0, pad_val=0),  # attention_mask
        )

        # 5. 构建DataLoader
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=batchify_fn,
            drop_last=True,
            num_workers=0
        )

        self.dev_loader = DataLoader(
            dataset=dev_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=batchify_fn,
            num_workers=0
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=test_batchify_fn,
            num_workers=0
        )

        print("\n✅ 所有数据加载器构建完成！")
        return self.train_loader, self.dev_loader, self.test_loader

    def get_additional_info(self):
        """获取额外信息（tokenizer/label2id/test_dataset）"""
        if not self.tokenizer or not self.label2id:
            raise RuntimeError("请先调用 build_dataloaders() 方法构建数据加载器")
        return self.tokenizer, self.label2id, self.test_dataset