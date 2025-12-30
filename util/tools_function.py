import re

# ======================== 2. 工具函数 ========================
def clean_text(text):
    """清理文本，确保是字符串格式并去除特殊空白字符"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # 去除不可见字符
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    # 统一空白字符为单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text