# NLP-model-work

基于 ERNIE 3.0 的微调分类模型，用于文本分类任务。
- 具体任务：CHIP-CTC（Chinese Medical Text Categorization）
- 输入数据：中文医疗文本（如电子病历、临床记录等）  
- 输出目标：为每条文本分配一个预定义的医学语义类别（如 “Disease”、“Diagnostic”、“Therapy or Surgery”、“Allergy Intolerance” 等）  

以下部分为日志

---
## 1.0
基础的模型，基于 ERNIE 3.0 模型进行微调，并完成文本分类任务。\
**于第二次训练之后出现明显过拟合问题** \
兴许是学习率过低以及训练次数过少?

## 1.1
提高训练轮数和容忍度\
采用更低的学习率衰减参数\
过拟合问题依旧严重
尝试采用别的策略 如增加classweight

