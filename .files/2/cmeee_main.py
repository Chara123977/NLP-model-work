
from src.cmeee_v2 import CMeEEEntityRecognizer

# ======================== 使用示例 ========================
if __name__ == "__main__":
	# 1. 初始化实体识别器
	recognizer = CMeEEEntityRecognizer()

	# 2. 训练模型（自动生成可视化图表）
	best_dev_f1 = recognizer.train()
	print(f"\n训练完成，最优验证集宏F1：{best_dev_f1:.4f}")

	# 3. 测试模型
	#test_f1, test_report, test_preds = recognizer.test(
	#	save_path="output/cmeee_test_predictions.json"
	#)

	# 4. 单条文本推理示例
	#test_text = "六、新生儿疾病筛查的发展趋势自1961年开展苯丙酮尿症筛查以来，随着医学技术的发展，符合进行新生儿疾病筛查标准的疾病也在不断增加。"
	#pred_results = recognizer.predict(test_text)
	#print("\n单条文本推理示例结果：")
	#print(f"输入文本：{test_text}")
	#print("识别的实体：")
	#for entity in pred_results:
	#	print(f"- 实体：{entity['text']}，类型：{entity['type']}，位置：{entity['start']}-{entity['end']}")