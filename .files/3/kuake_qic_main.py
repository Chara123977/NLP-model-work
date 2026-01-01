from src.kuake_qic import KUAKEQICTrainer
from src.kuake_qic import KUAKEQICPredictor
from paddlenlp.utils.log import logger
import os
import paddle
from config.kuake_qic_config import Config

# 主函数
def main():
	# 初始化配置对象
	config = Config()

	# 创建输出目录
	os.makedirs(config.output_dir, exist_ok=True)

	# 初始化训练器并开始训练
	trainer = KUAKEQICTrainer(config)
	trainer.train()

	# ========== 使用独立的预测器进行预测 ==========
	logger.info("\n===== 单条文本预测示例 =====")
	# 初始化预测器
	predictor = KUAKEQICPredictor(config)

	# 单条预测示例
	test_queries = [
		"如何预防流感",
		"散光能好吗",
		"白癜风不断恶化的原因",
		"脂肪瘤是否存在恶变的倾向？"
	]

	# 单条预测
	for query in test_queries:
		result = predictor.predict_single(query)
		logger.info(f"输入：{query}")
		logger.info(f"预测意图：{result['pred_label']}（置信度：{result['confidence'][result['pred_label']]:.4f}）")

	# 批量预测示例
	logger.info("\n===== 批量文本预测示例 =====")
	batch_results = predictor.predict_batch(test_queries)
	for result in batch_results:
		logger.info(
			f"输入：{result['text']} | 预测意图：{result['pred_label']} | 置信度：{result['confidence'][result['pred_label']]:.4f}")


if __name__ == "__main__":
	# 设置设备（自动识别GPU/CPU）
	paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
	main()

