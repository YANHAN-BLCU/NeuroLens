"""评估模块

提供模型评估功能，包括：
- ASR（攻击成功率）评估
- Utility（效用）评估
- 综合报告生成
"""

from .evaluate import evaluate_sample, evaluate_single, load_salad_data, main as run_evaluation
from .utility_evaluator import evaluate_utility, compute_wikitext_perplexity, evaluate_zero_shot_tasks
from .report import generate_report, generate_asr_report, generate_utility_report, calculate_metrics, load_results

__all__ = [
    # evaluate
    "evaluate_sample",
    "evaluate_single",
    "load_salad_data",
    "run_evaluation",
    # utility_evaluator
    "evaluate_utility",
    "compute_wikitext_perplexity",
    "evaluate_zero_shot_tasks",
    # report
    "generate_report",
    "generate_asr_report",
    "generate_utility_report",
    "calculate_metrics",
    "load_results",
]
