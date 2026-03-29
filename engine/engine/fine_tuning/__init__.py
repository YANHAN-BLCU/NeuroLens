"""
Fine-tuning modules for targeted safety fine-tuning (TSFT)
"""

from .refusal_templates import (
    extract_refusal_templates,
    analyze_refusal_patterns,
    save_refusal_templates,
    load_refusal_templates,
)
from .salad_taxonomy import get_prompt_category, load_salad_taxonomy
from .dataset_builder import (
    build_refusal_guided_dataset,
    combine_template_with_prompt,
    save_dataset,
)
from .tsft import (
    tsft_finetune,
    enable_safety_neuron_gradients,
    create_tsft_optimizer,
    load_dedicated_safety_neurons,
)

__all__ = [
    "extract_refusal_templates",
    "analyze_refusal_patterns",
    "save_refusal_templates",
    "load_refusal_templates",
    "get_prompt_category",
    "load_salad_taxonomy",
    "build_refusal_guided_dataset",
    "combine_template_with_prompt",
    "save_dataset",
    "tsft_finetune",
    "enable_safety_neuron_gradients",
    "create_tsft_optimizer",
    "load_dedicated_safety_neurons",
]
