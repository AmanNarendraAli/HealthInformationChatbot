from tokenize_dataset import *
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model = Accelerator.prepare_model(model)

