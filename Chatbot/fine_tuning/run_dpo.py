import os
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import DPOTrainer
from med_assist.config import CONFIG, set_project_wd

set_project_wd()

tuned_model_path = "models/tuned/llama-2-med-assist-v0.3" 

tuning_datasets = load_dataset(
    path="resources", 
    data_files={"train": "training_dataset_dpo.csv"}, 
    )

hf_token = os.environ['HUGGINGFACE_API_KEY']

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_config = AutoConfig.from_pretrained(
    CONFIG['llama']['base_path'],
    device_map="auto",
    do_sample=True,
    temperature=0.25,
    torch_dtype=torch.bfloat16,
    max_new_tokens=512,
    max_length=4096,
    num_return_sequences=1,
    top_p=1,
    use_cache=False,
    token=hf_token
    )
model_config.pad_token_id = model_config.eos_token_id

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=CONFIG["llama"]["base_path"],
    token=hf_token
)
tokenizer.pad_token_id = model_config.eos_token_id
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=CONFIG["llama"]["base_path"],
    config=model_config,
    quantization_config=quantization_config,
    token=hf_token
)

model.enable_input_require_grads()

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_alpha=16, 
    lora_dropout=0.1
    )

peft_model = PeftModel.from_pretrained(
    model,
    CONFIG['llama']['adapter_path'],
    is_trainable=True,
    adapter_name="sft",
)

peft_model.load_adapter(
    CONFIG['llama']['adapter_path'], 
    adapter_name="ref"
    )

training_args = TrainingArguments(
    output_dir = "models/med_assist/",
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = 5,
    weight_decay = 0.001,
    learning_rate = 1e-3,
    max_grad_norm=0.3,
    optim = "paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    group_by_length=True,
    gradient_checkpointing=True,
    evaluation_strategy = "steps",
    save_strategy = "epoch",
    eval_steps=100,
    logging_steps=100, 
    report_to="tensorboard"
)

dpo_trainer = DPOTrainer(
    peft_model,
    model_adapter_name="sft",
    ref_adapter_name="ref",
    args=training_args,
    beta=0.1,
    train_dataset=tuning_datasets['train'],
    tokenizer=tokenizer,
)