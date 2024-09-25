import os
import torch
from dotenv import load_dotenv
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM
from peft.peft_model import PeftModel
from med_assist.config import CONFIG, set_project_wd

set_project_wd()
load_dotenv()

hf_token = os.environ['HUGGINGFACE_API_KEY']
hf_token_write = os.environ['HUGGINGFACE_API_KEY_WRITE']
adapter_path = "models/med_assist/checkpoint-2000"
base_model_path = CONFIG['llama']['base_path']
hub_model_path = CONFIG['llama']['tuned_path']

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
        )

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    device_map="auto",
    do_sample=True,
    temperature=0.25,
    torch_dtype=torch.bfloat16,
    max_new_tokens=512,
    max_length=4096,
    num_return_sequences=1,
    top_p=1
    )

model_config.pad_token_id = model_config.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    config=model_config,
    quantization_config=quantization_config,
    token=hf_token
)
model_adapted = PeftModel. \
    from_pretrained(model, adapter_path). \
    merge_and_unload()

model_adapted.push_to_hub(hub_model_path, token=hf_token_write)