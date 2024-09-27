from transformers import AutoModelForCausalLM,AutoTokenizer , BitsAndBytesConfig
import torch
from datasets import load_dataset



dataset = load_dataset('ruslanmv/ai-medical-chatbot', split='train')
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
test_eval = dataset['test'].train_test_split(test_size=0.001) # just want a few test examples
eval_dataset = test_eval['train']
test_dataset = test_eval['test']

#base_model_id = "Meta-Llama/Llama-3.1"
base_model_id = "BioMistral/BioMistral-7B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side = "left",
    add_eos_token = True,
    add_bos_token = True
)

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""From the AI Medical Chatbot Dataset: Given the medical question and question type, provide an accurate answer.

### Question type:
{test_dataset[1]['Description']}

### Question:
{data_point["Patient"]}

### Answer:
{data_point["Doctor"]}
"""
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
max_length = 512 # Change depending on dataset
tokenizer.pad_token = tokenizer.eos_token

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)