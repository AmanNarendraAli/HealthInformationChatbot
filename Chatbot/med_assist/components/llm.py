import os
import torch
from typing import List, Mapping, Any, Optional, Union
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from med_assist.config import CONFIG

class Llama2(LLM):
    hf_token: Optional[str] = os.environ.get("HUGGINGFACE_API_KEY")
    tokenizer: Optional[AutoTokenizer] = CONFIG.get("llama", dict()).get("base_path")
    model: Optional[AutoModelForCausalLM] = CONFIG.get("llama", dict()).get("base_path")
    adapter: Optional[str] = None

    def __init__(self, **args) -> None:
        super().__init__()

        if isinstance(self.tokenizer, str):
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.tokenizer,
                token=self.hf_token
                )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if isinstance(self.model, str):

            model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model,
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
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model,
                config=model_config,
                quantization_config=quantization_config,
                token=self.hf_token
            )

        if self.adapter != None:
            
            self.model = PeftModel. \
                from_pretrained(self.model, self.adapter). \
                merge_and_unload()

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        max_length = self.model.config.max_length
        
        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids[:, :max_length].to("cuda")
        output_tokens = self.model.generate(input_tokens)
        generated_tokens = output_tokens[:, input_tokens.shape[1]:]
        
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return result

    @property
    def _llm_type(self) -> str:
        return "llama2"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model, "tokenizer": self.tokenizer, "adapter": self.adapter}