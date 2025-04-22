from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"  
save_dir = "/nlp/scr/kathli/repos/TinyZero/model"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_dir)
