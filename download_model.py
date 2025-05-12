from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"  
save_dir = "/nlp/scr/kathli/repos/TinyZero/model"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_dir)
