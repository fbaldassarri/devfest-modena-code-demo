from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round.auto_quantizer import AutoHfQuantizer

# quantized_model_dir = "Intel/Meta-Llama-3.1-8B-Instruct-AutoRound"
quantized_model_dir = "./AutoRound/Llama-3.1-8B-Instruct-AutoRound-GPTQ-INT4-asym"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
model = AutoModelForCausalLM.from_pretrained(quantized_model_dir,
                                             device_map="auto"
                                             )

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

text = "There is a girl who likes adventure,"

