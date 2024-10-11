from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig

device = "cpu"  ##cpu, hpu, cuda
quantization_config = AutoRoundConfig(
    backend=device
)
quantized_model_path = "./AutoRound/Llama-3.2-3B-Instruct-AutoRound-GPTQ-sym"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map=device, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))