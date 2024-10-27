from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig

backend = "cpu"  ##cpu, hpu, cuda, cuda:marlin(supported in auto_round>0.3.1 'pip install -v gptqmodel --no-build-isolation')
quantization_config = AutoRoundConfig(
    backend=backend
)
quantized_model_path = "./AutoRound/meta-llama_Llama-3.2-3B-Instruct-Autoround-GPTQ-asym"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map=backend.split(':')[0], quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

