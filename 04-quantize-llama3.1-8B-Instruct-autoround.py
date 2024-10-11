from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from auto_round import AutoRound

bits, group_size, sym = 4, 128, True
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, batch_size=2, seqlen=512, sym=sym, gradient_accumulate_steps=4, device='cpu')
autoround.quantize()
output_dir = "./AutoRound/GPTQ-sym/"
autoround.save_quantized(output_dir)
