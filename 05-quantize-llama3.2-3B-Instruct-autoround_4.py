from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from auto_round import AutoRound

bits, group_size, sym = 4, 128, False
#autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

## the best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

## fast and low memory, 2-3X speedup, slight accuracy drop at W4G128
autoround = AutoRound(model, tokenizer, nsamples=128, iters=200, seqlen=512, batch_size=4, bits=bits, group_size=group_size, sym=sym)

autoround.quantize()
output_dir = "./AutoRound/meta-llama_Llama-3.2-1B-Instruct-auto_round-int4-gs128-asym"
## format= 'auto_round'(default in version>0.3.0), 'auto_gptq', 'auto_awq'
autoround.save_quantized(output_dir, format='auto_round', inplace=True)

