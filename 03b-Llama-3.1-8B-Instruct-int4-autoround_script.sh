git clone https://github.com/intel/auto-round
cd auto-round/examples/language-modeling
pip install -r requirements.txt
python3 main.py \
--model_name  meta-llama/Meta-Llama-3.1-8B-Instruct \
--device 0 \
--group_size 128 \
--bits 4 \
--nsamples 512 \
--iters 1000 \
--model_dtype "fp16" \
--deployment_device 'auto_round' \
--eval_bs 16 \
--output_dir "./tmp_autoround" \

