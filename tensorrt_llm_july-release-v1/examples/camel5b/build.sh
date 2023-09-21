# python build.py \
# --model_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/camel_5b_hf/1-gpu/ \
# --max_batch_size=256 --dtype float16 \
# --use_gpt_attention_plugin 2>&1 | tee build.log

# python build.py \
# --model_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/camel_5b_hf/1-gpu/ \
# --max_batch_size=32 --dtype float16 \
# --max_input_len=512 \
# --use_gpt_attention_plugin \
# --use_layernorm_plugin \
# --use_lookup_plugin \
# 2>&1 | tee build_layer_lk.log

# python hf_gpt_convert.py -i ~/workspace/camel_5b/camel-5b-hf/ -o ./camel_5b_hf_kv --storage-type float16 -p 1  --model=gpt2 --calibrate-kv-cache

python build.py \
--model_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/camel_5b_hf_kv/1-gpu/ \
--max_batch_size=32 --dtype float16 \
--max_input_len=512 \
--use_gpt_attention_plugin \
--use_layernorm_plugin \
--use_lookup_plugin \
--int8_kv_cache \
2>&1 | tee build_layer_lk.log

# python hf_gpt_convert.py -i ~/workspace/camel_5b/camel-5b-hf/ -o ./camel_5b_hf_kv --storage-type float16 -p 1  --model=gpt2 --calibrate-kv-cache --smoothquant 0.5

# python build.py \
# --model_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/camel_5b_hf_kv/1-gpu/ \
# --max_batch_size=32 --dtype float16 \
# --max_input_len=512 \
# --use_gpt_attention_plugin \
# --use_layernorm_plugin \
# --use_lookup_plugin \
# --int8_kv_cache \
# --use_smooth_quant \
# 2>&1 | tee build_layer_lk.log
