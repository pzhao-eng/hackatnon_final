# python build.py \
# --model_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/camel_5b_hf/1-gpu/ \
# --max_batch_size=256 --dtype float16 \
# --use_gpt_attention_plugin 2>&1 | tee build.log

# python build.py \
# --model_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/camel_5b_hf/1-gpu/ \
# --max_batch_size=256 --dtype float16 \
# --use_gpt_attention_plugin \
# --use_layernorm_plugin \
# --use_lookup_plugin \
# 2>&1 | tee build_layer_lk.log

python build.py \
--model_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/camel_5b_hf/1-gpu/ \
--max_batch_size=256 --dtype float16 \
--use_gpt_attention_plugin \
--enable_context_fmha \
--use_layernorm_plugin \
2>&1 | tee build_layer_ctx.log
