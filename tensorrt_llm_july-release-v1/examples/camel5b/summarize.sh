python summarize.py \
--hf_model_location=/root/workspace/camel_5b/camel-5b-hf/ --data_type=fp16 \
--tokenizer=/root/workspace/camel_5b/camel-5b-hf/ \
--engine_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/gpt_outputs/ \
--test_hf --batch_size=1  
