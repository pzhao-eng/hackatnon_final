# GPUKiller Hackatnon 复赛报告

### 总述

本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目，我选择的是题目b：用TensorRT-LLM实现新模型。我们选择用TensorRT-LLM实现[camel-5b](https://huggingface.co/Writer/camel-5b-hf)模型的推理。camel-5b可以根据提示生成文本。

#### 编译运行步骤

以下为camel-5b fp16模型的部署的步骤

```bash
# 编译plugin 这一步可以忽略，本来计划优化layernormal plugin，但是由于时间关系还没有对齐答案
## 安装 libpthreads
apt-get update
apt-get install glibc-doc
apt-get install manpages-posix-dev

# clone 远程仓库到本地
cd /root/workspace
git clone https://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@github.com/pzhao-eng/hackatnon_final.git

cd /root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/cpp
mkdir build
cd build
#！ 注意：需要在 /root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/cpp/CMakeLists.txt 中添加TensorRT安装路径
cmake ..
make -j8
## 保存原来的plugin so
mv /usr/local/lib/python3.8/dist-packages/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so \
/usr/local/lib/python3.8/dist-packages/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so.save
## 将编译的so软连接到打开位置
ln -s /root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin.so.9.0.0 \
/usr/local/lib/python3.8/dist-packages/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so
## run trt-llm 就会加载新编译的plugin

# 下载并运行torch版本camel-5b
## install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh |bash
apt-get install git-lfs
git lfs install

## 如果可以链接到 huggingface clone camel-5b 
# mkdir -p /root/workspace/camel_5b
# cd /root/workspace/camel_5b
# git clone https://huggingface.co/Writer/camel-5b-hf
#！ 注意 如果不能clone成功 可以拷贝云主机保存的模型
#! 切换到云主机
docker cp /root/zp/camel_5b/ trt2023_valid:/root/workspace/camel_5b/
#! 切换到docker
cd /root/workspace/camel_5b
python /root/workspace/camel_5b/run_camel_5b.py

# 使用tensor-llm运行camel-5b
## 将hf权重转换为 tf权重
#！ 注意：1. 模型转换时将所有权重加载之后再进行转换会导致显存不足，在单进程下逐个权重进行转换
#！ 2. 由于docker不能访问huggingface 需要将在云主机提前缓存好的dataset 拷贝到docker中
docker cp /root/dataset_require/ trt2023_valid:/root/workspace/
#！切换到docker
tar -xvf /root/workspace/dataset_require/lambada/lambada.tar.xz -C ~/.cache/huggingface
cd /root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b
pip install datasets
python hf_gpt_convert.py -i ~/workspace/camel_5b/camel-5b-hf/ \
-o ./camel_5b_hf_kv --storage-type float16 -p 1  \
--model=gpt2 --calibrate-kv-cache

## 编译为tensorRT-llm engine
#！注意：1. Build 时必须带上use_gpt_attention_plugin 不然自动将attention模块fusion 为 ForeignNode{} (myelin)
#！会由于占用34561064968 bytes (~32GB) workspace 大于A10的显存大小（22GB）导致模型转换失败。
python build.py \
--model_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/camel_5b_hf_kv/1-gpu/ \
--max_batch_size=32 --dtype float16 \
--max_input_len=512 \
--use_gpt_attention_plugin \
--use_layernorm_plugin \
--use_lookup_plugin \
--int8_kv_cache \
2>&1 | tee build_layer_lk.log

## 运行tensorRT engine
#！ 切换到云主机 拷贝huggingface cache
docker cp /root/huggingface/hub/models--gpt2/ trt2023_valid:/root/.cache/huggingface/hub/
#! 切换到docker
cd /root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py \
--engine_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/gpt_outputs/ \
--max_output_len=128

## 运行summary
#！注意：由于显存有限，需要分别运行test_hf 以及 test_trt_llm测试
tar -xvf /root/workspace/dataset_require/cnn_dailiymail/ccdv___cnn_dailymail.tar.xz -C  ~/.cache/huggingface
tar -xvf /root/workspace/dataset_require/rouge/rouge.tar.xz -C  ~/.cache/huggingface
pip install nltk
pip install rouge_score

python summarize.py \
--hf_model_location=/root/workspace/camel_5b/camel-5b-hf/ --data_type=fp16 \
--tokenizer=/root/workspace/camel_5b/camel-5b-hf/ \
--engine_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/gpt_outputs/ \
--test_hf --batch_size=1

python summarize.py \
--hf_model_location=/root/workspace/camel_5b/camel-5b-hf/ --data_type=fp16 \
--tokenizer=/root/workspace/camel_5b/camel-5b-hf/ \
--engine_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/camel5b/gpt_outputs/ \
--test_trt_llm --batch_size=1
```

### 主要开发工作

camel-5b的网络结构和GPT2一致，现有的tensorRT-llm已经给出了网络结构以及plugin的实现，我的主要工作是参照example/gpt 目录中的内容实现camel-5b的推理，由于我们小组是CV背景，通过这次camel-5b的部署让我更加熟悉语言模型这种非结构的输入输出的处理方法。主要开发工作以及需要注意的地方我已经在 ***编译运行步骤*** 进行了介绍，下面是我对layerNormal优化的一些想法，不过这部分工作还没有完成

#### summary结果

- tensorRT-llm results

![img](https://xiaopeng.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2Q3ZWI0ZmJjZTNlMjBlZWY0M2ZiODY0Mzc3MDkzZTNfdkNIY2tpYXdnSjRFMDQ2d2RhSTVtaVFIa1lFRGNuc0tfVG9rZW46VUZjaWJwbTl5b3ZzUUJ4aWtPNWN6VFkyblhSXzE2OTUyNzk5Mjc6MTY5NTI4MzUyN19WNA)

- Hugging Face results

![img](https://xiaopeng.feishu.cn/space/api/box/stream/download/asynccode/?code=NDE5M2UyYzAxNzVhNWIwZGMzZWUzYzQwNzgwOGZjM2ZfaGs0Q1oxcEdQSGZ5OXNleVZUdTU1MGRiWmE2ekJFU3NfVG9rZW46UEEwTWJaZzZzbzZXS3F4QjR6OWM1STQ3bmFlXzE2OTUyNzkxNDM6MTY5NTI4Mjc0M19WNA)

#### 优化（这部分工作还未完成）

1. 原始的layerNormal 计算以及内存访问的效率较低

![img](https://xiaopeng.feishu.cn/space/api/box/stream/download/asynccode/?code=MjE2NWJkMjExMGNmN2MzYzJhNDY1N2JiYjFiNGM0MzdfUWZ0blJCVmc5MmE3RmxDdU4xcEJRUldNRDNDMnl6dERfVG9rZW46UjR1b2JnQ1Vsb1ZNN0N4Vzk4Z2NvWWVmblNiXzE2OTUyNzkxNzM6MTY5NTI4Mjc3M19WNA)

计划通过循环展开增加计算吞吐量，通过矢量访存，降低内存聚并开销，提升内存访问效率。

#### 开发环境

A10 Graphics Card

- TensorRT 9.0.0.1
- Versions of CUDA, CUBLAS, CuDNN used
  - CUDA：cuda 12.0
  - CUBLAS：cublas 12.1
  - CuDNN：cuDNN 8.9.0
- Container used： registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1
- NVIDIA driver version：525.105.17

### **送分题答案**

#### 送分题1：

```Bash
## git clone gpt2-medium model
git lfs install
git clone https://huggingface.co/gpt2-medium
#！ 建议直接copy 
#！切换到云主机将已经clone好的模型拷贝到docker
docker cp /root/zp/gpt2-medium trt2023_valid:/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/gpt/
#! 切换到docker
cd /root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/gpt

## conver hf wieght to tf weight
python hf_gpt_convert.py -i ./gpt2-medium/ --storage-type float16 -p 1 -o ./gpt2_hf  --model=gpt2

## build engine
python build.py --model_dir=./gpt2_hf/1-gpu/ --max_batch_size=1 --dtype float16 --max_input_len=1024

## run engine
python run.py --engine_dir=/root/workspace/hackatnon_final/tensorrt_llm_july-release-v1/examples/gpt/gpt_outputs/ --max_output_len=8
```

![img](https://xiaopeng.feishu.cn/space/api/box/stream/download/asynccode/?code=MDljMzFhYWYzNDIwMzg4ODI4MmQzNWNkOTE2NTkwY2FfR1JQOWc4TWIzcHIxWlJTbmJaem1DMFVBV2tqWk1CMjRfVG9rZW46RVpneWJGYXB3b0FsVzN4OEFINmM0c1NQbnVnXzE2OTUyNzkxODU6MTY5NTI4Mjc4NV9WNA)

#### 送分题2：

```Bash
python summarize.py \
--hf_model_location=./gpt2-medium/  --engine_dir=./gpt_outputs/  \
--test_hf --test_trt_llm  --check_accuracy --tensorrt_llm_rouge1_threshold=14 \
2>&1 | tee q2.log
```

![img](https://xiaopeng.feishu.cn/space/api/box/stream/download/asynccode/?code=MDU4Yjk1ZWM2MTAxMzY4ZWViMWI0MmM3ZDIxOWZlMzVfcFMyRnFSWk55T0h5TXgzZHBrYjNNMVN6OFE4WjVXOWtfVG9rZW46TTE2MWJnaEowb0dwVGd4ZmVUZGNGU0FmbldlXzE2OTUyNzkxODg6MTY5NTI4Mjc4OF9WNA)
