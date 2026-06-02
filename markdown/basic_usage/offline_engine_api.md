# Offline Engine API

SGLang provides a direct inference engine without the need for an HTTP server, especially for use cases where additional HTTP server adds unnecessary complexity or overhead. Here are two general use cases:

- Offline Batch Inference
- Custom Server on Top of the Engine

This document focuses on the offline batch inference, demonstrating four different inference modes:

- Non-streaming synchronous generation
- Streaming synchronous generation
- Non-streaming asynchronous generation
- Streaming asynchronous generation

Additionally, you can easily build a custom server on top of the SGLang offline engine. A detailed example working in a python script can be found in [custom_server](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/custom_server.py).



## Nest Asyncio
Note that if you want to use **Offline Engine** in ipython or some other nested loop code, you need to add the following code:
```python
import nest_asyncio

nest_asyncio.apply()

```

## Advanced Usage

The engine supports [vlm inference](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py) as well as [extracting hidden states](https://github.com/sgl-project/sglang/blob/main/examples/runtime/hidden_states). 

Please see [the examples](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine) for further use cases.

## Offline Batch Inference

SGLang offline engine supports batch inference with efficient scheduling.


```python
# launch the offline engine
import asyncio

import sglang as sgl
import sglang.test.doc_patch  # noqa: F401
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:05,  7.52it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 12.37it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:05<00:00, 25.05it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 33.72it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 33.72it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 33.72it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 33.72it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 33.72it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 33.72it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 33.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   2%|▏         | 1/58 [00:00<00:06,  8.55it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:06,  8.55it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:05,  9.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:05,  9.67it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:05,  9.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):  10%|█         | 6/58 [00:00<00:04, 12.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:04, 12.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  10%|█         | 6/58 [00:00<00:04, 12.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.46it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:01<00:01, 25.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:01<00:01, 25.59it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:01<00:01, 25.59it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  31%|███       | 18/58 [00:01<00:01, 25.59it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.42it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.42it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.42it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.42it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.42it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.42it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.75it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=192 avail_mem=74.27 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=176 avail_mem=74.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.63it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.02 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=160 avail_mem=74.02 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.73it/s]Capturing num tokens (num_tokens=144 avail_mem=74.25 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.73it/s]Capturing num tokens (num_tokens=128 avail_mem=74.25 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.73it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.25 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.73it/s]Capturing num tokens (num_tokens=96 avail_mem=74.23 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.73it/s] Capturing num tokens (num_tokens=96 avail_mem=74.23 GB):  81%|████████  | 47/58 [00:01<00:00, 25.89it/s]Capturing num tokens (num_tokens=80 avail_mem=74.23 GB):  81%|████████  | 47/58 [00:01<00:00, 25.89it/s]Capturing num tokens (num_tokens=64 avail_mem=74.22 GB):  81%|████████  | 47/58 [00:01<00:00, 25.89it/s]Capturing num tokens (num_tokens=48 avail_mem=74.21 GB):  81%|████████  | 47/58 [00:01<00:00, 25.89it/s]Capturing num tokens (num_tokens=32 avail_mem=74.21 GB):  81%|████████  | 47/58 [00:02<00:00, 25.89it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.21 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.40it/s]Capturing num tokens (num_tokens=28 avail_mem=74.20 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.40it/s]Capturing num tokens (num_tokens=24 avail_mem=74.19 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.40it/s]Capturing num tokens (num_tokens=20 avail_mem=74.19 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.40it/s]Capturing num tokens (num_tokens=16 avail_mem=74.18 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.40it/s]Capturing num tokens (num_tokens=16 avail_mem=74.18 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.64it/s]Capturing num tokens (num_tokens=12 avail_mem=74.17 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.64it/s]Capturing num tokens (num_tokens=8 avail_mem=74.17 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.64it/s] Capturing num tokens (num_tokens=4 avail_mem=74.16 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.64it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.16 GB): 100%|██████████| 58/58 [00:02<00:00, 25.71it/s]


### Non-streaming Synchronous Generation


```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Hello, my name is
    Generated text:  Lex and I am an artificial intelligence assistant. My name means "Light" in the Kriol language, an indigenous language of the Caribbean islands. I was created by Anthropic, a non-profit research lab based in San Francisco. My purpose is to provide user-friendly responses to questions and tasks in a conversational manner. I enjoy discussing the human and social aspects of life, particularly those that revolve around the idea of "light" and the way that the world works. Please ask me anything you'd like to know! Let's chat! I'm here to help! 📱✨
    What are the most common types of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a five-year term. The people who make up the Senate are from each state. How many seats would a person have to hold to be elected to the Senate, if the president of the United States is a five-year term? To determine how many seats a person would have to hold in the Senate to be elected, we need to understand the structure of the U. S. Senate. The Senate is composed of 100 members, with the president of the United States serving as the Vice President and being a five-year term. Each state has one vote, so the seats are divided among the states. Therefore, the total number
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. New York D. Rome
    
    Paris
    
    根据弗兰克·莱特海姆的课程理论，影响课程设计的决定因素包括（）。 A：课程的环境 B：课程的性质 C：课程的内容 D：课程的顺序 E：课程的组织
    
    ABCDE
    
    根据弗兰克·莱特海姆的课程理论，影响课程设计的决定因素包括（）。 A：课程的性质 B：课程的环境 C：课程的内容 D：课程的组织 E：课程的顺序
    
    ABCD
    
    2005版第五套人民币100
    ===============================
    Prompt: The future of AI is
    Generated text:  bright for the job market, with a growing need for software that can automate repetitive tasks, analyze large amounts of data, and perform complex tasks in a variety of fields.
    AI is the science of building computer systems that can perform tasks that typically require human intelligence. These tasks can include speech recognition, image and video processing, natural language understanding, and machine learning. AI can be applied to virtually any industry, from healthcare and finance to transportation and manufacturing. As the technology becomes more advanced, it is becoming increasingly important for businesses to keep up with the latest trends and technologies in the field.
    One of the biggest benefits of AI is that it can


### Streaming Synchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {
    "temperature": 0.2,
    "top_p": 0.9,
}

print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

for prompt in prompts:
    print(f"Prompt: {prompt}")
    merged_output = stream_and_merge(llm, prompt, sampling_params)
    print("Generated text:", merged_output)
    print()
```

    
    === Testing synchronous streaming generation with overlap removal ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major economic center and a major tourist destination. It is a popular destination for tourists and locals alike. The city is known for its fashion, art, and food scene, and it is a hub for many cultural and artistic events. Paris is a city of contrasts, with its modern and traditional elements, and it is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation and robotics: AI is already being used in manufacturing, healthcare, and transportation, but it has the potential to revolutionize these industries by automating tasks and reducing human error. In the future, we may see even more widespread automation, with AI systems becoming more sophisticated and capable of performing tasks that were previously done by humans.
    
    2. Enhanced cognitive abilities: AI is already capable of processing large amounts of data and recognizing patterns, but it has the potential to become even more intelligent and capable in the future. This could lead to breakthroughs in areas such
    


### Non-streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous batch generation ===")


async def main():
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")


asyncio.run(main())
```

    
    === Testing asynchronous batch generation ===


    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text:  [Name], and I am a [Age] year old [City]. I was born and raised in [Your Country], where I grew up in [Your Place]. I attended [Your School], [Your Major], and have a degree in [Your Field]. I have always been passionate about [Your Hobby/Interest] and have been pursuing it with passion and dedication for [Number of Years] years. I am always looking for new ways to learn and grow, and I am a true believer in [Your Values]. I am currently [Your Job Title] at [Your Company], and I have been there for [Your Company]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its name is derived from the Latin word for “Paris” (Parisium), meaning the “City of Love”. Paris is a cosmopolitan city with a rich history and vibrant culture that has shaped the city and its surrounding regions for centuries. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Palace of Versailles. Paris is also a major center for business, finance, and the arts, and is a popular tourist destination. It is home to many notable landmarks and museums, including the Louvre and the Acropolis. Paris has a diverse population
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and unpredictable, but here are some possible trends that could potentially shape the field in the coming years:
    
    1. Increased use of AI for medical diagnosis and treatment: AI is being used in medical diagnosis and treatment to improve the accuracy of diagnoses and develop new treatments. This could lead to more effective medical care and a decrease in the cost of treatments.
    
    2. AI-powered personal assistants: AI-powered personal assistants, such as voice-activated devices and virtual assistants, are becoming more prevalent. These assistants can help users with tasks such as setting reminders, managing schedules, and controlling devices.
    
    3. AI-driven self-driving cars: AI is being used


### Streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous streaming generation (no repeats) ===")


async def main():
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text: ", end="", flush=True)

        # Replace direct calls to async_generate with our custom overlap-aware version
        async for cleaned_chunk in async_stream_and_merge(llm, prompt, sampling_params):
            print(cleaned_chunk, end="", flush=True)

        print()  # New line after each prompt


asyncio.run(main())
```

    
    === Testing asynchronous streaming generation (no repeats) ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text: 

     [

    Name

    ]

     and

     I

     am

     a

     [

    role

     or

     profession

    ]

     who

     has

     been

     living

     in

     [

    location

    ]

     for

     [

    number

    ]

     years

    .

     I

     am

     a

     [

    character

     trait

    ]

     person

    ,

     and

     I

     enjoy

     [

    life

     interests

     or

     passions

    ].

     I

     love

     [

    occupation

     or

     hobby

    ],

     and

     I

     value

     my

     [

    character

     trait

    ].

     I

     have

     a

     sense

     of

     [

    mental

     ability

     or

     personality

     trait

    ].

     I

     am

     [

    average

     or

     exceptional

    ]

     in

     [

    ability

     or

     personality

    ].

     I

     am

     [

    current

     job

     or

     hobby

    ],

     and

     I

     am

     looking

     forward

     to

     [

    future

     goal

     or

     achievement

    ].

     What

     are

     your

     hobbies

     or

     interests

    ?

     What

     are

     your

     strengths

     and

     weaknesses

    ?

     And

     what

     is

     your

     career

     goal

    ?

     Please

     provide

     examples

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     the

     French

     capital

    ,

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Mont

    mart

    re

    .

     The

     city

     is

     also

     famous

     for

     its

     rich

     cultural

     heritage

    ,

     including

     the

     Lou

    vre

     Museum

     and

     the

     Palace

     of

     Vers

    ailles

    .

     Additionally

    ,

     Paris

     is

     a

     vibrant

     and

     multicultural

     city

     that

     attracts

     millions

     of

     visitors

     each

     year

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     The

     city

     is

     home

     to

     many

     renowned

     museums

    ,

     theaters

    ,

     and

     restaurants

    ,

     as

     well

     as

     a

     rich

     history

     that

     dates

     back

     over

     

    1

    ,

    0

    0

    0

     years

    .

     With

     its

     beautiful

     architecture

    ,

     lively

     atmosphere

    ,

     and

     cultural

     richness

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     with

     several

     trends

     expected

     to

     shape

     the

     technology

    's

     development

     and

     applications

    .

     Here

     are

     some

     of

     the

     key

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Deep

     Learning

    :

     As

     AI

     continues

     to

     advance

    ,

     deep

     learning

     will

     become

     increasingly

     important

    .

     This

     involves

     building

     models

     that

     can

     learn

     from

     large

     datasets

    ,

     making

     them

     more

     powerful

     and

     capable

     of

     handling

     complex

     tasks

    .
    


    2

    .

     Automation

    :

     AI

     has

     the

     potential

     to

     automate

     a

     wide

     range

     of

     tasks

    ,

     from

     routine

     to

     critical

     operations

    ,

     freeing

     up

     time

     and

     resources

     for

     human

     workers

     to

     focus

     on

     more

     complex

     tasks

    .
    


    3

    .

     Bi

    ometric

     Security

    :

     Bi

    ometric

     security

     technologies

    ,

     such

     as

     facial

     recognition

     and

     fingerprint

     scanning

    ,

     will

     become

     more

     prevalent

     as

    



```python
llm.shutdown()
```
