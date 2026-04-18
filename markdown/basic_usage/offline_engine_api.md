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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-18 01:10:06] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.98it/s]


    2026-04-18 01:10:11,680 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-18 01:10:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.26it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:23,  2.26it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.58it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.43it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 20.45it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 29.24it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 38.29it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 47.51it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 47.51it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 47.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.43 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   9%|▊         | 5/58 [00:00<00:02, 19.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 19.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 19.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 19.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.09it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.71 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.71 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.71 GB):  21%|██        | 12/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.70 GB):  21%|██        | 12/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.70 GB):  21%|██        | 12/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.70 GB):  21%|██        | 12/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.70 GB):  21%|██        | 12/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.69 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.69 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=120.69 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.53it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.24it/s]

    Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  50%|█████     | 29/58 [00:00<00:00, 36.09it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  50%|█████     | 29/58 [00:00<00:00, 36.09it/s]Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  50%|█████     | 29/58 [00:00<00:00, 36.09it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:00<00:00, 36.09it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 36.09it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.17it/s]Capturing num tokens (num_tokens=352 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.17it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.17it/s]

    Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.17it/s]Capturing num tokens (num_tokens=256 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.17it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.17it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=208 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=192 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.02it/s]

    Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=96 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.02it/s] Capturing num tokens (num_tokens=96 avail_mem=120.19 GB):  81%|████████  | 47/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=48 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  81%|████████  | 47/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.00it/s]Capturing num tokens (num_tokens=28 avail_mem=120.17 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.00it/s]

    Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.00it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.00it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.00it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.00it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.09it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 34.84it/s]


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
    Generated text:  Brian. I am from Los Angeles, California. I am a college student. I am a firm believer in the 'curiosity' and 'curiosity' game. I have decided to teach the 'curiosity' game to other people. I have prepared a little game of my own. Here is the game: Pick a number. I will ask you a question and you will have to guess the number. If you guess correctly, I will give you a happy smile and gold star. I will always be happy to help you learn to be a better problem solver. The game is free and anyone can play. Please come along with
    ===============================
    Prompt: The president of the United States is
    Generated text:  a(n) ______. ( 选择）
    A. Prime Minister
    B. Cabinet
    C. House of Representatives
    D. President of the United States
    答案:
    
    D
    
    根据《金融机构反洗钱规定》，金融机构应当建立客户身份识别制度，客户身份识别制度包括（）。 A.识别客户的国籍、地址 B.识别客户的经济状况 C.识别客户的交易目的和性质 D.识别客户的交易习惯 E.识别客户的交易金额和频率
    答案:
    
    A, B, C, D, E
    
    【单选题】在计算总体均值的抽样标准误差时，变量值的变异
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of France is (　　)
    A: Shanghai
    B: Moscow
    C: Sydney
    D: London To determine the capital of France, let's analyze each option step by step:
    
    1. **Paris**: This is the capital of France. It is known for its rich history and cultural heritage, including the Eiffel Tower.
    2. **Moscow**: This is the capital of Russia. Russia is a different country, and it does not have a capital city.
    3. **Sydney**: This is the capital of Australia. Australia is a country, not a city.
    4. **London**: This
    ===============================
    Prompt: The future of AI is
    Generated text:  being rapidly shaped by artificial intelligence systems that learn to make decisions on their own, predict outcomes based on data, and make decisions based on the evolving data, which, in turn, is causing many of the most important questions to be asked: Are AI systems truly intelligent? Are they able to understand the world around them? How can they be trusted to make decisions when the data available to them is incomplete and uncertain? These are the questions that can be answered by a researcher at the University of the Sunshine Coast in Queensland, Australia.
    The University of the Sunshine Coast is one of the top universities in Australia, and its researchers are making a significant


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm [age] years old, and I have [number] years of experience in [job title]. I'm a [gender] and [race] person. I'm [height] inches tall and [weight] pounds. I have [eye color] eyes and [hair color] hair. I'm [gender] and [race] and I'm [gender] and [race]. I'm [gender] and [race]. I'm [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its cuisine, fashion, and art scene, making it a popular tourist destination. The city is also home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. Its status as the capital of France is a testament to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate tasks that are currently done by humans, such as data analysis, decision-making, and routine maintenance. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. Enhanced human-computer interaction: AI will continue to improve its ability to understand and respond to human language, emotions, and intentions. This will lead to more natural and intuitive interactions between humans and machines, and will also create new opportunities for collaboration and communication.
    
    3. AI will become more integrated with other technologies: AI will continue to
    


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
    Generated text:  [Name] and I'm a [Age] year old [Gender] person who has been studying [Field of Study] for [Number of Years] years. I have a love for [Favorite Activity or Hobby], and I enjoy spending my days [Time/Location]. I also enjoy [Other Hobby or Activity], and I find [Name] to be a [Favorite Hobby/Activity] that I enjoy [Number] times per week. I am always looking for [Opportunity or Challenge], and I enjoy [More Opportunity/Challenge] that I get to do. What interests you the most? 
    In your experience, what has
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the largest and most populous city in the European Union, and it is the cultural, economic, and political capital of France. It is located in the western part of the country, on the island of France, and has a population of over 2 million people. It is home to many of France’s most famous landmarks, including Notre-Dame Cathedral, the Arc de Triomphe, the Eiffel Tower, and the Louvre Museum. The city is known for its rich history, beautiful architecture, and delicious cuisine, and it is a popular destination for tourists and locals alike. Paris is also a major center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly diverse and complex, with new developments and trends emerging regularly. Some potential future trends include:
    
    1. Increased automation: As AI technology continues to evolve, we can expect to see more and more automation in various industries. This could lead to the creation of more efficient and productive machines, as well as the reduction of human labor.
    
    2. Enhanced understanding of human emotions: AI is currently limited in its ability to fully understand and interpret human emotions. However, as the technology improves, we may see an increase in AI that is better able to understand and respond to human emotions.
    
    3. Increased collaboration and communication: As AI becomes


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

    career

    ]

     [

    Field

    ]

     in

     the

     [

    industry

    ]

     field

    .

     I

     have

     been

     working

     in

     this

     field

     for

     [

    number

     of

     years

    ]

     years

    ,

     and

     I

     have

     gained

     a

     lot

     of

     experience

     in

     this

     field

    .

     I

     am

     a

     [

    respons

    ibility

    ]

     person

     and

     I

     always

     take

     responsibility

     for

     my

     work

    .

     I

     have

     a

     [

    strength

    /

    weak

    ness

    ]

     in

     [

    mention

     a

     specific

     skill

     or

     experience

    ].

     I

     love

     [

    job

     title

    ]

     because

     I

     get

     to

     [

    mention

     something

     specific

     about

     the

     job

    ].

     I

     am

     [

    job

     title

    ]

     and

     I

    'm

     excited

     to

     [

    mention

     something

     specific

     about

     the

     job

    ]

     with

     you

    .

     [

    Name

    ]

     "

    N

    /A

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     hosts

     the

     E

    iff

    el Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Notre

     Dame

     Music

     Hall

    .

     Its

     name

     translates

     to

     "

    The

     City

     of

     Light

    "

     in

     French

    ,

     and

     it

     is

     also

     known

     as

     the

     City

     of

     Love

    ,

     and

     the

     City

     of

     Literature

    .
    


    France

    's

     capital

     city

    ,

     Paris

    ,

     is

     renowned

     for

     its

     grand

     architecture

    ,

     cultural

     richness

    ,

     and

     love

     for

     art

     and

     literature

    .

     The

     city

    's

     iconic

     landmarks

     include

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Notre

     Dame

     Music

     Hall

    ,

     while

     its

     museums

     such

     as

     the

     Lou

    vre

     and

     Mus

    ée

     d

    '

    Or

    say

     are

     also

     highly

     regarded

    .

     Paris

     is

     also

     home

     to

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     grow

     and

     evolve

    ,

     with

     many

     new

     trends

     and

     developments

     in

     the

     field

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     integration

     with

     humans

    :

     AI

     systems

     are

     likely

     to

     become

     more

     integrated

     with

     humans

    ,

     creating

     a

     more

     human

    -like

     interaction

     with

     AI

    .

     This

     could

     include

     more

     natural

     language

     processing

    ,

     better

     understanding

     of

     human

     emotions

     and

     preferences

    ,

     and

     more

     human

    -like

     decision

    -making

    .
    


    2

    .

     Adv

    ancements

     in

     data

     analytics

    :

     AI

     systems

     are

     likely

     to

     become

     more

     sophisticated

     in

     their

     ability

     to

     analyze

     large

     amounts

     of

     data

     and

     make

     accurate

     predictions

    .

     This

     could

     lead

     to

     more

     accurate

     medical

     diagnoses

    ,

     more

     efficient

     financial

     analysis

    ,

     and

     more

     personalized

     marketing

    .
    


    3

    .

    



```python
llm.shutdown()
```
