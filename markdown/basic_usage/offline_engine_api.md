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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]


    2026-04-06 04:18:44,262 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 04:18:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:19,  2.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:19,  2.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:19,  2.44s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:19,  2.44s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.22it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.92it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:02<00:01, 21.44it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:02<00:00, 29.44it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:02<00:00, 29.44it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 29.44it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 29.44it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 29.44it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 29.44it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 29.44it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 29.44it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 36.07it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 36.07it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 36.07it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 36.07it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 36.07it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 36.07it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 36.07it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 36.07it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 42.20it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.09 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.58it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.03 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.01 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=960 avail_mem=55.03 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.58it/s] Capturing num tokens (num_tokens=896 avail_mem=55.03 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=832 avail_mem=55.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.58it/s]

    Capturing num tokens (num_tokens=768 avail_mem=55.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=704 avail_mem=55.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=704 avail_mem=55.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.33it/s]Capturing num tokens (num_tokens=640 avail_mem=55.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.33it/s]Capturing num tokens (num_tokens=576 avail_mem=55.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.33it/s]Capturing num tokens (num_tokens=512 avail_mem=55.00 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.33it/s]Capturing num tokens (num_tokens=480 avail_mem=55.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.33it/s]Capturing num tokens (num_tokens=448 avail_mem=55.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.33it/s]Capturing num tokens (num_tokens=416 avail_mem=55.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.33it/s]Capturing num tokens (num_tokens=416 avail_mem=55.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.61it/s]Capturing num tokens (num_tokens=384 avail_mem=55.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.61it/s]Capturing num tokens (num_tokens=352 avail_mem=55.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.61it/s]Capturing num tokens (num_tokens=320 avail_mem=55.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.61it/s]

    Capturing num tokens (num_tokens=288 avail_mem=55.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.61it/s]Capturing num tokens (num_tokens=256 avail_mem=54.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.61it/s]Capturing num tokens (num_tokens=240 avail_mem=54.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.61it/s]Capturing num tokens (num_tokens=240 avail_mem=54.99 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.95it/s]Capturing num tokens (num_tokens=224 avail_mem=54.99 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.95it/s]Capturing num tokens (num_tokens=208 avail_mem=54.98 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.95it/s]Capturing num tokens (num_tokens=192 avail_mem=54.98 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.95it/s]Capturing num tokens (num_tokens=176 avail_mem=54.98 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.95it/s]Capturing num tokens (num_tokens=160 avail_mem=54.98 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.95it/s]Capturing num tokens (num_tokens=144 avail_mem=54.97 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.95it/s]Capturing num tokens (num_tokens=144 avail_mem=54.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 51.61it/s]Capturing num tokens (num_tokens=128 avail_mem=54.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 51.61it/s]Capturing num tokens (num_tokens=112 avail_mem=54.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 51.61it/s]Capturing num tokens (num_tokens=96 avail_mem=54.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 51.61it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=54.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 51.61it/s]Capturing num tokens (num_tokens=64 avail_mem=54.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 51.61it/s]Capturing num tokens (num_tokens=48 avail_mem=54.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 51.61it/s]Capturing num tokens (num_tokens=48 avail_mem=54.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 52.44it/s]Capturing num tokens (num_tokens=32 avail_mem=54.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 52.44it/s]Capturing num tokens (num_tokens=28 avail_mem=54.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 52.44it/s]Capturing num tokens (num_tokens=24 avail_mem=54.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 52.44it/s]Capturing num tokens (num_tokens=20 avail_mem=54.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 52.44it/s]Capturing num tokens (num_tokens=16 avail_mem=54.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 52.44it/s]Capturing num tokens (num_tokens=12 avail_mem=54.93 GB):  86%|████████▌ | 50/58 [00:01<00:00, 52.44it/s]Capturing num tokens (num_tokens=12 avail_mem=54.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 53.29it/s]Capturing num tokens (num_tokens=8 avail_mem=54.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 53.29it/s] Capturing num tokens (num_tokens=4 avail_mem=54.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 53.29it/s]Capturing num tokens (num_tokens=4 avail_mem=54.93 GB): 100%|██████████| 58/58 [00:01<00:00, 45.61it/s]


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
    Generated text:  Matias and I am a 28 year old female with a 24 hour dietary restriction. I am an active in my community, volunteer for local food banks, and donate to charities. I also recently received a diagnosis of uterine fibroids. I was wondering if you could share some advice or tips on managing my diet and overall well-being while being a vegetarian and with a 24 hour dietary restriction?
    Absolutely, managing a diet while adhering to a 24-hour dietary restriction and being a vegetarian can be challenging, but with a bit of planning and a balanced approach, it’s definitely possible. Here are
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. (判断正误）
    A. 正确
    B. 错误
    答案: A
    
    在使用水力压差计进行测量时，当水被放向水力压差计中的左旋圆筒时，水的流速会怎样？
    A. 保持不变
    B. 减缓
    C. 加快
    D. 立刻停止
    答案: C
    
    在进行油品测量时，如果遇到以下哪种情况，应立即停止测量？
    A. 测量油品出现溢出
    B. 量油尺出现磨损
    C.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the capital of the United Kingdom is London, and the capital of the United States is Washington D. C. Which of these cities is the capital of the United States?
    A: Paris
    B: London
    C: Washington D. C.
    D: Not enough information to determine
    To determine which city is the capital of the United States, we need to consider the capital cities of France, the United Kingdom, and the United States.
    
    1. **Paris, France**: This city is the capital of France.
    2. **London, United Kingdom**: This city is the capital of the United Kingdom.
    3. **Washington D
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it requires a clear vision of the goal and a plan to achieve it. One of the key issues is the need to ensure that data privacy and security is prioritized. This article will explore the importance of data privacy and security in the context of AI and how they relate to the goals of this field. Additionally, it will offer some recommendations for how to ensure data privacy and security in the AI ecosystem.
    The importance of data privacy and security in the context of AI and its goals cannot be overstated. It is crucial to protect the personal information of individuals who use AI technology. This information may include financial data, medical records


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] with [Number of Wheels] wheels. I have [Number of Doors] doors and [Number of Seats] seats. I'm [Type of Vehicle] and I'm [Vehicle Type] with [Number of Wheels] wheels. I have [Number of Doors] doors and [Number of Seats] seats. I'm [Type of Vehicle] and I'm [Vehicle Type] with [Number of Wheels] wheels. I have [Number of Doors] doors and [Number of Seats] seats. I'm [Type
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, art, and cuisine, and is a major cultural and economic center of France. It is also home to the French Parliament and the French Riviera. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city is known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, healthcare, finance, and transportation. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy concerns: As AI becomes more advanced, there will be increased concerns about its ethical implications and potential privacy violations. There will likely be a need for regulations and guidelines to
    


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
    Generated text:  __________ and I am a/an ___________. I live in __________ and I enjoy __________. I have a/an __________ friend, __________. I am a/an __________, and I have a/an __________ job. I am passionate about __________. I have a/an __________ relationship with __________. I have a/an __________ connection with __________. I am looking forward to meeting you soon. How would you like to meet me? Remember, your intro will be neutral and general so that it can build trust and rapport with the reader. Use your imagination and creativity to create your own unique intro! Sure
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital and largest city of France, and is located on the Loire River in the south of the country. It was founded by the French crown in the 12th century as the center of the Old City of Paris, which had been occupied by the Romans during the Roman Empire. Paris is known for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, the Louvre Museum, and the Champs-Élysées. It has a rich history, including several major battles, including the Battle of Tours and the Battle of Agincourt. In 2001, Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field with many potential trends shaping the technology and applications. Some of the most significant trends in AI include:
    
    1. Advanced Machine Learning and Deep Learning: These techniques are becoming increasingly powerful, allowing AI systems to learn from large datasets, improve over time and adapt to new inputs.
    
    2. Integration with Other Technologies: AI is becoming more integrated with other technologies, such as the Internet of Things (IoT) and the Cloud, creating new opportunities and challenges.
    
    3. Ethical and Legal Concerns: As AI systems become more advanced, there is a growing need to address ethical and legal issues, such as bias, privacy,


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

     Sarah

    .

     I

    'm

     a

     [

    insert

     character

    's

     age

    ,

     name

    ,

     or

     profession

    ]

     who

     has

     always

     been

     curious

     about

     the

     world

    .

     I

    'm

     always

     eager

     to

     learn

     and

     to

     make

     new

     friends

    .

     My

     love

     for

     exploring

     new

     places

     and

     trying

     new

     foods

     has

     made

     me

     a

     food

    ie

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     am

     always

     looking

     for

     new

     ways

     to

     share

     my

     knowledge

     and

     experience

     with

     others

    .

     I

    'm

     a

     person

     who

     is

     passionate

     about

     making

     the

     world

     a

     better

     place

    ,

     and

     I

     hope

     to

     continue

     doing

     that

     in

     my

     own

     ways

    .

     I

     believe

     that

     being

     kind

    ,

     honest

    ,

     and

     open

    -minded

     is

     important

     in

     creating

     a

     positive

     impact

     on

     the

     world

    .

     Thank

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     historical

     landmarks

    ,

     vibrant

     culture

    ,

     and

     rich

     cuisine

    .


    Paris

    ,

     the

     historic

     capital

     of

     France

    ,

     is

     renowned

     for

     its

     blend

     of

     historical

     architecture

    ,

     cultural

     richness

    ,

     and

     culinary

     delights

    .

     The

     city

    's

     lively

     atmosphere

     and

     iconic

     landmarks

     like

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

     and

     Notre

    -D

    ame

     Cathedral

     draw

     millions

     of

     tourists

     each

     year

    ,

     while

     its

     culinary

     scene

    ,

     known

     for

     its

     cuisine

     that

     combines

     French

     flavors

     with

     international

     influences

    ,

     is

     a

     delightful

     gastr

    onomic

     experience

    .

     The

     city

    's

     rich

     history

     and

     ongoing

     cultural

     activities

     make

     it

     a

     major

     destination

     for

     those

     interested

     in

     exploring

     France

    's

     cultural

     landscape

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     a

     wide

     range

     of

     technological

     developments

     and

     applications

    ,

     and

     there

     are

     many

     potential

     trends

     that

     could

     emerge

     in

     the

     coming

     years

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     AI

     models

     will

     continue

     to

     get

     even

     more

     sophisticated

     and

     powerful

    ,

     with

     the

     ability

     to

     learn

     from

     large

     amounts

     of

     data

     and

     make

     better

     and

     more

     accurate

     predictions

     and

     decisions

    .
    


    2

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     integrate

     more

     with

     other

     technologies

    ,

     including

     IoT

    ,

     the

     Internet

     of

     Things

    ,

     and

     other

     types

     of

     advanced

     computing

     and

     storage

     systems

    .
    


    3

    .

     Enhanced

     privacy

     and

     data

     security

    :

     AI

     systems

     will

     become

     more

     sophisticated

    



```python
llm.shutdown()
```
