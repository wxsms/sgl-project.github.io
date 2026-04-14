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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.64it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.63it/s]


    2026-04-14 12:18:03,391 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 12:18:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.08it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.08it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.08it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.08it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.08it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.08it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.08it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.08it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.08it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.27it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.27it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.27it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.27it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.27it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.27it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.27it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.27it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.75it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.75it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.75it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.75it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.75it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.75it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.75it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.65it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.65it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.65it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.65it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.65it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.65it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.65it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.43it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.43it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.44it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.44it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.44it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.44it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.44it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.44it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.44it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s] Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.48it/s]

    Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.18it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.18it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.18it/s]

    Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  60%|██████    | 35/58 [00:01<00:00, 25.46it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 25.46it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 25.46it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 25.46it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 25.46it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  60%|██████    | 35/58 [00:01<00:00, 25.46it/s]

    Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.11it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.11it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.11it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.11it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.11it/s]

    Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.11it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.54it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.54it/s]

    Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 33.19it/s]


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
    Generated text:  Marie and I’m 20 years old. I have recently started to pursue a career in English and I’m very passionate about it. My dream is to write fiction as a career but I have a lot of questions that I want to ask the community. I would like to start a debate on the topic and share my thoughts with other English learners who might be interested in this topic. I would love to hear from you all. Feel free to share your thoughts and opinions. Let’s make this discussion interesting and engaging! What are some common misconceptions about writing fiction for English learners?
    Certainly! Here are some common misconceptions about writing
    ===============================
    Prompt: The president of the United States is
    Generated text:  a职位是什么?...
    A. President of the United States
    B. President of the Senate
    C. President of the House of Representatives
    D. President of the State of New York
    答案:
    
    A
    
    5月12日，习近平总书记出席____节活动，再次强调要切实把思想和行动统一到党中央决策部署上来，为实施创新驱动发展战略、推动经济高质量发展、实现全面建成小康社会目标，以优异成绩庆祝中国共产主义青年团成立100周年而努力奋斗。
    A. 中国青年节
    B. 国际青年节
    C. 青年节
    D.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the seat of government, of diplomacy, of the arts, of the sciences, of the military, and of the state itself. It is a city of 3 million people, the largest in the world. The city has a long history, and has seen the rise and fall of many kings, queens, and emperors. The most famous is of course, Charles the Great, the first French Emperor. He led the French army to victory over the Mongols in the First Crusade, and built Paris as the seat of the kingdom of France. The city remained a powerful city-state until it was captured by
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it also brings risks and challenges. For example, there is an increasing amount of personal information being collected and processed by big data platforms, which can pose a significant threat to personal privacy. At the same time, AI systems that can predict major events or make decisions are becoming more prominent in society, but they also raise concerns about the safety and ethics of these systems. How to balance the potential benefits of AI with the risks and challenges is crucial.
    The future of AI is bright, but it also brings risks and challenges. For example, there is an increasing amount of personal information being collected and processed by big data platforms, which


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


    Generated text:  Paris, also known as the City of Light, and is the largest city in the European Union. It is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, and many other iconic landmarks. Paris is known for its rich history, art, and culture, and is a popular tourist destination. It is also home to many important institutions, including the French Academy of Sciences and the French National Library. The city is known for its fashion industry, with many famous fashion designers and boutiques. Paris is a vibrant and dynamic city with a rich cultural and historical heritage. It is a popular destination
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in medical diagnosis and treatment, but there is a lot of potential for further development in areas such as personalized medicine, early detection of diseases, and treatment of chronic conditions.
    
    2. AI in finance: AI is already being used in financial services to automate trading, fraud detection, and risk management. There is a lot of potential for further development in areas such as personalized investment strategies, risk management, and fraud detection.
    
    3. AI in
    


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
    Generated text:  [Name] and I'm a [职业] who has a [born date] birthday. I'm [born month] and [born year]. I'm a [occupation] and I was born in [birthday/month/year]. I am [age] years old and [weight] pounds. I have [physical characteristic(s)] and [I am passionate about]. I like to [describe your interest/experience/experience]. I have been [time spent in a specific profession, hobby, sport, etc.]. I like to [describe your hobby/experience/experience]. What would you like to know about [Name]?
    To make
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its rich history, beautiful architecture, and vibrant culture. It is the largest city in France and serves as the capital of the country. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its cultural scene, including music, film, and literature. The city is home to many universities and institutions, including the Sorbonne and the University of Paris. As the cultural and political capital of France, Paris is an important hub for international trade and diplomacy. It is also an important hub for tourism, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a variety of trends, including:
    
    1. Increased accuracy: As AI algorithms become more advanced, they are likely to become even more accurate in their predictions and judgments. This could lead to more reliable and accurate decisions in various fields such as healthcare, finance, and security.
    
    2. Enhanced creativity: AI is increasingly capable of generating novel and innovative ideas, which could lead to more innovative products and services. This could also open up new avenues for creative expression and artistic expression.
    
    3. Automation of repetitive tasks: AI is being increasingly used to automate a variety of tasks, from manufacturing to transportation, which could reduce the need for human


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

    ],

     and

     I

     am

     [

    Age

    ].

     I

     am

     currently

     [

    Position

    ]

     at

     [

    Company

    ],

     and

     I

     enjoy

     [

    Favorite

     Activity

    /

    Interest

    ].

     If

     there

     is

     anything

     I

     can

     assist

     you

     with

     today

    ,

     please

     let

     me

     know

    !

     [

    Optional

    :

     mention

     a

     specific

     task

     or

     project

     you

    're

     working

     on

     right

     now

    ].

     [

    Optional

    :

     mention

     any

     notable

     achievements

     or

     accomplishments

    .]

     [

    Optional

    :

     share

     any

     personal

     experiences

     or

     stories

     that

     have

     impacted

     you

    .

    ]


    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

     am

     [

    Age

    ].

     I

     am

     currently

     [

    Position

    ]

     at

     [

    Company

    ],

     and

     I

     enjoy

     [

    Favorite

     Activity

    /

    Interest

    ].

     If

     there

     is

     anything

     I

     can

     assist

     you

     with

     today

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    Paris

     is

     a

     historical

     and

     cultural

     center

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     nightlife

    .


    Paris

     is

     the

     capital

     of

     France

     and

     was

     founded

     by

     the

     ancient

     Romans

     in

     the

     

    1

    st

     century

     BCE

    .

     Today

    ,

     it

     is

     home

     to

     over

     

    2

     million

     people

     and

     is

     known

     as

     the

     city

     of

     lights

    ,

     a

     major

     shopping

     and

     entertainment

     center

    .


    The

     city

    's

     famous

     landmarks

     include

     the

     E

    iff

    el

     Tower

    ,

     the

     Arc

     de

     Tri

    omp

    he

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     also

     has

     a

     rich

     culinary

     culture

    ,

     with

     traditional

     French

     cuisine

     being

     a

     staple

     of

     the

     city

    's

     dining

     scene

    .


    Paris

     is

     a

     bustling

     hub

     of

     commerce

    ,

     with

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

    ,

     and

     there

     are

     many

     potential

     trends

     that

     could

     shape

     the

     development

     of

     this

     technology

    .

     Here

     are

     some

     potential

     trends

     that

     could

     lead

     to

     advancements

     in

     AI

     and

     new

     applications

     for

     this

     technology

    :
    


    1

    .

     Increased

     integration

     with

     other

     fields

    :

     AI

     is

     already

     being

     integrated

     with

     other

     fields

    ,

     such

     as

     healthcare

    ,

     education

    ,

     and

     transportation

    .

     As

     these

     fields

     continue

     to

     grow

     and

     evolve

    ,

     it

     is

     likely

     that

     AI

     will

     be

     integrated

     more

     deeply

     into

     these

     areas

     as

     well

    .
    


    2

    .

     Enhanced

     decision

    -making

     capabilities

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     is

     likely

     that

     it

     will

     be

     able

     to

     make

     more

     accurate

     and

     informed

     decisions

     in

     a

     wider

     range

     of

     situations

    .

     This

     could

     lead

     to

    



```python
llm.shutdown()
```
