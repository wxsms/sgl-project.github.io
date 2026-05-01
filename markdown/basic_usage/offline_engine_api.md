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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.15it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.14it/s]


    2026-05-01 08:31:44,924 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 08:31:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.44it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.44it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.44it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.44it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.44it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.44it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.44it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.44it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:05,  6.75it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:05,  6.75it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:05,  6.75it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:05,  6.75it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:05,  6.75it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:05,  6.75it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:05<00:05,  6.75it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:03, 10.24it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:03, 10.24it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:03, 10.24it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:03, 10.24it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:03, 10.24it/s]

    Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:03, 10.24it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:05,  5.21it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:05,  5.21it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:05,  5.21it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:05,  5.21it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:05,  5.21it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:05,  5.21it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:07<00:03,  7.20it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:07<00:03,  7.20it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:07<00:03,  7.20it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:07<00:03,  7.20it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:07<00:03,  7.20it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:07<00:03,  7.20it/s]

    Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:07<00:03,  7.20it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:07<00:00, 16.44it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:07<00:00, 22.79it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:07<00:00, 22.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.27 GB):   3%|▎         | 2/58 [00:00<00:03, 16.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:03, 16.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:03, 16.78it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:03, 16.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.26 GB):   9%|▊         | 5/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.25 GB):   9%|▊         | 5/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.25 GB):   9%|▊         | 5/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.24 GB):   9%|▊         | 5/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.24 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.22 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.28it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.22 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.21 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.21 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.21 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.21 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.38it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.21 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.19 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.71 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.57 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.54 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.90it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=70.51 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.51 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=960 avail_mem=70.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.54it/s] Capturing num tokens (num_tokens=896 avail_mem=70.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=832 avail_mem=70.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=768 avail_mem=70.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=704 avail_mem=70.52 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.54it/s]Capturing num tokens (num_tokens=704 avail_mem=70.52 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.46it/s]Capturing num tokens (num_tokens=640 avail_mem=70.51 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.46it/s]Capturing num tokens (num_tokens=576 avail_mem=70.51 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.46it/s]Capturing num tokens (num_tokens=512 avail_mem=70.50 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.46it/s]

    Capturing num tokens (num_tokens=480 avail_mem=70.51 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.46it/s]Capturing num tokens (num_tokens=480 avail_mem=70.51 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=448 avail_mem=70.51 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=416 avail_mem=70.51 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=384 avail_mem=70.51 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=352 avail_mem=70.50 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=352 avail_mem=70.50 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.53it/s]Capturing num tokens (num_tokens=320 avail_mem=70.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.53it/s]Capturing num tokens (num_tokens=288 avail_mem=70.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.53it/s]

    Capturing num tokens (num_tokens=256 avail_mem=70.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.53it/s]Capturing num tokens (num_tokens=240 avail_mem=70.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.53it/s]Capturing num tokens (num_tokens=240 avail_mem=70.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=224 avail_mem=70.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=208 avail_mem=70.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=192 avail_mem=70.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=176 avail_mem=70.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=176 avail_mem=70.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=160 avail_mem=70.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.29it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=128 avail_mem=70.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=112 avail_mem=70.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.29it/s]

    Capturing num tokens (num_tokens=112 avail_mem=70.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.00it/s]Capturing num tokens (num_tokens=96 avail_mem=70.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.00it/s] Capturing num tokens (num_tokens=80 avail_mem=70.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.00it/s]Capturing num tokens (num_tokens=64 avail_mem=70.45 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.00it/s]Capturing num tokens (num_tokens=48 avail_mem=70.45 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.00it/s]Capturing num tokens (num_tokens=48 avail_mem=70.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=32 avail_mem=70.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=28 avail_mem=70.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=24 avail_mem=70.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 26.83it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=20 avail_mem=70.44 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=16 avail_mem=70.44 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=12 avail_mem=70.43 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=8 avail_mem=70.43 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s] Capturing num tokens (num_tokens=4 avail_mem=70.42 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=4 avail_mem=70.42 GB): 100%|██████████| 58/58 [00:02<00:00, 30.26it/s]Capturing num tokens (num_tokens=4 avail_mem=70.42 GB): 100%|██████████| 58/58 [00:02<00:00, 27.32it/s]


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
    Generated text:  Ada. I'm a girl with a big smile and I like to be creative. My favorite hobby is playing with my toys. When I'm not playing, I like to read books and listen to music. I love to travel and explore new places.
    I like to spend my time thinking about the world and trying to understand different cultures. I believe in the power of creativity to make the world a better place.
    I am a member of the Big Picture (or Big Picture A) club at my school. In this club, we meet every Wednesday to discuss a different topic that we believe is important to society. Ada enjoys sharing her thoughts
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking an increasing number of foreign assignments. Most of these assignments involve work in foreign countries, but sometimes he will go on one or two overseas trips to see friends. Sometimes the president does not have many or no personal friends, and thus the president may choose to travel abroad for business or political purposes. The president usually chooses his assignments after a careful study of various assignments which he has completed. When he chooses an assignment, he usually takes the time to learn all that he can about the country of which he is a member. The president is busy in the United States and has little time to devote to his trips abroad. This is why it
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is around 2.4 million. The capital of the United Kingdom is London. The population of London is around 8.1 million. The capital of the United States is Washington D. C. The population of Washington D. C. is around 3.9 million. Which of these capitals has the largest population?
    The answer is London. London has the largest population of 8.1 million. Therefore, the answer is London.
    ===============================
    Prompt: The future of AI is
    Generated text:  poised to be shaped by the diversity of its users. As more people get involved in AI research and development, it is more important than ever that the technology aligns with its intended purpose and does not accidentally cause harm. Developers must take into account a wide range of ethical and legal factors, including the potential impact on marginalized communities, the right to privacy, the need for transparency, and the responsibility to prevent AI misuse. By doing so, we can ensure that AI is used for the greater good and that it is built to serve society as a whole. To learn more about the ethical implications of AI, consider attending a conference or workshop focused


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] with a passion for [Interest]. I'm [Hobby] and I enjoy [Favorite Activity]. I'm [Personality] and I'm always [Positive/Positive] about [Things]. I'm [Future Goals] and I'm always [Positive/Positive] about [Things]. I'm [Future Goals] and I'm always [Positive/Positive] about [Things]. I'm [Future Goals] and I'm always [Positive/Positive] about [Things]. I'm [Future Goals] and I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is the largest city in France and one of the most visited cities in the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also home to many international organizations
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for innovation and creativity.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, we will need to address the ethical and privacy concerns that come with it. This could lead to new regulations and standards, as well as new opportunities
    


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
    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to start my new role and learn new things. What can I expect from my time here? I'm looking forward to making a positive impact and contributing to our company's success. How can I be a good team player? I'm always ready to help others and learn from their experiences. What are some of your favorite activities to do with friends and family? I love going hiking and camping with my family and friends. I also love trying new recipes and cooking meals for my family and friends. How do you handle work-related stress? I try
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as “la ville d'été” (the summer capital). Paris is the most populous city in France, with a population of over 2. 4 million as of 2018. The city is the second largest metropolitan area in Europe and the third-largest urban agglomeration in the world. Paris is home to the Louvre Museum, Eiffel Tower, and many other landmarks. It is also the birthplace of numerous notable figures, including the French Revolution, Napoleon Bonaparte, and Gustave Flaubert. The city is known for its rich cultural heritage, fashion, food
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly likely to be characterized by a wide range of potential developments and applications. Here are some of the potential trends that are likely to shape the future of AI:
    
    1. Increased Personalization: AI will continue to become more sophisticated and personal, enabling organizations to create more accurate and relevant algorithms that tailor interactions and products to individual customers.
    
    2. Autonomous Vehicles: Autonomous vehicles are already in the works, with some manufacturers beginning to test these vehicles on public roads. AI will continue to play a crucial role in driving the development of these vehicles, enabling them to navigate more safely and efficiently.
    
    3. Medical Applications: AI will continue to play an increasingly


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

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    Your

     Profession

    /

    Field

    ]

     with

     over

     [

    X

    ]

     years

     of

     experience

    .

     I

    'm

     passionate

     about

     [

    X

    ]

     and

     have

     a

     keen

     interest

     in

     [

    X

    ].

     I

    'm

     also

     [

    X

    ],

     and

     I

     believe

     I

     have

     a

     lot

     to

     offer

     in

     this

     field

    .

     If

     you

    're

     interested

    ,

     I

    'd

     love

     to

     chat

     with

     you

     about

     [

    X

    ].

     [

    Your

     Name

    ]

     [

    Your

     Profession

    /

    Field

    ]

     Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    Your

     Profession

    /

    Field

    ]

     with

     over

     [

    X

    ]

     years

     of

     experience

    .

     I

    'm

     passionate

     about

     [

    X

    ]

     and

     have

     a

     keen

     interest

     in

     [

    X

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     about

     the

     city

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     
    


    To

     elaborate

     on

     this

     statement

    ,

     Paris

     is

     the

     most

     populous

     city

     in

     France

     and

     the

     largest

     city

     in

     Europe

     by

     area

    .

     It

     is

     the

     seat

     of

     the

     Government

     of

     France

     and

     the

     country

    's

     largest

     city

    ,

     containing

     over

     

    2

    .

    2

     million

     people

    .

     The

     city

     is

     known

     for

     its

     many

     historic

     sites

    ,

     museums

    ,

     and

     monuments

    ,

     as

     well

     as

     its

     renowned

     cuisine

    ,

     fashion

    ,

     and

     entertainment

     industry

    .

     It

     is

     also

     the

     world

    's

     

    1

    1

    th

    -largest

     economy

    .

     Paris

     is

     also

     a

     major

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     Overall

    ,

     Paris

     is

     a

     vibrant

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     promising

    ,

     with

     many

     potential

     areas

     of

     growth

     and

     development

    .

     Some

     of

     the

     most

     promising

     areas

     include

    :
    


    1

    .

     Autonomous

     vehicles

    :

     As

     the

     technology

     advances

    ,

     autonomous

     vehicles

     are

     likely

     to

     become

     more

     prevalent

    ,

     offering

     improved

     safety

    ,

     efficiency

    ,

     and

     convenience

    .
    


    2

    .

     Personal

    ized

     healthcare

    :

     AI

     will

     help

     develop

     more

     accurate

     and

     personalized

     medical

     treatments

    ,

     helping

     patients

     receive

     the

     care

     they

     need

    .
    


    3

    .

     Autonomous

     manufacturing

    :

     With

     the

     use

     of

     AI

    ,

     manufacturing

     processes

     can

     be

     optimized

    ,

     reducing

     waste

     and

     improving

     quality

    .
    


    4

    .

     Fraud

     detection

    :

     AI

     is

     already

     being

     used

     to

     detect

     fraud

     and

     protect

     financial

     institutions

     from

     malicious

     attacks

    .
    


    5

    .

     Climate

     change

    :

     AI

     has

     the

     potential

     to

    



```python
llm.shutdown()
```
