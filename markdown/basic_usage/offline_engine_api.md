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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.19it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.76it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.76it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.76it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.76it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.41it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.41it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.41it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.41it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:14,  3.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 10.34it/s]

    Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:03, 10.34it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 16.20it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 16.20it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 16.20it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 16.20it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 16.20it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 16.20it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 16.20it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 16.20it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 23.63it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 23.63it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 23.63it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 23.63it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 23.63it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 23.63it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 23.63it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 23.63it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 31.31it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 40.03it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 47.74it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 47.74it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 47.74it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 47.74it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 47.74it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 47.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.70 GB):   2%|▏         | 1/58 [00:00<00:06,  8.51it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.67 GB):   2%|▏         | 1/58 [00:00<00:06,  8.51it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.67 GB):   3%|▎         | 2/58 [00:00<00:06,  8.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.67 GB):   3%|▎         | 2/58 [00:00<00:06,  8.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:05,  9.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:05,  9.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:05,  9.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   9%|▊         | 5/58 [00:00<00:05, 10.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.66 GB):   9%|▊         | 5/58 [00:00<00:05, 10.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.65 GB):   9%|▊         | 5/58 [00:00<00:05, 10.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.65 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.65 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.63it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.65 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.65 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.60it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.64 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.64 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.64 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.64 GB):  21%|██        | 12/58 [00:00<00:02, 16.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.63 GB):  21%|██        | 12/58 [00:00<00:02, 16.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.63 GB):  21%|██        | 12/58 [00:00<00:02, 16.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.63 GB):  21%|██        | 12/58 [00:00<00:02, 16.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.63 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.62 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.62 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.62 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.62 GB):  31%|███       | 18/58 [00:01<00:01, 21.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.61 GB):  31%|███       | 18/58 [00:01<00:01, 21.07it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=55.61 GB):  31%|███       | 18/58 [00:01<00:01, 21.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.59 GB):  31%|███       | 18/58 [00:01<00:01, 21.07it/s]Capturing num tokens (num_tokens=960 avail_mem=55.61 GB):  31%|███       | 18/58 [00:01<00:01, 21.07it/s] Capturing num tokens (num_tokens=960 avail_mem=55.61 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.41it/s]Capturing num tokens (num_tokens=896 avail_mem=55.60 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.41it/s]Capturing num tokens (num_tokens=832 avail_mem=55.60 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.41it/s]Capturing num tokens (num_tokens=768 avail_mem=55.60 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.41it/s]Capturing num tokens (num_tokens=704 avail_mem=55.59 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.41it/s]

    Capturing num tokens (num_tokens=704 avail_mem=55.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.00it/s]Capturing num tokens (num_tokens=640 avail_mem=55.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.00it/s]Capturing num tokens (num_tokens=576 avail_mem=55.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.00it/s]Capturing num tokens (num_tokens=512 avail_mem=55.58 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.00it/s]Capturing num tokens (num_tokens=480 avail_mem=55.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.00it/s]Capturing num tokens (num_tokens=480 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.06it/s]Capturing num tokens (num_tokens=448 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.06it/s]Capturing num tokens (num_tokens=416 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.06it/s]Capturing num tokens (num_tokens=384 avail_mem=55.58 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.06it/s]

    Capturing num tokens (num_tokens=352 avail_mem=55.58 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.06it/s]Capturing num tokens (num_tokens=352 avail_mem=55.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=320 avail_mem=55.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=288 avail_mem=55.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=256 avail_mem=55.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=240 avail_mem=55.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=240 avail_mem=55.57 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=224 avail_mem=55.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=208 avail_mem=55.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=192 avail_mem=55.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.28it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=176 avail_mem=55.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.31it/s]Capturing num tokens (num_tokens=160 avail_mem=55.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.31it/s]Capturing num tokens (num_tokens=144 avail_mem=55.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.31it/s]Capturing num tokens (num_tokens=128 avail_mem=55.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.31it/s]Capturing num tokens (num_tokens=112 avail_mem=55.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.31it/s]Capturing num tokens (num_tokens=112 avail_mem=55.54 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.01it/s]Capturing num tokens (num_tokens=96 avail_mem=55.54 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.01it/s] Capturing num tokens (num_tokens=80 avail_mem=55.54 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.01it/s]Capturing num tokens (num_tokens=64 avail_mem=55.53 GB):  79%|███████▉  | 46/58 [00:02<00:00, 34.01it/s]

    Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  79%|███████▉  | 46/58 [00:02<00:00, 34.01it/s]Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  86%|████████▌ | 50/58 [00:02<00:00, 34.00it/s]Capturing num tokens (num_tokens=32 avail_mem=55.53 GB):  86%|████████▌ | 50/58 [00:02<00:00, 34.00it/s]Capturing num tokens (num_tokens=28 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 34.00it/s]Capturing num tokens (num_tokens=24 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 34.00it/s]Capturing num tokens (num_tokens=20 avail_mem=55.51 GB):  86%|████████▌ | 50/58 [00:02<00:00, 34.00it/s]Capturing num tokens (num_tokens=20 avail_mem=55.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 30.53it/s]Capturing num tokens (num_tokens=16 avail_mem=55.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 30.53it/s]

    Capturing num tokens (num_tokens=12 avail_mem=55.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 30.53it/s]Capturing num tokens (num_tokens=8 avail_mem=55.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 30.53it/s] Capturing num tokens (num_tokens=4 avail_mem=55.50 GB):  93%|█████████▎| 54/58 [00:02<00:00, 30.53it/s]Capturing num tokens (num_tokens=4 avail_mem=55.50 GB): 100%|██████████| 58/58 [00:02<00:00, 31.84it/s]Capturing num tokens (num_tokens=4 avail_mem=55.50 GB): 100%|██████████| 58/58 [00:02<00:00, 24.78it/s]


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
    Generated text:  Hirohisa. I live in Kyoto, a famous city in Japan, where I'm interested in the history and culture of the region. I'm eager to learn more about Japan's unique style of tea drinking, which is characterized by its warm, buttery, and sweet taste. Can you explain the techniques and ingredients that are used in making tea in Japan and how they differ from those used in other countries?
    Certainly! Tea drinking in Japan is a very special art form that has evolved over centuries, and each type of tea has its own unique flavor and history. In Japan, there are many different types of tea, including ool
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she is the leader of the country. The president has a lot of important things to do. He or she has to work hard every day to make the country a better place. The president is always busy. His or her work takes up a lot of people's time. He or she has to make sure that the country's people get the best things for their families. The president has a lot of responsibilities. He or she has to make sure that the country's people are fed well and have a good life. The president has to make sure that his or her country is peaceful and
    ===============================
    Prompt: The capital of France is
    Generated text:  _____. ____
    A. Paris
    B. Toulouse
    C. Toulouse
    D. Lyon
    Answer:
    A
    
    The minimum distance from a plane to a point is called the perpendicular distance from the plane to the point.
    A. Correct
    B. Incorrect
    Answer:
    A
    
    ____ is the highest level of aesthetic education, and it is the foundation of all types of aesthetic education.
    A. Ideological Education
    B. Scientific Education
    C. Moral Education
    D. Aesthetic Education
    Answer:
    D
    
    A and B enter into a contract for the sale of a house. When B sells it to A, A
    ===============================
    Prompt: The future of AI is
    Generated text:  very much in question. The issue of privacy has become a major concern for the public. It is expected that we will see a major shift in the way we use and access technology. With the rise of AI, we may see some major changes in the way we interact with technology. In the following paragraphs, we will discuss the future of AI in the following aspects.
    1. General use of AI in daily life
    AI will become even more widespread in the future. We can expect to see more and more of it being used in our daily lives. For example, we may see more and more AI assistants like Siri, Alexa and Google


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your favorite hobby or activity? I'm always looking for new experiences and I'm always eager to try new things. What's your favorite book or movie? I'm always looking for new experiences and I'm always eager
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is the largest city in France and the second-largest city in the European Union, with a population of over 10 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as its diverse cultural scene, including the annual Eiffel Tower Festival and the annual World Cup football tournament. The city is also home to many world-renowned museums, including the Louvre and the Musée d'Orsay, and is a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and cost-effective solutions, but it could also lead to job displacement for some workers.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, there will be a need to ensure that it is used in a way that respects privacy and
    


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
    Generated text:  [insert name]. I'm a professional journalist and freelance writer with over ten years of experience in publishing. I'm passionate about breaking news and sharing my opinions on the world. I love to challenge opinions and take a stand on important issues. My work has been published in several prestigious journals and online publications. I'm a natural storyteller and have a knack for writing compelling stories that engage readers. My writing style is easy to read and informative, but I'm also able to add unique perspectives and insights to any topic. I'm a very reliable and trustworthy source for any questions or topics related to my field. I'm excited to help you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest and most famous city in the country. 
    
    The city is home to the iconic Eiffel Tower and the Louvre Museum, and is also known for its rich history, including the Siege of 1792 and the creation of the French Revolution. Paris is also home to many famous landmarks and attractions, including the Champs-Élysées, Montmartre, and the Notre-Dame Cathedral. 
    
    Overall, Paris is a bustling city with a rich history and culture that is a major tourist destination for many visitors. Its iconic architecture, beautiful landscapes, and vibrant nightlife attract millions of visitors every year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by an explosion of new technologies and applications, as well as significant advancements in existing technologies. Some possible trends in AI include:
    
    1. Increased focus on ethical considerations: As more AI systems are deployed in various industries and applications, there will be an increased emphasis on ethical considerations. This could lead to a greater focus on creating AI that is not only effective but also ethical, responsible, and transparent.
    
    2. More natural language processing: Natural language processing (NLP) is an area of AI that has been rapidly advancing in recent years. As more AI systems are trained on NLP tasks, we can expect to see more


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

    'm

     a

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

    've

     always

     been

     passionate

     about

     [

    Primary

     Interest

     or

     Hobby

    ],

     and

     I

     recently

     started

     a

     [

    Skill

     or

     Passion

     Project

    ],

     that

    's

     really

     helping

     me

     get

     closer

     to

     my

     goals

    .

     What

    's

     your

     primary

     interest

     or

     hobby

    ?


    I

    'm

     a

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ],

     a

     passionate

     advocate

     for

     [

    Primary

     Interest

     or

     Hobby

    ].

     I

     recently

     started

     a

     [

    Skill

     or

     Passion

     Project

    ]

     that

    's

     helping

     me

     get

     closer

     to

     my

     goals

    .

     What

    's

     your

     primary

     interest

     or

     hobby

    ?

     [

    Name

    ],

     what

    's

     your

     primary

     interest

     or

     hobby

    ?

     #

    Meet

    Your

    Job

    Title

     #

    Personal

    Goals

     #

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     north

    western

     region

     of

     the

     country

     and

     known

     for

     its

     historical

     architecture

    ,

     romantic

     ambiance

    ,

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

     and

     the

     most

     populous

     city

     in

     Europe

    .

     Paris

     is

     also

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     art

     galleries

    ,

     including

     the

     Centre

     Pom

    pid

    ou

    ,

     the

     Lou

    vre

    ,

     and

     the

     Pal

    ais

     Garn

    ier

    .

     It

     is

     the

     center

     of

     the

     French

     economy

     and

     hosts

     the

     headquarters

     of

     many

     major

     companies

     and

     organizations

    .

     Paris

     is

     the

     world

    's

     eighth

    -largest

     city

     and

     is

     renowned

     for

     its

     fashion

     industry

    ,

     gastr

    onomy

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     advancements

     in

     areas

     such

     as

    :
    


    1

    .

     Self

    -driving

     vehicles

    :

     Self

    -driving

     cars

     are

     already

     being

     developed

     and

     are

     expected

     to

     become

     more

     prevalent

     in

     the

     coming

     years

     as

     they

     become

     more

     capable

     of

     driving

     safely

     and

     efficiently

    .
    


    2

    .

     Aug

    mented

     reality

    :

     Aug

    mented

     reality

     is

     already

     being

     used

     in

     gaming

     and

     education

    ,

     and

     is

     expected

     to

     become

     even

     more

     prevalent

     in

     the

     coming

     years

     as

     technology

     advances

    .
    


    3

    .

     Robotics

    :

     Robotics

     is

     already

     being

     used

     in

     a

     variety

     of

     industries

    ,

     from

     manufacturing

     to

     healthcare

    ,

     and

     is

     expected

     to

     continue

     to

     grow

     in

     importance

     as

     technology

     advances

    .
    


    4

    .

     Machine

     learning

    :

     Machine

     learning

     is

     already

     being

     used

     in

     a

     variety

    



```python
llm.shutdown()
```
