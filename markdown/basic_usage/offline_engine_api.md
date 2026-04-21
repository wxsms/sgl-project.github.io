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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-21 03:13:43] `torch_dtype` is deprecated! Use `dtype` instead!
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.05it/s]


    2026-04-21 03:13:48,310 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 03:13:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.77it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.77it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.50it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 19.58it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]

    Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 28.64it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.25it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.25it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.25it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.25it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.25it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.25it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.25it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 34.25it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 36.60it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 36.60it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 36.60it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 36.60it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 36.60it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 36.60it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 36.60it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 36.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.64 GB):   3%|▎         | 2/58 [00:00<00:03, 15.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.64 GB):   3%|▎         | 2/58 [00:00<00:03, 15.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.64 GB):   3%|▎         | 2/58 [00:00<00:03, 15.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.64 GB):   3%|▎         | 2/58 [00:00<00:03, 15.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.64 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.63 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.63 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.63 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.38it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=118.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.62 GB):  21%|██        | 12/58 [00:00<00:01, 28.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.59 GB):  21%|██        | 12/58 [00:00<00:01, 28.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.72it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=118.31 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.54 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=960 avail_mem=118.55 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.90it/s] Capturing num tokens (num_tokens=896 avail_mem=118.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.90it/s]

    Capturing num tokens (num_tokens=832 avail_mem=118.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=832 avail_mem=118.35 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.00it/s]Capturing num tokens (num_tokens=768 avail_mem=118.37 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.00it/s]Capturing num tokens (num_tokens=704 avail_mem=118.37 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.00it/s]Capturing num tokens (num_tokens=640 avail_mem=118.38 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.00it/s]Capturing num tokens (num_tokens=640 avail_mem=118.38 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=576 avail_mem=118.40 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=512 avail_mem=118.41 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.71it/s]

    Capturing num tokens (num_tokens=480 avail_mem=118.54 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=480 avail_mem=118.54 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=448 avail_mem=118.52 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=416 avail_mem=118.51 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=384 avail_mem=118.51 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=352 avail_mem=118.50 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=352 avail_mem=118.50 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.94it/s]Capturing num tokens (num_tokens=320 avail_mem=118.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.94it/s]Capturing num tokens (num_tokens=288 avail_mem=118.48 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.94it/s]

    Capturing num tokens (num_tokens=256 avail_mem=117.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.94it/s]Capturing num tokens (num_tokens=240 avail_mem=117.09 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.94it/s]Capturing num tokens (num_tokens=240 avail_mem=117.09 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=224 avail_mem=117.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=208 avail_mem=117.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=192 avail_mem=117.09 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=176 avail_mem=117.09 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=176 avail_mem=117.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.98it/s]Capturing num tokens (num_tokens=160 avail_mem=117.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.98it/s]Capturing num tokens (num_tokens=144 avail_mem=117.07 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.98it/s]

    Capturing num tokens (num_tokens=128 avail_mem=117.07 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.98it/s]Capturing num tokens (num_tokens=112 avail_mem=117.06 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.98it/s]Capturing num tokens (num_tokens=112 avail_mem=117.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=96 avail_mem=117.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.97it/s] Capturing num tokens (num_tokens=80 avail_mem=117.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=64 avail_mem=117.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=48 avail_mem=117.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=48 avail_mem=117.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=32 avail_mem=117.04 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=28 avail_mem=117.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.72it/s]

    Capturing num tokens (num_tokens=24 avail_mem=117.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=20 avail_mem=117.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=20 avail_mem=117.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.77it/s]Capturing num tokens (num_tokens=16 avail_mem=117.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.77it/s]Capturing num tokens (num_tokens=12 avail_mem=117.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.77it/s]Capturing num tokens (num_tokens=8 avail_mem=117.00 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.77it/s] Capturing num tokens (num_tokens=4 avail_mem=116.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.77it/s]Capturing num tokens (num_tokens=4 avail_mem=116.97 GB): 100%|██████████| 58/58 [00:01<00:00, 35.42it/s]Capturing num tokens (num_tokens=4 avail_mem=116.97 GB): 100%|██████████| 58/58 [00:01<00:00, 29.76it/s]


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
    Generated text:  Emma and I'm currently 15 years old. I'm interested in music and am excited to share my music with you. I'm a big fan of hip-hop music and I'm going to learn to play the guitar. My name is 944105 and I'm from a small village in Russia. What's your name? My name is Emma. What's your name? Oh, you're from a small village in Russia! That's exciting! What kind of music do you like to listen to? I like hip-hop music and I'm really excited to learn to play the guitar. Do you have any
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 43 years old. If the president of the United States was 36 years old 6 years ago, how many years later is the president of the United States than he is now?
    
    To determine how many years later the president of the United States is than he is now, we need to follow these steps:
    
    1. **Identify the current age of the president:**
       The president is currently 43 years old.
    
    2. **Determine the age of the president 6 years ago:**
       The president was 36 years old 6 years ago.
    
    3. **Calculate the number of years
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the capital of France. Paris is in the south of France. Paris is in a very beautiful part of France. This beautiful part of France is called the Seine valley. The Seine is the longest river in France. In Paris, you can see many big buildings like the Louvre and the Notre Dame Cathedral. They are very famous places. The Seine is very long and wide. It has many bridges and canals. The Seine is a very important part of Paris. It can help the people to live in the city. The Seine valley is very special. It is in the middle of Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  undeniably bright, but we must also be aware that it's not a panacea and should not be seen as the end goal. To achieve AI's full potential, it requires a holistic approach that goes beyond the technical aspects of development. One approach that could help achieve this goal is incorporating ethical considerations into the development of AI. Ethical considerations can help ensure that AI is developed in a way that respects the rights and dignity of all individuals involved, including those with disabilities, those from marginalized groups, and those with varying levels of intelligence and knowledge. By incorporating ethical considerations into AI development, we can create a more inclusive and just society


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is home to many world-renowned museums and art galleries, including the Louvre and the Musée d'Orsay. Paris is a city of contrasts, with its modern architecture and historical landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose diseases, predict patient outcomes, and personalize treatment plans. As AI technology continues to improve, we can expect to see even more advanced applications in healthcare, such as personalized medicine, drug discovery, and patient monitoring.
    
    2. AI in finance: AI is already being used in finance to automate trading, fraud detection, and risk management. As AI technology continues to improve, we can expect to see even
    


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
    Generated text:  [Name], and I'm a [Occupation] with [Number] years of experience in [Industry]!
    
    I'm an energetic and self-motivated individual who thrives on the challenge of tackling complex problems and working with clients to create meaningful solutions. Whether it's designing innovative software solutions, leading business transformation initiatives, or mentoring underperforming colleagues, I believe in the power of applying my unique perspective and skills to drive positive change.
    
    I am a compassionate, empathetic leader who is always ready to listen and support my team, fostering a positive and inclusive work environment. I believe in the value of continuous learning and growth, and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a city renowned for its rich history, architecture, and vibrant cultural scene. 
    
    - The Louvre Museum, with its stunning 17th-century paintings and treasures, is a major attraction in Paris.
    - The Eiffel Tower, a symbol of Paris, offers panoramic views of the city.
    - The Notre-Dame Cathedral, a UNESCO World Heritage site, has intricate stained-glass windows and a famous bell tower.
    - The Arc de Triomphe, a monumental monument to France's military victories, is a must-see.
    - The Champs-Élysées, a lively avenue with over 100
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with many possibilities and potential. Here are some of the trends that we can expect to see in the near future:
    
    1. Increased focus on ethical AI: As concerns over AI ethics and the impact of AI on society grow, we can expect to see a greater emphasis on ethical guidelines and safeguards for AI systems. This could lead to the development of more transparent and accountable AI systems that are designed to minimize biases and risks.
    
    2. AI-based healthcare: AI is already being used in healthcare to help diagnose and treat diseases, but there is significant potential for further development. We could see the development of more AI-powered tools that can assist


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

    'm

     [

    Age

    ]

     years

     old

    .

     I

    'm

     a

     [

    Occup

    ation

    ]

     with

     a

     passion

     for

     [

    Occup

    ation

    ].

     I

     enjoy

     [

    Occup

    ation

    ]

     because

     [

    Reason

     for

     Interest

    ].

     I

     recently

     moved

     to

     [

    City

    /

    State

    ]

     from

     [

    Previous

     City

    /

    State

    ],

     and

     I

    've

     been

     living

     here

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

    'm

     a

     [

    Occup

    ation

    ]

     with

     a

     [

    H

    obby

    /

    Interest

    /

    Dis

    qualification

    ]

     that

     I

    'm

     passionate

     about

    .

     I

    'm

     [

    What

     I

     Do

     for

     Fun

    ].

     I

     love

     [

    Fun

     Activity

    ].

     I

     hope

     you

     can

     see

     me

     as

     someone

     who

     has

     a

     unique

     combination

     of

     interests

     and

     passions

     that

     make

     me

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     stunning

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     historical

     landmarks

    .

     Paris

     is

     also

     a

     world

    -ren

    owned

     fashion

     capital

     and

     home

     to

     the

     Lou

    vre

     Museum

     and

     the

     Museum

     of

     Modern

     Art

    .

     The

     city

     is

     home

     to

     the

     French

     Parliament

     building

    ,

     and

     has

     a

     long

     history

     of

     architecture

     and

     art

    ,

     making

     it

     a

     major

     hub

     for

     the

     arts

     and

     culture

     industry

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     has

     become

     synonymous

     with

     Paris

    ian

     culture

     and

     sophistication

    .

     The

     French

     capital

     is

     known

     for

     its

     cultural

     richness

     and

     rich

     history

    .

     The

     capital

     of

     France

     has

     a

     long

     and

     stor

    ied

     history

    ,

     and

     it

     is

     one

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     trends

     that

     will

     shape

     its

     development

     and

     impact

     on

     society

    .

     Here

     are

     some

     of

     the

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     can

     be

     used

     to

     improve

     diagnosis

    ,

     treatment

    ,

     and

     patient

     outcomes

     in

     the

     healthcare

     industry

    .

     For

     example

    ,

     AI

     can

     be

     used

     to

     analyze

     medical

     images

    ,

     identify

     patterns

     in

     patient

     data

    ,

     and

     predict

     disease

     outbreaks

    .

     In

     addition

    ,

     AI

     can

     be

     used

     to

     automate

     routine

     tasks

    ,

     such

     as

     drug

     development

     and

     patient

     billing

    .
    


    2

    .

     Autonomous

     vehicles

    :

     With

     the

     increasing

     number

     of

     accidents

     and

     fatalities

     in

     the

     automobile

     industry

    ,

     there

     is

     a

     growing

     need

     for

     autonomous

     vehicles

     that

     can

    



```python
llm.shutdown()
```
