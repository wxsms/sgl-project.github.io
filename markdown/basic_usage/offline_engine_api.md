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
    [2026-04-17 05:48:20] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.01it/s]


    2026-04-17 05:48:24,522 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 05:48:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.25it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 48.37it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 48.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.82it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.62it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.62it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.62it/s]

    Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.62it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.62it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.83it/s]

    Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]

    Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 28.71it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 28.71it/s]

    Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 28.71it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 28.71it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 28.71it/s]Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 26.65it/s]Capturing num tokens (num_tokens=64 avail_mem=120.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 26.65it/s]

    Capturing num tokens (num_tokens=48 avail_mem=120.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 26.65it/s]Capturing num tokens (num_tokens=32 avail_mem=118.98 GB):  83%|████████▎ | 48/58 [00:01<00:00, 26.65it/s]Capturing num tokens (num_tokens=32 avail_mem=118.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 25.09it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 25.09it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 25.09it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 25.09it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.79it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.79it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.79it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.79it/s] Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.79it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 27.92it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 29.06it/s]


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
    Generated text:  Gary, and I'm a master of the arts of human trafficking. I specialize in assisting individuals or groups in transiting into the United States, including the flow of people seeking work, women trafficked from child sex exploitation, and children trafficked from traffickers and families. My expertise includes initial screening, a staged process, and assistance in the pre-trip and transiting environment.
    My process of screening, which may include background checks, immigration applications, and labor law documentation, ensures that I am vetted thoroughly and ensure a safe environment for trafficking victims. My expertise in pre-trip and transiting environments is specifically focused on assisting victims
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she is the leader of the country. The president has many important things to do every day. He or she is supposed to make important decisions about the country. 
    
    The president has to speak in Congress. They are the leaders of the country. The president says important things like new laws and important things. The president also talks about the future of the country.
    
    When the president is busy working, his or her friends and family members will take care of him or her. But when the president is working and meeting with important people, he or she will have a lot of people around him or
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In which continent does Paris belong to?
    Answer:
    
    Europe
    ===============================
    Prompt: The future of AI is
    Generated text:  where the future of gaming is
    
    A conversation with Andrew Travers, CTO of Telechips
    
    By the end of today, you will know what the future of gaming is going to be. Unfortunately, many of the people who are involved in the development of gaming software, hardware and software are in the field of AI and machine learning. There are now several exciting companies and firms that are dedicated to making the next big thing in AI, a large subset of what Andrew Travers, CTO of Telechips, a company that develops AI algorithms that power smart cars, drones, AR, VR and gaming, is doing.
    
    The


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title] at [company name]. I'm always looking for ways to [job title] and [job title] at [company name], and I'm always eager to learn new things and grow as a professional. I'm a [job title] at [company name], and I'm always looking for ways to [job title] and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also a major financial center and a major tourist destination. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major hub for international business and trade. Paris is a city that is constantly evolving and changing, with new developments and attractions being added regularly. The city is known for its diverse population and cultural diversity, and it is a popular destination for tourists and locals alike
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI systems in the public eye.
    
    3. Increased use of AI for creative tasks: AI is likely
    


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
    Generated text:  [Your Name]. I am an [Occupation] with experience in [Industry/Field]. I have a passion for [Career objective or niche]. I’m excited to meet you and discuss how I can contribute to your team or project.
    
    Yourself:
    
    Good day! I'm [Your Name] from [Your Organisation]. I have a passion for [career objective or niche] and I'm excited to meet you. How are you today? Let's talk about how I can be of service to your team. 
    
    If you have any questions, feel free to reach out. I look forward to hearing from you. Have a great day
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the capital of France, the largest and most populous city in Europe, and a UNESCO World Heritage site. It is the historical and cultural center of France and an international metropolis, known for its grand cathedrals, iconic landmarks, and annual Paris Fashion Week. The city is also a major transportation hub, with extensive rail, road, and airport networks. Paris is an important center for business, culture, and politics, known for its fashion, food, and music scenes. It is also home to numerous international institutions and landmarks, including the Eiffel Tower, Louvre Museum, and the Notre-Dame
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  unpredictable and constantly evolving, but some trends that are likely to shape the future of AI include:
    
      1. Increased reliance on AI for decision-making: As AI technology continues to improve, it is likely to become a more integral part of decision-making processes, particularly in areas such as healthcare, finance, and transportation.
      2. AI will be used in new ways: AI will continue to be used in new ways, such as through the development of new technologies and tools for data analysis and automation.
      3. AI will be used in more diverse applications: AI will likely continue to be used in more diverse applications, including


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

     am

     a

     [

    Your

     Profession

    ]

     with

     a

     passion

     for

     [

    Your

     Hobby

     or

     Interest

    ].

     I

     am

     currently

     [

    Your

     Age

    ],

     [

    Your

     Education

    ],

     and

     [

    Your

     Experience

    ].

     I

     am

     a

     dedicated

     [

    Your

     Profession]

     with

     a

     deep

     appreciation

     for

     [

    Your

     Hobby

     or

     Interest

    ].

     I

     am

     always

     looking

     for

     ways

     to

     [

    Your

     Action

    ],

     and

     I

     am

     eager

     to

     learn

     and

     grow

    .

     What

     makes

     you

     unique

     and

     what

     can

     we

     expect

     from

     us

     today

    ?

     [

    Your

     Name

    ]

     [

    Your

     Profession

    ]

     [

    Your

     Hobby

     or

     Interest

    ]

     Welcome

    ,

     [

    Name

    ]

    !

     It

    's

     a

     pleasure

     to

     meet

     you

    .

     [

    Your

     Name

    ]

     I

     am

     a

     [

    Your

     Profession

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     major

     city

     located

     in

     the

     eastern

     part

     of

     France

     and

     is

     known

     for

     its

     historic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     a

     cultural

     and

     business

     hub

     for

     the

     country

     and

     is

     an

     important

     center

     for

     art

    ,

     fashion

    ,

     and

     cuisine

    .

     It

     has

     a

     vibrant

     urban

     environment

     with

     a

     rich

     history

     and

     diverse

     cultural

     scene

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     and

     a

     major

     cultural

     and

     economic

     center

     of

     the

     world

    .

     

     The

     city

     is

     known

     for

     its

     iconic

     landmarks

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     Paris

     is

     the

     seat

     of

     the

     French

     government

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

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

     Some

     possible

     future

     trends

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     into

     various

     industries

    ,

     leading

     to

     the

     automation

     of

     many

     tasks

     that

     were

     once

     done

     manually

    .

     This

     could

     result

     in

     increased

     efficiency

    ,

     reduced

     costs

    ,

     and

     improved

     productivity

    .
    


    2

    .

     Enhanced

     privacy

    :

     With

     the

     increasing

     amount

     of

     personal

     data

     being

     collected

     and

     shared

     online

    ,

     there

     will

     be

     increased

     pressure

     to

     protect

     and

     manage

     this

     data

    .

     AI

     will

     likely

     become

     more

     transparent

     about

     how

     it

     is

     used

     and

     who

     has

     access

     to

     the

     data

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     will

     continue

     to

     play

     a

    



```python
llm.shutdown()
```
