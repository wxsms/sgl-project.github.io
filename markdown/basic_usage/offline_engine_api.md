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
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-21 06:55:40] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.15it/s]


    2026-04-21 06:55:45,080 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 06:55:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.01it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.01it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.01it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.01it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.01it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.01it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.01it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:03<00:01, 23.63it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:03<00:00, 32.54it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 41.36it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 52.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 18.05it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.05it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.05it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.05it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  21%|██        | 12/58 [00:00<00:01, 27.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 27.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 27.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 27.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  21%|██        | 12/58 [00:00<00:01, 27.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.14it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.08it/s]Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.08it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.08it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.08it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.08it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.08it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]

    Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.15it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.15it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.15it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.15it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.15it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.50it/s]

    Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.62it/s]

    Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.13it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.13it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.13it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.13it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.13it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.52it/s]

    Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.39it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.39it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.39it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.39it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 34.06it/s]


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
    Generated text:  Sam, and I'm a 17-year-old college student who is looking for a job. I'm looking for a job in the field of accounting. Can you suggest a good college program that I can apply for?
    Yes, I can help you find a good college program for accounting. There are several options available for accounting programs, and you can start by researching them online or contacting them directly. Some popular options for accounting programs include University of Missouri, University of Kentucky, University of Southern Mississippi, and Northeastern University.
    Another option is to consider volunteering or internships at a local accounting firm or charity organization. This can provide you
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. He was born in the United States and has never lived in another country. He is the most powerful and influential person in the world. He is the head of government, and he is the president of the United States. What would happen if the president of the United States was born in another country?
    Well, it's highly unlikely that such a scenario would ever actually occur, as the president of the United States is a man and is born in the United States. However, if the president of the United States were to be born in a different country, it would have significant consequences for the country and its people.
    For one
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which region?
    A. Provence
    B. Provence-Alpes-Côte d'Azur
    C. Provence-Alpes-Roussillon
    D. Languedoc-Roussillon
    
    Answer:
    
    The capital of France is Paris. Therefore, the correct answer is:
    
    Answer: A
    
    As the capital of France, Paris is located in the region of the Provence-Alpes-Côte d'Azur.
    
    A. Provence
    B. Provence-Alpes-Côte d'Azur
    C. Provence-Alpes-Roussillon
    D. Languedoc-Rouss
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is equally expensive to deploy. A new study finds that, at the global level, AI deployment is less expensive than it used to be. On the other hand, deployment can be expensive in the field of AI for example, in the form of developing the specific AI technology or the development of the infrastructure needed to deploy it. This study found that, in the first case, deployment cost is 45 percent of the total deployment cost, in the second case, the cost is 70 percent. The paper is published in the journal Nature.
    The paper, co-authored by the researchers from University of Toronto, University


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


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as its vibrant arts scene and culinary delights. Paris is a popular tourist destination and a major economic hub, with a strong economy and a thriving culture. It is also home to many notable French institutions, including the French Academy of Sciences and the Louvre Museum. The city is known for its fashion industry, with many famous designers and boutiques
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration will enable AI to perform tasks that are currently difficult or impossible for humans to do.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare:
    


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
    Generated text:  [insert name] and I specialize in [insert profession or expertise]. I'm passionate about [insert what you are passionate about] and have been learning about it from a young age. I'm determined to become an expert in my field and continue to learn and grow. What's your name, and what do you do? [Insert name]. (30 seconds) Write in English, no more than 500 words. Hello, my name is [insert name] and I specialize in [insert profession or expertise]. I'm passionate about [insert what you are passionate about] and have been learning about it from a young age
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This is the largest and most important city in France. It is home to many cultural institutions, such as the Louvre, the Musée Rodin, and the Moulin Rouge. It is also a major transportation hub, with many important roads and bridges that connect the city to other parts of France and the world. 
    
    Paris is a city of contrasts, with its elegant architecture, historic landmarks, and vibrant cultural scene. The city is also a popular tourist destination, with millions of visitors each year. It is known for its love of classical music, fashion, and art. 
    
    Overall, Paris is a fascinating city with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by continued advancements in several key areas, including:
    
    1. Enhanced AI capabilities: As AI technology continues to improve, we can expect to see even more sophisticated and capable AI systems. This will likely include more advanced machine learning algorithms, better memory and processing power, and even the ability to interact with humans in complex ways.
    
    2. Increased reliance on AI: As AI becomes more integrated into our daily lives, we can expect to see an increasing reliance on it for a wide range of tasks, from healthcare and education to transportation and transportation management.
    
    3. Increased ethical considerations: With AI systems becoming more sophisticated, we will need


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

     a

     [

    Job

     Title

    ]

     with

     experience

     in

     [

    Experience

    ].

     I

     am

     [

    Age

    ]

     years

     old

     and

     [

    Position

    ]

     in

     [

    Industry

    ]

     role

    .


    I

     am

     [

    Name

    ]

     and

     I

     have

     always

     been

     a

     [

    h

    obby

    /

    interest

    /

    特长

    ]

     lover

    .

     I

     enjoy

     [

    List

     

    3

     hobbies

    /

    interest

    s

    ],

     and

     I

     believe

     in

     [

    Two

     or

     Three

     Words

     That

     Define

     Me

    ].

     What

     interests

     me

     most

     about

     the

     job

     or

     industry

     I

    'm

     in

     is

     [

    Brief

    ly

     explain

     why

    ].


    I

    ’m

     a

     [

    Type

     of

     Person

    ]

     and

     I

     am

     [

    Level

     of

     Hard

     Work

    ]

     in

     [

    Job

     Title

    ].

     I

    'm

     looking

     forward

     to

     [

    Future

     Plans

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Se

    ine

     River

    ,

     known

     for

     its famous

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

     lively

     neighborhoods

     like

     the

     Mar

    ais

     and

     Saint

    -G

    er

    main

    -des

    -

    Pr

    és

    .

     It

    's

     a

     bustling

     met

    ropolis

     with

     a

     rich

     cultural

     heritage

     and

     is

     often

     considered

     the

     heart

     of

     French

     society

    .

     Paris

     is

     renowned

     for

     its

     romantic

     charm

    ,

     iconic

     landmarks

    ,

     and

     annual

     festivals

     such

     as

     the

     E

    iff

    el

     Tower

     Par

    c

     de

     Mars

     and

     the

     International

     Fr

    inge

     Festival

    .

     Paris

     is

     a

     major

     hub

     of

     business

    ,

     politics

    ,

     and

     culture

    ,

     making

     it

     a

     globally

     prominent

     city

    .

     With

     its

     historical

     importance

     and

     cultural

     significance

    ,

     it

     remains

     a

     significant

     influence

     on

     French

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     trends

    ,

     including

    :
    


    1

    .

     Automation

    :

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     it

     is

     likely

     to

     automate

     more

     of

     the

     work

     that

     humans

     do

    ,

     such

     as

     data

     collection

    ,

     analysis

    ,

     and

     decision

    -making

    .

     This

     will

     have

     both positive

     and

     negative

     consequences

    ,

     as

     some

     jobs

     may

     become

     obsolete

     but

     others

     may

     be

     transformed

    .
    


    2

    .

     Personal

    ization

    :

     AI

     is

     increasingly

     being

     used

     to

     personalize

     the

     user

     experience

     by

     analyzing

     data

     and

     identifying

     patterns

    .

     This

     can

     lead

     to

     more

     accurate

     and

     relevant recommendations

    ,

     better

     customer

     service

    ,

     and

     more

     efficient

     use

     of

     resources

    .
    


    3

    .

     Eth

    ical

     considerations

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     a

     growing

    



```python
llm.shutdown()
```
