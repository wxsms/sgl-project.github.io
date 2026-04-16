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
    [2026-04-16 07:47:06] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.47it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.46it/s]


    2026-04-16 07:47:10,885 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 07:47:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:16,  3.09it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:16,  3.09it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:16,  3.09it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:16,  3.09it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:16,  3.09it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:07,  6.04it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 14.71it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 14.71it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 14.71it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 14.71it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 14.71it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 14.71it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 14.71it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 14.71it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:03<00:01, 21.70it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 30.21it/s]

    Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 36.79it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 36.79it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 36.79it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 36.79it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 36.79it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 36.79it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 36.79it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 36.79it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 42.84it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 42.84it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 42.84it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 42.84it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 42.84it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 42.84it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 42.84it/s]

    Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 42.84it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 48.78it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 48.78it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 48.78it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 48.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.17 GB):   3%|▎         | 2/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.16 GB):   3%|▎         | 2/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.16 GB):   3%|▎         | 2/58 [00:00<00:03, 15.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=116.16 GB):   7%|▋         | 4/58 [00:00<00:03, 17.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.16 GB):   7%|▋         | 4/58 [00:00<00:03, 17.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.16 GB):   7%|▋         | 4/58 [00:00<00:03, 17.87it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=116.16 GB):  10%|█         | 6/58 [00:00<00:05,  9.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.16 GB):  10%|█         | 6/58 [00:00<00:05,  9.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.15 GB):  10%|█         | 6/58 [00:00<00:05,  9.98it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=116.15 GB):  14%|█▍        | 8/58 [00:00<00:06,  7.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:06,  7.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.32it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  26%|██▌       | 15/58 [00:01<00:02, 15.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.73 GB):  26%|██▌       | 15/58 [00:01<00:02, 15.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  26%|██▌       | 15/58 [00:01<00:02, 15.16it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  26%|██▌       | 15/58 [00:01<00:02, 15.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.72 GB):  26%|██▌       | 15/58 [00:01<00:02, 15.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.72 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.25it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.25it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.25it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.25it/s]Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.20it/s]Capturing num tokens (num_tokens=832 avail_mem=118.54 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.20it/s]Capturing num tokens (num_tokens=768 avail_mem=118.54 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.20it/s]

    Capturing num tokens (num_tokens=704 avail_mem=118.54 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.20it/s]Capturing num tokens (num_tokens=640 avail_mem=118.53 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.20it/s]Capturing num tokens (num_tokens=640 avail_mem=118.53 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.72it/s]Capturing num tokens (num_tokens=576 avail_mem=118.53 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.72it/s]Capturing num tokens (num_tokens=512 avail_mem=118.52 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.72it/s]Capturing num tokens (num_tokens=480 avail_mem=118.18 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.72it/s]Capturing num tokens (num_tokens=448 avail_mem=118.17 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.72it/s]Capturing num tokens (num_tokens=448 avail_mem=118.17 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.83it/s]Capturing num tokens (num_tokens=416 avail_mem=118.17 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.83it/s]

    Capturing num tokens (num_tokens=384 avail_mem=117.99 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.83it/s]Capturing num tokens (num_tokens=352 avail_mem=117.98 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.83it/s]Capturing num tokens (num_tokens=320 avail_mem=117.98 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.83it/s]Capturing num tokens (num_tokens=320 avail_mem=117.98 GB):  60%|██████    | 35/58 [00:01<00:00, 27.03it/s]Capturing num tokens (num_tokens=288 avail_mem=117.98 GB):  60%|██████    | 35/58 [00:01<00:00, 27.03it/s]Capturing num tokens (num_tokens=256 avail_mem=117.97 GB):  60%|██████    | 35/58 [00:01<00:00, 27.03it/s]Capturing num tokens (num_tokens=240 avail_mem=117.97 GB):  60%|██████    | 35/58 [00:01<00:00, 27.03it/s]Capturing num tokens (num_tokens=240 avail_mem=117.97 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.70it/s]Capturing num tokens (num_tokens=224 avail_mem=117.97 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.70it/s]

    Capturing num tokens (num_tokens=208 avail_mem=117.96 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.70it/s]Capturing num tokens (num_tokens=192 avail_mem=117.96 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.70it/s]Capturing num tokens (num_tokens=192 avail_mem=117.96 GB):  71%|███████   | 41/58 [00:02<00:00, 27.72it/s]Capturing num tokens (num_tokens=176 avail_mem=117.96 GB):  71%|███████   | 41/58 [00:02<00:00, 27.72it/s]Capturing num tokens (num_tokens=160 avail_mem=117.96 GB):  71%|███████   | 41/58 [00:02<00:00, 27.72it/s]Capturing num tokens (num_tokens=144 avail_mem=117.95 GB):  71%|███████   | 41/58 [00:02<00:00, 27.72it/s]Capturing num tokens (num_tokens=128 avail_mem=117.95 GB):  71%|███████   | 41/58 [00:02<00:00, 27.72it/s]Capturing num tokens (num_tokens=128 avail_mem=117.95 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.38it/s]Capturing num tokens (num_tokens=112 avail_mem=117.95 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.38it/s]

    Capturing num tokens (num_tokens=96 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.38it/s] Capturing num tokens (num_tokens=80 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.38it/s]Capturing num tokens (num_tokens=64 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.38it/s]Capturing num tokens (num_tokens=64 avail_mem=117.94 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.55it/s]Capturing num tokens (num_tokens=48 avail_mem=117.94 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.55it/s]Capturing num tokens (num_tokens=32 avail_mem=117.93 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.55it/s]Capturing num tokens (num_tokens=28 avail_mem=117.93 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.55it/s]Capturing num tokens (num_tokens=24 avail_mem=117.92 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.55it/s]Capturing num tokens (num_tokens=24 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.62it/s]Capturing num tokens (num_tokens=20 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.62it/s]

    Capturing num tokens (num_tokens=16 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.62it/s]Capturing num tokens (num_tokens=12 avail_mem=117.91 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.62it/s]Capturing num tokens (num_tokens=8 avail_mem=117.91 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.62it/s] Capturing num tokens (num_tokens=8 avail_mem=117.91 GB):  98%|█████████▊| 57/58 [00:02<00:00, 26.63it/s]Capturing num tokens (num_tokens=4 avail_mem=118.80 GB):  98%|█████████▊| 57/58 [00:02<00:00, 26.63it/s]

    Capturing num tokens (num_tokens=4 avail_mem=118.80 GB): 100%|██████████| 58/58 [00:02<00:00, 21.46it/s]


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
    Generated text:  Juan. I'm a little bit shy. I usually stay at home on weekends because I don't want to do some complicated things. But today I will invite some of my friends over and want to know if they are ready to go. My question is: Can you invite your friends over? Sure, I can invite my friends over. Could you tell me if they are ready to go? I'm just wondering if they are interested in your interests or hobbies. And if so, what are they? Please share your interests with me so I can invite you. That's all for now. Let's make a great day! That sounds
    ===============================
    Prompt: The president of the United States is
    Generated text:  now looking to ensure the next president will be able to effectively address the issues of their time. Which of the following, when added to the sentence to make it true, is most likely to be the answer? 
      
     The president knows that his next term in office will be in the middle of a global pandemic. 
      
     The president knows that his next term in office will be in the middle of a global pandemic that is being caused by the climate crisis. 
      
     The president knows that his next term in office will be in the middle of a global pandemic that is being caused by the climate crisis and that the world is being affected by it. 
      
    
    ===============================
    Prompt: The capital of France is
    Generated text:  a massive building on Île Sainte Victoire. It was the seat of the King of France and the only seat of the French monarchy. It was the seat of the French royal dynasty, and it served as the seat of the French monarchy for two centuries. At the time, it was also the capital of France. In 1792, the population of the city was 125,000. It is now the seat of the Conseil d'État, a body that is responsible for the constitution-making process of France. It is located in the Île de la Cité. It is made
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but one thing is certain: the world is changing. The impact of automation and AI on the workforce is, in many ways, an essential part of the evolving landscape. By understanding these effects, we can better anticipate the challenges that lie ahead and work towards creating a more inclusive and equitable future. In this post, we will examine the impact of automation and AI on the workforce in the US, as well as some of the ways that these technologies are reshaping the workplace. We will also explore the potential benefits and drawbacks of automation and AI, and discuss some of the key challenges that we face in implementing these technologies.
    The impact


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a major international hub for business, politics, and entertainment, and is home to many famous landmarks and attractions. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate tasks that are currently performed by humans, such as data analysis, decision-making, and routine maintenance. This will lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. Improved privacy and security: As AI systems become more sophisticated, they will need to be designed with privacy and security in mind. This will require ongoing research and development to ensure that AI systems are not used to harm or mislead individuals.
    
    3. Enhanced human-computer interaction: AI will continue to improve its ability to interact with humans
    


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
    Generated text:  Sarah and I'm a professional writer with a passion for writing short stories and non-fiction. I enjoy being able to pen some kind of material that people can read and enjoy. I'm excited to chat with you about your own writing needs and passions. How would you like to meet? Be sure to include a brief quote from a famous writer or other source that inspires you to be a writer. Sarah. It sounds like you have a great set of skills for a writer. Can you tell me more about your writing process and what makes you unique in the world of storytelling?
    Certainly! As a professional writer with a passion for writing short stories
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city and the seat of government of the country. It has a rich history, including its romantic past and its influence on the French Revolution. It is famous for its art, culture, and cuisine, and it is home to numerous museums and historical sites. Paris is also known for its opera, ballet, and its many festivals and events throughout the year. Its status as a major international city has made it a cultural hub and a major economic center, making it a global city. Paris is also a major center for fashion and tourism, attracting millions of visitors each year. Paris is the 9th largest city in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by a continued expansion and expansion of its applications, as well as a focus on improving its ethical and moral standards. Here are some possible future trends in AI:
    
    1. Increased integration with other technologies: AI will continue to be integrated with other technologies, such as IoT, blockchain, and virtual reality, to create more complex and flexible systems.
    
    2. Enhanced privacy and security: As AI becomes more prevalent, it will continue to face increased scrutiny and concerns about privacy and security. Governments will continue to develop new privacy and security regulations to protect people's data and prevent misuse of AI systems.
    
    3. Increased use of AI in


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

    insert

     character

     name

    ].

     I

     am

     [

    insert

     character

    's

     age

    ],

     [

    insert

     character

    's

     nationality

    ],

     and

     I

     am

     a

     [

    insert

     character

    's

     occupation

    ]

     from

     [

    insert

     character

    's

     current

     location

    ].

     I

     am

     excited

     to

     meet

     you

     and

     learn

     more

     about

     our

     shared

     interests

     and

     experiences

    .
    


    I

     am

     a

     [

    insert

     character

    's

     profession

    ]

     who

     has

     always

     been

     fascinated

     by

     [

    insert

     one

     or

     two

     things

     that

     interest

     you

    ].

     I

     believe

     that

     it

     is

     my

     duty

     to

     share

     my

     knowledge

     with

     others

    ,

     and

     to

     make

     the

     world

     a

     better

     place

    .

     What

     do

     you

     do

     for

     a

     living

    ?

     
    


    I

     have

     always

     been

     inspired

     by

     the

     idea

     of

     [

    insert

     one

     or

     two

     things

     that

     inspire

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     love

     and

     war

    ,

     a

     UNESCO

     World

     Heritage

     site

    .

     Known

     for

     its

     

    1

    9

    th

    -century

     architecture

    ,

     it

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

     and

     the

     nation

    's

     cultural

    ,

     economic

     and

     political

     center

    .

     Its

     unique

     blend

     of

     traditional

     French

     culture

     and

     modern

    ity

     has

     made

     it

     an

     important

     part

     of

     French

     identity

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     drawing

     millions

     of

     visitors

     each

     year

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     historic

     neighborhoods

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     advancements

     in

     areas

     such

     as

     deep

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

    .

     These

     technologies

     will

     likely

     be

     used

     in

     a

     wide

     range

     of

     applications

    ,

     from

     healthcare

     and

     finance

     to

     transportation

     and

     manufacturing

    .

     AI

     will

     also

     become

     more

     integrated

     into

     everyday

     life

    ,

     as

     more

     people

     will

     use

     it

     to

     perform

     routine

     tasks

     and

     access

     information

    .

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     will

     likely

     see

     a

     shift

     towards

     more

     personalized

     and

     adaptive

     technologies

    ,

     which

     will

     require

     a

     new

     level

     of

     understanding

     and

     management

    .

     Additionally

    ,

     AI

     will

     likely

     continue

     to

     evolve

     in

     response

     to

     new

     challenges

    ,

     such

     as

     ethical

     concerns

     and

     the

     potential

     for

     AI

     to

     create

     new

     forms

     of

    



```python
llm.shutdown()
```
