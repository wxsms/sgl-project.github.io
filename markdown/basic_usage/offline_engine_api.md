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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.94it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.93it/s]


    2026-05-12 04:19:23,245 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 04:19:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:12,  3.85it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:12,  3.85it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:12,  3.85it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:12,  3.85it/s]

    Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:12,  3.85it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  6.62it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  6.62it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  6.62it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  6.62it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  6.62it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:06,  6.62it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:06,  6.62it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 11.84it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 11.84it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 11.84it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 11.84it/s] 

    Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:03, 11.84it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:03, 11.84it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:03, 11.84it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 17.74it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 17.74it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 17.74it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 17.74it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:01, 17.74it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:01, 17.74it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:01, 17.74it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 23.42it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 23.42it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 23.42it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 23.42it/s]

    Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 23.42it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 23.42it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 23.42it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 23.42it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 30.84it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 30.84it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 30.84it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 30.84it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 30.84it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 30.84it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 30.84it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 30.84it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 37.38it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 37.38it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 37.38it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 37.38it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 37.38it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 37.38it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 37.38it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 41.53it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 41.53it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 41.53it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 41.53it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 41.53it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 41.53it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 41.53it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 41.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.85 GB):   2%|▏         | 1/58 [00:00<00:07,  7.69it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.58 GB):   2%|▏         | 1/58 [00:00<00:07,  7.69it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=69.58 GB):   3%|▎         | 2/58 [00:00<00:06,  8.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.63 GB):   3%|▎         | 2/58 [00:00<00:06,  8.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.63 GB):   5%|▌         | 3/58 [00:00<00:06,  8.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.78 GB):   5%|▌         | 3/58 [00:00<00:06,  8.72it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=69.78 GB):   7%|▋         | 4/58 [00:00<00:06,  8.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.77 GB):   7%|▋         | 4/58 [00:00<00:06,  8.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.76 GB):   7%|▋         | 4/58 [00:00<00:06,  8.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.76 GB):  10%|█         | 6/58 [00:00<00:04, 11.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.63 GB):  10%|█         | 6/58 [00:00<00:04, 11.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.74 GB):  10%|█         | 6/58 [00:00<00:04, 11.16it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=69.74 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.21 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.22 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.22 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.30 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.62it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.30 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.29 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.29 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.24 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.75it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.24 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.24 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.24 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.27 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.04it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.26 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.25 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.25 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.25 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.22 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.55it/s]Capturing num tokens (num_tokens=960 avail_mem=74.24 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.55it/s] Capturing num tokens (num_tokens=960 avail_mem=74.24 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.80it/s]Capturing num tokens (num_tokens=896 avail_mem=74.22 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.80it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.23 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.80it/s]Capturing num tokens (num_tokens=768 avail_mem=74.22 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.80it/s]Capturing num tokens (num_tokens=768 avail_mem=74.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.06it/s]Capturing num tokens (num_tokens=704 avail_mem=74.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.06it/s]Capturing num tokens (num_tokens=640 avail_mem=74.19 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.06it/s]Capturing num tokens (num_tokens=576 avail_mem=74.20 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.06it/s]Capturing num tokens (num_tokens=576 avail_mem=74.20 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.08it/s]Capturing num tokens (num_tokens=512 avail_mem=74.18 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.08it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.08it/s]Capturing num tokens (num_tokens=448 avail_mem=74.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.08it/s]Capturing num tokens (num_tokens=448 avail_mem=74.17 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.09it/s]Capturing num tokens (num_tokens=416 avail_mem=74.16 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.09it/s]Capturing num tokens (num_tokens=384 avail_mem=74.16 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.09it/s]Capturing num tokens (num_tokens=352 avail_mem=74.15 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.09it/s]Capturing num tokens (num_tokens=352 avail_mem=74.15 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.77it/s]Capturing num tokens (num_tokens=320 avail_mem=74.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.77it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.14 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.77it/s]Capturing num tokens (num_tokens=256 avail_mem=74.14 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.77it/s]Capturing num tokens (num_tokens=256 avail_mem=74.14 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.62it/s]Capturing num tokens (num_tokens=240 avail_mem=74.13 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.62it/s]Capturing num tokens (num_tokens=224 avail_mem=74.14 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.62it/s]Capturing num tokens (num_tokens=208 avail_mem=74.13 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.62it/s]Capturing num tokens (num_tokens=192 avail_mem=74.11 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.62it/s]Capturing num tokens (num_tokens=192 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:01<00:00, 29.14it/s]Capturing num tokens (num_tokens=176 avail_mem=74.10 GB):  71%|███████   | 41/58 [00:01<00:00, 29.14it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.12 GB):  71%|███████   | 41/58 [00:02<00:00, 29.14it/s]Capturing num tokens (num_tokens=144 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:02<00:00, 29.14it/s]Capturing num tokens (num_tokens=128 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:02<00:00, 29.14it/s]Capturing num tokens (num_tokens=128 avail_mem=74.11 GB):  78%|███████▊  | 45/58 [00:02<00:00, 31.66it/s]Capturing num tokens (num_tokens=112 avail_mem=74.10 GB):  78%|███████▊  | 45/58 [00:02<00:00, 31.66it/s]Capturing num tokens (num_tokens=96 avail_mem=74.10 GB):  78%|███████▊  | 45/58 [00:02<00:00, 31.66it/s] Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  78%|███████▊  | 45/58 [00:02<00:00, 31.66it/s]Capturing num tokens (num_tokens=64 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:02<00:00, 31.66it/s]Capturing num tokens (num_tokens=64 avail_mem=74.08 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.36it/s]Capturing num tokens (num_tokens=48 avail_mem=74.08 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.36it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.36it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.36it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.36it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.82it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.82it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.82it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.82it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.82it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  91%|█████████▏| 53/58 [00:02<00:00, 32.82it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:02<00:00, 35.85it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:02<00:00, 23.59it/s]


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
    Generated text:  Alan Green and I am a software developer at Bullseye Research. I have been working with Ruby and JavaScript for many years and am now in a more advanced state of learning. I have been working in JavaScript for the last few months, but I have started to get more involved in Ruby and Ruby on Rails lately. I have a lot of questions about Ruby and Rails, and I would like to have a conversation with you to share my questions and my experiences with Ruby and Rails.
    
    Sure, I'd be happy to have a conversation with you about Ruby and Rails. Can you tell me about your background and any experiences you have with Ruby
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. He is so busy that he has to be out for two weeks each year. He has to travel to 35 countries around the world to work. When he is not traveling, he reads about how his country is doing and he tries to learn something new. He has been there for five years and he loves every country he visits! What's he going to do next? The answer is that he will go to visit his home country! But he has to be home for two weeks. After that, he will fly to the United States, stay in a big hotel and eat lots of yummy food. Then he will
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lyon
    C. Marseille
    D. Dijon
    Answer:
    A
    
    When did the Golden Era of the West in the Middle Ages begin?
    A. 10th century
    B. 11th century
    C. 12th century
    D. 13th century
    Answer:
    B
    
    In the development of the European feudal society, the main functions of the 'prime minister' were mainly to ____
    A. manage military affairs
    B. manage ecclesiastical affairs
    C. manage domestic affairs
    D. manage foreign relations
    Answer:
    C
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  very bright. In fact, the potential of AI technology is more than we can imagine. With the rapid increase of data, the need for more advanced processing power and computing speed, the acceleration of AI algorithms, the development of AI applications, and the introduction of AI technologies into our daily lives, it is clear that the application of AI technology has reached a new level. As the two main players in the field of AI, Google and Microsoft, have entered the domain of creating new AI applications and technologies. Can we expect to see more and more innovative and advanced AI applications in the near future?
    
    Google, the leader in AI, has already


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, known for its rich history, art, and cuisine. It is also home to the French Riviera, a popular tourist destination. The city is known for its diverse population, including French, Spanish, Italian, and other nationalities. Paris is a major transportation hub, with the Eiffel Tower serving as a landmark and the Louvre Museum serving as a cultural institution
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased Use of AI in Finance: AI is already being used in
    


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
    Generated text:  [Name] and I'm a computer programmer working for [Company name]. I have a passion for [mention a relevant skill or hobby], and I enjoy helping [mention a relevant project]. I'm excited to work with a diverse team and contribute to [mention a relevant project]. What excites you most about programming? I love solving complex problems and turning them into something that can be useful and easy to use for others. I'm confident that I can make a valuable contribution to your team, and I'm looking forward to a chance to learn more about [Company name]. Looking forward to our meeting! [Name] [Company name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and is home to the Eiffel Tower, the Louvre Museum, and other cultural landmarks. Paris is known for its rich history, architecture, and lively atmosphere, and it is often referred to as the "City of Love" due to its romantic architecture. It is also home to many famous French artists, writers, and composers, including the painter Monet and the composer Beethoven. Paris is a significant cultural hub and is recognized for hosting numerous world-renowned events, including the Paris International Film Festival and the Summer Olympics. Its modern skyline and beautiful views of the Seine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some possible trends to consider:
    
    1. Increased automation: As AI becomes more advanced, we may see more automation of tasks, including jobs, in various industries.
    
    2. Personalization: AI will enable machines to learn and adapt to individual users, leading to more personalized and accurate predictions and recommendations.
    
    3. Advancements in ethics and privacy: There is a growing need for AI to be developed and used with a focus on ethical considerations and privacy concerns.
    
    4. Cognitive enhancement: AI will continue to advance, leading to faster and more accurate decision-making.
    
    5. AI in health care: AI will play a crucial


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

    career

    ]

     with

     [

    years

     of

     experience

    ].

     I

    'm

     a

     [

    profession

    ]

     with

     [

    salary

    ]

     per

     [

    week

    ].

     My

     specialty

     is

     [

    your

     specialty

    ,

     e

    .g

    .,

     teaching

    ,

     writing

    ,

     or

     research

    ].

     I

    'm

     excited

     to

     work

     with

     you

     and

     learn

     something

     new

     today

    .

     How

     can

     I

     assist

     you

    ?

     [

    Name

    ]

     [

    Contact

     Information

    ]

     [

    Experience

    ]

     [

    Education

    ]

     [

    Skills

    ]

     [

    Professional

     Experience

    ]

     [

    Research

     and

     Publications

    ]

     [

    Professional

     Awards

    ]

     [

    Professional

     Contributions

    ]

     [

    Professional

     Contributions

     to

     Acad

    emia

    ]

     [

    Professional

     Contributions

     to

     Industry

    ]

     [

    Professional

     Contributions

     to

     Non

    -profit

     Organizations

    ]

     [

    Professional

     Contributions

     to

     Education

    ]

     [

    Professional

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     “

    City

     of

     Love

    ”

     for

     its

     cultural

     richness

    ,

     romantic

     ambiance

    ,

     and

     fashion

     scene

    .

     Paris

     is

     located

     in

     the

     Lo

    ire

     Valley

     region

    ,

     about

     

    6

    0

     miles

     south

     of

     Paris

    ,

     and

     is

     the

     largest

     city

     in

     the

     French

     region

     of

     the

     same

     name

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

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

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     for

     its

     gastr

    onomic

     delights

    ,

     such

     as

     classic

     French

     cuisine

     and

     elegant

     restaurants

    .

     It

     is

     also

     home

     to

     the

     Luxembourg

     Palace

    ,

     a

     former

     royal

     residence

     that

     features

     beautiful

     gardens

     and

     a

     picturesque

     view

     of

     the

     Se

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancement

     and

     integration

    ,

     with

     the

     following

     possible

     future

     trends

    :
    


    1

    .

     Increased

     complexity

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     machine

     learning

    ,

     computer

     vision

    ,

     and

     natural

     language

     processing

    ,

     the

     complexity

     of

     the

     AI

     system

     is

     likely

     to

     increase

    .

     This

     can

     lead

     to

     more

     sophisticated

     and

     robust

     AI

     systems

     that

     can

     handle

     increasingly

     complex

     and

     unpredictable

     situations

    .
    


    2

    .

     Enhanced

     human

    -machine

     collaboration

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     there

     is

     a

     potential

     for

     human

    -machine

     collaboration

     to

     improve

     the

     efficiency

     and

     effectiveness

     of

     AI

     systems

    .

     For

     example

    ,

     AI

    -powered

     tools

     could

     assist

     humans

     in

     tasks

     that

     are

     currently

     done

     by

     humans

    ,

     such

     as

    



```python
llm.shutdown()
```
