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
    [2026-04-22 17:18:46] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.97it/s]


    2026-04-22 17:18:51,186 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 17:18:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 12.97it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.17it/s]

    Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:01, 22.73it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:01, 22.73it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:01, 22.73it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:01, 22.73it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:01, 22.73it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:01, 22.73it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:01, 22.73it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:01, 22.73it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 29.15it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 29.15it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 29.15it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 29.15it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 29.15it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 29.15it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 29.15it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 29.15it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 43.16it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 43.16it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 43.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   7%|▋         | 4/58 [00:00<00:03, 14.15it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 14.44it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 14.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 14.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.57it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.89it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.63it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.63it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.63it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.63it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.63it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.44it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.44it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.44it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.44it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.44it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  50%|█████     | 29/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 30.58it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.55it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.55it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.55it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.55it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.55it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.15it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.15it/s]Capturing num tokens (num_tokens=224 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.15it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.15it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.15it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  71%|███████   | 41/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  71%|███████   | 41/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 32.91it/s]

    Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.26it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.30it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.30it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.30it/s]

    Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.30it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.30it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.41it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.41it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 36.27it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 28.07it/s]


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
    Generated text:  Miguel and I was born and raised in Peru and now I'm living in Houston, Texas. In my spare time I like to read, watch TV shows and movies, and get lost in a good book. I've always been a musician and loved to play the ukulele since I was a kid. I play a variety of styles including jazz, pop, rock, and folk music. I'm also a karaoke champ and I love to perform in front of the crowd. When I'm not playing music, I enjoy spending time with my family, traveling, and enjoying my hobbies.
    I'm excited to bring my love for music
    ===============================
    Prompt: The president of the United States is
    Generated text:  32 years older than the president of Brazil, and the president of Brazil is 2/3 the age of the current president of the United States. If the current president of the United States is 57 years old, how old would the president of Brazil be if he were to retire?
    To determine the age of the president of Brazil, we need to follow the relationships given in the problem step by step.
    
    1. Identify the current age of the president of the United States:
       \[
       \text{Current age of the U.S. president} = 57 \text{ years old}
       \]
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris has many landmarks, including the Eiffel Tower and the Louvre Museum. The Eiffel Tower is a famous landmark in Paris, and it is a symbol of the city. 
    
    The Louvre Museum is another famous landmark in Paris. It is a museum that houses a large collection of art, including paintings, sculptures, and other works of art. The Louvre Museum is a very important and important landmark in Paris.
    
    The Eiffel Tower is located in the city of Paris, while the Louvre Museum is located in the city of Paris as well.
    
    The Eiffel Tower is an iconic landmark in Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  here.
    The world is changing rapidly. New forms of technology are emerging at an unprecedented rate, and technological change is also changing how we work. In a world where information technology (IT) forms the backbone of every business, and where the future of work is becoming increasingly digital, the question of the future of AI must be more complex than ever.
    The future of AI will also be an increasing reliance on data. AI models are learning from data sets and algorithms are learning from data. It will require the ability to capture and process large data sets that are being created at a rate that requires multiple data sources and new ways of working. With


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for fashion, art, and music, and is home to many world-renowned museums, theaters, and restaurants. The city is known for its romantic atmosphere and its role as a cultural and political center of Europe. Paris is a popular tourist destination, with millions of visitors each year. The city is also home to many important institutions of higher education, including the University of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in various industries, including manufacturing, transportation, and healthcare. Automation will likely lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for AI developers and entrepreneurs.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. This includes issues such as bias in AI algorithms, privacy concerns, and the potential
    


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
    Generated text:  __________. I'm a/an (insert profession or job title) who has been dedicated to helping others since I was a child. I've always been passionate about making the world a better place and I'm always trying to learn new skills and techniques to help others achieve their goals. Whether it's through writing, photography, or just being kind, I'm always looking for opportunities to make a difference. So if you need a helping hand, a new perspective, or just someone to smile for, I'm always here to assist you! 😊✨
    
    What does your character specialize in? As a/an (insert profession or job title
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.  
    
    (A) True (B) False
    (A) True
    
    The statement is true. Paris, the largest city in France, is a UNESCO World Heritage site and is known for its iconic Eiffel Tower, as well as its vibrant cultural scene and history. Its rich history, as well as its diverse art, architecture, and cuisine, make it a popular tourist destination, and it plays a significant role in French culture and politics. The city is known for its lively nightlife, and many visitors come to enjoy the city's festivals, cultural events, and gastronomic delights. Despite its challenges, Paris remains an important center of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be complex and multifaceted, with many possible trends shaping the technology's direction. Some possible future trends in AI include:
    
    1. Increased AI transparency and accountability: As AI becomes more integrated into our lives, there is a growing concern about the potential for biased or malicious AI systems. As a result, there is a push towards more transparent and accountable AI that is designed to be trustworthy and equitable.
    
    2. Enhanced AI ethics: As AI becomes more advanced, there will be a need for a set of ethical guidelines and principles to guide the development and deployment of AI systems. This will include considerations of privacy, safety, fairness,


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

     who

     has

     [

    Your

     Personal

     Interest

     or

     Experience

    ]

     in

     the

     [

    Your

     Field

     of

     Study

     or

     Area

     of

     Expert

    ise

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     Certainly

    !

     I

     am

     a

     [

    Your

     Name

    ],

     a

     [

    Your

     Profession

    ],

     dedicated

     to

     [

    Your

     Personal

     Interest

     or

     Experience

    ]

     in

     the

     [

    Your

     Field

     of

     Study

     or

     Area

     of

     Expert

    ise

    ].

     While

     my

     expertise

     lies

     in

     [

    Your

     Field

     of

     Study

     or

     Area

     of

     Expert

    ise

    ],

     I

     am

     always

     open

     to

     learning

     and

     growing

     in

     the

     field

     of

     [

    Your

     Profession

    ].

     If

     you

     have

     any

     questions

     or

     need

     information

     about

     [

    Your

     Field

     of

     Study

     or

     Area

     of

     Expert

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     most

     populous

     city

     in

     France

    ,

     and

     its

     population

     is

     approximately

     

    2

    .

    1

     million

     as

     of

     

    2

    0

    2

    1

    .

     It

     is

     the

     administrative

     and

     cultural

     center

     of

     France

     and

     one

     of

     the

     world

    's

     most

     famous

     cities

    ,

     known

     for

     its

     historical

     architecture

    ,

     fashion

    ,

     cuisine

    ,

     and

     music

    .

     The

     city

     is

     also

     a

     major

     financial

     hub

    ,

     with

     the

     headquarters

     of

     major

     banks

     and

     insurance

     companies

    ,

     as

     well

     as

     the

     headquarters

     of

     many

     large

     companies

    .

     Paris

     is

     a

     major

     tourist

     destination

    ,

     and

     attracts

     millions

     of

     visitors

     each

     year

     for

     its

     beautiful

     architecture

    ,

     art

    ,

     and

     cultural

     attractions

    .

     It

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

     several

     key

     areas

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Increased

     autonomy

     and

     decision

    -making

    :

     With

     the

     development

     of

     AI

    ,

     we

     may

     see

     more

     autonomous

     vehicles

    ,

     robots

    ,

     and

     other

     intelligent

     machines

     taking

     control

     of

     daily

     tasks

    .

     This

     could

     lead

     to

     greater

     freedom

     and

     autonomy

     for

     humans

    .
    


    2

    .

     Integration

     with

     human

     consciousness

    :

     AI

     may

     become

     even

     more

     integrated

     with

     our

     own

     consciousness

    ,

     potentially

     leading

     to

     a

     form

     of

     "

    aut

    onomy

    "

     where

     we

     are

     not

     just

     machines

     but

     are

     also

     conscious

     beings

    .
    


    3

    .

     AI

     ethics

     and

     regulatory

     framework

    :

     The

     development

     of

     AI

     will

     likely

     lead

     to

     more

     stringent

     regulations

     and

     ethical

     standards

    ,

     as

     AI

     systems

    



```python
llm.shutdown()
```
