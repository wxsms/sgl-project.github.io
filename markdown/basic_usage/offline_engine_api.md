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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.32it/s]


    2026-05-02 13:01:49,919 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 13:01:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.70it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.70it/s]

    Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 12.45it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 12.45it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 12.45it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 12.45it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:02, 12.45it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:02, 12.45it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:04<00:02, 12.45it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 12.45it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 17.75it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 17.75it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 17.75it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 17.75it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 17.75it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 17.75it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 17.75it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 17.75it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.78it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]

    Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 38.11it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 38.11it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 38.11it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 38.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=36.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=36.09 GB):   2%|▏         | 1/58 [00:00<00:06,  9.28it/s]Capturing num tokens (num_tokens=7680 avail_mem=36.06 GB):   2%|▏         | 1/58 [00:00<00:06,  9.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=36.03 GB):   2%|▏         | 1/58 [00:00<00:06,  9.28it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=36.03 GB):   5%|▌         | 3/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=36.03 GB):   5%|▌         | 3/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=36.03 GB):   5%|▌         | 3/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=36.02 GB):   5%|▌         | 3/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=36.02 GB):  10%|█         | 6/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=36.01 GB):  10%|█         | 6/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=36.01 GB):  10%|█         | 6/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=36.01 GB):  10%|█         | 6/58 [00:00<00:02, 19.52it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=36.01 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=35.95 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=35.94 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=35.94 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=35.94 GB):  21%|██        | 12/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=35.94 GB):  21%|██        | 12/58 [00:00<00:02, 22.15it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=35.92 GB):  21%|██        | 12/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=35.90 GB):  21%|██        | 12/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=35.90 GB):  26%|██▌       | 15/58 [00:00<00:02, 17.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=35.89 GB):  26%|██▌       | 15/58 [00:00<00:02, 17.37it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=35.88 GB):  26%|██▌       | 15/58 [00:00<00:02, 17.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=35.88 GB):  29%|██▉       | 17/58 [00:00<00:02, 16.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=35.86 GB):  29%|██▉       | 17/58 [00:00<00:02, 16.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=35.85 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.15it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=35.85 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=35.84 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=35.81 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=35.81 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.01it/s]Capturing num tokens (num_tokens=960 avail_mem=35.82 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.01it/s] Capturing num tokens (num_tokens=896 avail_mem=35.81 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.01it/s]

    Capturing num tokens (num_tokens=896 avail_mem=35.81 GB):  40%|███▉      | 23/58 [00:01<00:02, 15.95it/s]Capturing num tokens (num_tokens=832 avail_mem=35.81 GB):  40%|███▉      | 23/58 [00:01<00:02, 15.95it/s]Capturing num tokens (num_tokens=768 avail_mem=35.80 GB):  40%|███▉      | 23/58 [00:01<00:02, 15.95it/s]Capturing num tokens (num_tokens=768 avail_mem=35.80 GB):  43%|████▎     | 25/58 [00:01<00:02, 15.70it/s]Capturing num tokens (num_tokens=704 avail_mem=35.30 GB):  43%|████▎     | 25/58 [00:01<00:02, 15.70it/s]

    Capturing num tokens (num_tokens=640 avail_mem=35.30 GB):  43%|████▎     | 25/58 [00:01<00:02, 15.70it/s]Capturing num tokens (num_tokens=640 avail_mem=35.30 GB):  47%|████▋     | 27/58 [00:01<00:02, 12.11it/s]Capturing num tokens (num_tokens=576 avail_mem=35.30 GB):  47%|████▋     | 27/58 [00:01<00:02, 12.11it/s]

    Capturing num tokens (num_tokens=512 avail_mem=35.28 GB):  47%|████▋     | 27/58 [00:01<00:02, 12.11it/s]Capturing num tokens (num_tokens=512 avail_mem=35.28 GB):  50%|█████     | 29/58 [00:02<00:02, 10.24it/s]Capturing num tokens (num_tokens=480 avail_mem=35.29 GB):  50%|█████     | 29/58 [00:02<00:02, 10.24it/s]

    Capturing num tokens (num_tokens=448 avail_mem=35.29 GB):  50%|█████     | 29/58 [00:02<00:02, 10.24it/s]Capturing num tokens (num_tokens=448 avail_mem=35.29 GB):  53%|█████▎    | 31/58 [00:02<00:02,  9.60it/s]Capturing num tokens (num_tokens=416 avail_mem=35.29 GB):  53%|█████▎    | 31/58 [00:02<00:02,  9.60it/s]

    Capturing num tokens (num_tokens=384 avail_mem=35.29 GB):  53%|█████▎    | 31/58 [00:02<00:02,  9.60it/s]Capturing num tokens (num_tokens=384 avail_mem=35.29 GB):  57%|█████▋    | 33/58 [00:02<00:02,  9.48it/s]Capturing num tokens (num_tokens=352 avail_mem=35.28 GB):  57%|█████▋    | 33/58 [00:02<00:02,  9.48it/s]Capturing num tokens (num_tokens=320 avail_mem=35.28 GB):  57%|█████▋    | 33/58 [00:02<00:02,  9.48it/s]Capturing num tokens (num_tokens=288 avail_mem=35.27 GB):  57%|█████▋    | 33/58 [00:02<00:02,  9.48it/s]

    Capturing num tokens (num_tokens=288 avail_mem=35.27 GB):  62%|██████▏   | 36/58 [00:02<00:01, 12.73it/s]Capturing num tokens (num_tokens=256 avail_mem=35.27 GB):  62%|██████▏   | 36/58 [00:02<00:01, 12.73it/s]Capturing num tokens (num_tokens=240 avail_mem=35.27 GB):  62%|██████▏   | 36/58 [00:02<00:01, 12.73it/s]Capturing num tokens (num_tokens=224 avail_mem=35.26 GB):  62%|██████▏   | 36/58 [00:02<00:01, 12.73it/s]Capturing num tokens (num_tokens=208 avail_mem=35.26 GB):  62%|██████▏   | 36/58 [00:02<00:01, 12.73it/s]Capturing num tokens (num_tokens=208 avail_mem=35.26 GB):  69%|██████▉   | 40/58 [00:02<00:01, 17.18it/s]

    Capturing num tokens (num_tokens=192 avail_mem=55.56 GB):  69%|██████▉   | 40/58 [00:02<00:01, 17.18it/s]Capturing num tokens (num_tokens=176 avail_mem=55.56 GB):  69%|██████▉   | 40/58 [00:02<00:01, 17.18it/s]Capturing num tokens (num_tokens=160 avail_mem=55.55 GB):  69%|██████▉   | 40/58 [00:02<00:01, 17.18it/s]Capturing num tokens (num_tokens=160 avail_mem=55.55 GB):  74%|███████▍  | 43/58 [00:02<00:00, 15.89it/s]Capturing num tokens (num_tokens=144 avail_mem=55.55 GB):  74%|███████▍  | 43/58 [00:02<00:00, 15.89it/s]Capturing num tokens (num_tokens=128 avail_mem=55.55 GB):  74%|███████▍  | 43/58 [00:02<00:00, 15.89it/s]Capturing num tokens (num_tokens=112 avail_mem=55.55 GB):  74%|███████▍  | 43/58 [00:02<00:00, 15.89it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.55 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.18it/s]Capturing num tokens (num_tokens=96 avail_mem=55.54 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.18it/s] Capturing num tokens (num_tokens=80 avail_mem=55.54 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.18it/s]Capturing num tokens (num_tokens=64 avail_mem=55.53 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.18it/s]Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.18it/s]Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.73it/s]Capturing num tokens (num_tokens=32 avail_mem=55.53 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.73it/s]Capturing num tokens (num_tokens=28 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.73it/s]Capturing num tokens (num_tokens=24 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.73it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.73it/s]Capturing num tokens (num_tokens=20 avail_mem=55.52 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.56it/s]Capturing num tokens (num_tokens=16 avail_mem=55.52 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.56it/s]Capturing num tokens (num_tokens=12 avail_mem=55.51 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.56it/s]Capturing num tokens (num_tokens=8 avail_mem=55.51 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.56it/s] Capturing num tokens (num_tokens=4 avail_mem=55.50 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.56it/s]Capturing num tokens (num_tokens=4 avail_mem=55.50 GB): 100%|██████████| 58/58 [00:03<00:00, 26.94it/s]Capturing num tokens (num_tokens=4 avail_mem=55.50 GB): 100%|██████████| 58/58 [00:03<00:00, 17.08it/s]


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
    Generated text:  Richard and I am a graduate student in the School of Mathematical Sciences at the University of Liverpool.
    My research interests are in algorithmic information theory and its applications, which can be applied in many different areas. The field of algorithmic information theory seeks to understand how to extract useful information from the input data of a computation. It aims to determine whether certain computations yield useful answers and what information can be extracted from the data. For example, it could be used to determine the complexity of a computation, to distinguish between different computational models, or to determine the computational power of an algorithm. It also has many applications in areas like machine learning, bio
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy on earth. He gets up early in the morning and has breakfast with his family and his staff. He spends most of the day driving around the country to make important decisions and attending to important things in the world. On the weekends, he travels to different parts of the world to see his friends and do important things. He also enjoys spending time with his family. But he has a short vacation every four years. On this vacation, he is able to travel around the country, see his friends and family, and relax in Hawaii for the weekend. The vacation lasts only two weeks and he spends most of the time in Hawaii.
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Tokyo
    D. New York
    Answer:
    
    A
    
    Which of the following is the correct process for a company to implement customer service?
    A. Customer service management → Customer service policy → Customer service training → Customer service execution
    B. Customer service management → Customer service policy → Customer service implementation → Customer service execution
    C. Customer service management → Customer service policy → Customer service execution → Customer service training
    D. Customer service management → Customer service training → Customer service policy → Customer service implementation
    Answer:
    
    B
    
    Which of the following statements about the structure of an AS (Automatic
    ===============================
    Prompt: The future of AI is
    Generated text:  still very much uncertain, but it is clear that the future of work will be fundamentally different from the past. An AI-powered assistant to a company’s HR department, for instance, would allow the company to make critical decisions and react to challenges quickly and effectively, ultimately improving efficiency and reducing costs. The assistant would also help human HR staff with tasks such as monitoring employee performance, scheduling appointments, and making recommendations on benefits and perks. And the assistant would be able to communicate with employees on a level that is familiar to them, making them feel valued and motivated.
    
    One area where AI can have the biggest impact is in the field of education.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, interesting fact about yourself]. I'm always looking for new challenges and opportunities to grow and learn. What are your hobbies or interests? I enjoy [insert a hobby or interest]. I'm always looking for new experiences and adventures to try. What's your favorite book or movie? I love [insert a favorite book or movie]. I'm always looking for new ways to expand my knowledge and learn more about the world. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country. It is located on the Seine River and is the seat of government, administration, and culture for the country. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. The city is a major economic center and plays a significant role in France's economy.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be a greater emphasis on developing AI that is designed to be ethical and responsible. This will likely involve developing AI that is transparent, accountable, and accountable to human values.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including diagnosis,
    


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
    Generated text:  [Your Name]. I'm a [character type] who enjoys [insert hobby or interest]. If you have any questions, don't hesitate to ask me anything. I'm friendly, helpful, and always ready to assist you. Welcome to my world! [Tell a brief introduction of yourself or something about yourself that will set the tone of your character's personality. For example, if you're a detective, mention your knowledge of detective tactics, if you're a writer, mention your writing style, if you're a musician, mention your musical style, etc. Your introduction should be short and to the point, capturing the essence of your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement is: Paris is the capital city of France. 
    
    To elaborate, Paris is the largest and most populous city in France, located on the River Seine in the northern region of the country. It is a UNESCO World Heritage site and the seat of government, administration, and culture for France. The city is known for its historical landmarks, museums, and fine dining, and it is one of the world's most visited cities. It also hosts numerous cultural events and is an important center for French politics, economics, and popular culture. 
    
    The statement is accurate as of 2023, and it is considered
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve more integration with human-machine interactions, leading to a more dynamic and collaborative relationship between AI and humans. This will include a greater use of AI in decision-making, automation, and decision support. AI will also be integrated into various industries and sectors, such as healthcare, finance, and transportation, to improve efficiency and effectiveness. AI will be used to develop new technologies and services, such as virtual assistants, autonomous vehicles, and chatbots. AI will also be used to address some of the most pressing challenges of our time, such as climate change, pandemics, and global poverty. Finally, AI will continue to evolve and improve


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

     __

    ________

    _

     and

     I

     am

     a

    /an

     ______

    __

    _.

     I

     am

     currently

     living

     in

     the

     city

     of

     ______

    ,

     and

     I

     have

     always

     been

     interested

     in

     art

    .

     I

     enjoy

     exploring

     new

     places

     and

     trying

     out

     new

     things

    .

     I

     like

     to

     be

     outdoors

    ,

     and

     I

     love

     spending

     time

     with

     my

     friends

     and

     family

    .

     I

     am

     always

     looking

     for

     new

     challenges

    ,

     and

     I

     am

     excited

     to

     see

     where

     my

     adventures

     will

     take

     me

    .

     Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    .

     Let

     me

     know

     if

     you

     would

     like

     me

     to

     do

     any

     more

     work

     for

     you

    .

     Hello

    ,

     my

     name

     is

     __

    ________

    _

     and

     I

     am

     a

    /an

     __

    ________

    _

    _.

     I

     am

     currently

     living

     in

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Paris

     is

     a

     city

     with

     a

     rich

     history

     and

     a

     diverse

     culture

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     famous

     for

     its

     French

     language

    ,

     cuisine

    ,

     and

     fashion

    .

     The

     population

     of

     Paris

     is

     approximately

     

    2

    .

    1

     million

     people

    ,

     making

     it

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    .

     
    


    Paris

     is

     a

     vibrant

     and

     multicultural

     city

     that

     is

     home

     to

     a

     range

     of

     people

     from

     all

     over

     the

     world

    .

     It

     is

     also

     a

     major

     cultural

     and

     economic

     center

     of

     France

    ,

     with

     many

     international

     institutions

     and

     businesses

    .

     The

     city

     is

     also

     known

     for

     its

     scientific

     and

     cultural

     institutions

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     different

     trends

     and

     developments

     that

     could

     shape

     the

     technology

     and

     impact

     society

    .

     Here

     are

     a

     few

     potential

     future

     trends

    :
    


    1

    .

     AI

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    :

     As

     AI

     continues

     to

     evolve

    ,

     it

     is

     likely

     that

     we

     will

     see

     more

     and

     more

     integration

     into

     our

     daily

     lives

    .

     This

     could

     include

     things

     like

     voice

     assistants

    ,

     self

    -driving

     cars

    ,

     and

     personalized

     health

     and

     wellness

     solutions

    .
    


    2

    .

     AI

     will

     become

     more

     accessible

     and

     affordable

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

     that

     we

     will

     see

     a

     reduction

     in

     the

     cost

     and

     complexity

     of

     implementing

     AI

     solutions

    .

     This

     could

     make

     it

     easier

     for

     businesses

     and

     individuals

     to

     incorporate

     AI

    



```python
llm.shutdown()
```
