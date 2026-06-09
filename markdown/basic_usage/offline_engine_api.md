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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:12,  3.88it/s]

    Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:04,  9.77it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:04,  9.77it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:04,  9.77it/s]

    Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:04,  9.77it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:04,  9.77it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 19.32it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 19.32it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 19.32it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 19.32it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 19.32it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 19.32it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 23.78it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 23.78it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 23.78it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 23.78it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 23.78it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 23.78it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 27.66it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 27.66it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 27.66it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 27.66it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 27.66it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 27.66it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 27.66it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 27.66it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 36.02it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 36.02it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 37.56it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 37.56it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 37.56it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 37.56it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 37.56it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 37.56it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 37.56it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 42.80it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 42.80it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 42.80it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 42.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.62 GB):   2%|▏         | 1/58 [00:00<00:07,  7.29it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.25 GB):   2%|▏         | 1/58 [00:00<00:07,  7.29it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=59.25 GB):   3%|▎         | 2/58 [00:00<00:06,  8.67it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.65 GB):   3%|▎         | 2/58 [00:00<00:06,  8.67it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.65 GB):   5%|▌         | 3/58 [00:00<00:06,  8.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.25 GB):   5%|▌         | 3/58 [00:00<00:06,  8.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.72 GB):   5%|▌         | 3/58 [00:00<00:06,  8.31it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.72 GB):   9%|▊         | 5/58 [00:00<00:05,  9.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.74 GB):   9%|▊         | 5/58 [00:00<00:05,  9.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.73 GB):   9%|▊         | 5/58 [00:00<00:05,  9.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.73 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.73 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.34it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=59.22 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.22 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.75 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.21 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.32it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.21 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.78 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.20 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.20 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.80 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.19 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.90it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=59.19 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.83 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.19 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.19 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.18 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.96 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.77it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.96 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.91it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.18 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.89 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.91it/s]Capturing num tokens (num_tokens=960 avail_mem=59.17 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.91it/s] Capturing num tokens (num_tokens=960 avail_mem=59.17 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.22it/s]Capturing num tokens (num_tokens=896 avail_mem=59.16 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.22it/s]Capturing num tokens (num_tokens=832 avail_mem=58.95 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.22it/s]

    Capturing num tokens (num_tokens=768 avail_mem=59.15 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.22it/s]Capturing num tokens (num_tokens=768 avail_mem=59.15 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.99it/s]Capturing num tokens (num_tokens=704 avail_mem=59.14 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.99it/s]Capturing num tokens (num_tokens=640 avail_mem=59.00 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.99it/s]Capturing num tokens (num_tokens=576 avail_mem=59.01 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.99it/s]Capturing num tokens (num_tokens=576 avail_mem=59.01 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=512 avail_mem=59.11 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.12it/s]

    Capturing num tokens (num_tokens=480 avail_mem=59.12 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=448 avail_mem=59.12 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=448 avail_mem=59.12 GB):  53%|█████▎    | 31/58 [00:02<00:01, 22.09it/s]Capturing num tokens (num_tokens=416 avail_mem=59.11 GB):  53%|█████▎    | 31/58 [00:02<00:01, 22.09it/s]Capturing num tokens (num_tokens=384 avail_mem=59.01 GB):  53%|█████▎    | 31/58 [00:02<00:01, 22.09it/s]Capturing num tokens (num_tokens=352 avail_mem=59.03 GB):  53%|█████▎    | 31/58 [00:02<00:01, 22.09it/s]Capturing num tokens (num_tokens=352 avail_mem=59.03 GB):  59%|█████▊    | 34/58 [00:02<00:01, 23.63it/s]Capturing num tokens (num_tokens=320 avail_mem=59.03 GB):  59%|█████▊    | 34/58 [00:02<00:01, 23.63it/s]

    Capturing num tokens (num_tokens=288 avail_mem=59.03 GB):  59%|█████▊    | 34/58 [00:02<00:01, 23.63it/s]Capturing num tokens (num_tokens=256 avail_mem=59.03 GB):  59%|█████▊    | 34/58 [00:02<00:01, 23.63it/s]Capturing num tokens (num_tokens=256 avail_mem=59.03 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.25it/s]Capturing num tokens (num_tokens=240 avail_mem=59.02 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.25it/s]Capturing num tokens (num_tokens=224 avail_mem=59.07 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.25it/s]Capturing num tokens (num_tokens=208 avail_mem=59.06 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.25it/s]Capturing num tokens (num_tokens=192 avail_mem=59.05 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.25it/s]Capturing num tokens (num_tokens=192 avail_mem=59.05 GB):  71%|███████   | 41/58 [00:02<00:00, 26.97it/s]Capturing num tokens (num_tokens=176 avail_mem=59.05 GB):  71%|███████   | 41/58 [00:02<00:00, 26.97it/s]

    Capturing num tokens (num_tokens=160 avail_mem=59.05 GB):  71%|███████   | 41/58 [00:02<00:00, 26.97it/s]Capturing num tokens (num_tokens=144 avail_mem=59.04 GB):  71%|███████   | 41/58 [00:02<00:00, 26.97it/s]Capturing num tokens (num_tokens=128 avail_mem=59.04 GB):  71%|███████   | 41/58 [00:02<00:00, 26.97it/s]Capturing num tokens (num_tokens=128 avail_mem=59.04 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.67it/s]Capturing num tokens (num_tokens=112 avail_mem=59.03 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.67it/s]Capturing num tokens (num_tokens=96 avail_mem=59.02 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.67it/s] Capturing num tokens (num_tokens=80 avail_mem=59.01 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.67it/s]Capturing num tokens (num_tokens=64 avail_mem=59.01 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.67it/s]

    Capturing num tokens (num_tokens=64 avail_mem=59.01 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.95it/s]Capturing num tokens (num_tokens=48 avail_mem=59.00 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.95it/s]Capturing num tokens (num_tokens=32 avail_mem=59.00 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.95it/s]Capturing num tokens (num_tokens=28 avail_mem=58.99 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.95it/s]Capturing num tokens (num_tokens=24 avail_mem=58.99 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.95it/s]Capturing num tokens (num_tokens=24 avail_mem=58.99 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.10it/s]Capturing num tokens (num_tokens=20 avail_mem=58.98 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.10it/s]Capturing num tokens (num_tokens=16 avail_mem=58.95 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.10it/s]Capturing num tokens (num_tokens=12 avail_mem=58.94 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.10it/s]Capturing num tokens (num_tokens=8 avail_mem=58.96 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.10it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=58.96 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.59it/s]Capturing num tokens (num_tokens=4 avail_mem=58.93 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.59it/s]Capturing num tokens (num_tokens=4 avail_mem=58.93 GB): 100%|██████████| 58/58 [00:02<00:00, 20.29it/s]


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
    Generated text:  Miki. I am 12 years old. I have a friend named Kathy. We're both kind and helpful people. We're both good at drawing pictures. We're in the same class. We're also in the same grade. We like to play games together. We play hide and seek. When we play hide and seek, we give each other names. We never use names that are the same, and we don't tell anyone our names. We're in the same class, but we're not friends. It's just that we're always playing together. I'm not in the class, but I'm always playing
    ===============================
    Prompt: The president of the United States is
    Generated text:  considering the following potential national programs:  
    1. The creation of a new federal tax on corporations.
    2. Expanding the federal health insurance program.
    3. Increasing federal funding for international aid.
    4. Building an international military base.
    
    How many of the above programs would be considered anti-market economic activities?
    
    a) 0  
    b) 1  
    c) 2  
    d) 3  
    e) 4
    
    To determine how many of the proposed programs would be considered anti-market economic activities, we need to analyze each program and determine if it promotes market-based economic activities or if it is an anti-market activity.
    
    1.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. ____
    A. Correct
    B. Incorrect
    Answer:
    A
    
    The contents of China's constitution include:
    A. Basic Rights of Citizens
    B. Fundamental Principles of Law
    C. Basic Duties of Citizens
    D. Fundamental Rights of Citizens
    Answer:
    A
    
    The main subject of Marxism is ____
    A. Capitalist Society
    B. Feudal Society
    C. Communist Society
    D. Socialist Society
    Answer:
    D
    
    The main contradiction in contemporary China is
    A. The contradiction between the people's growing needs for a better life and unbalanced and inadequate development.
    B. The contradiction between the people
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain, but the adoption of artificial intelligence is expected to expand across all industries. This post explores how AI is being used in various industries, including healthcare, finance, and education, as well as the benefits and challenges of AI adoption.
    AI has already transformed various industries, including healthcare and finance. In healthcare, AI is being used to improve patient outcomes through predictive analytics, drug discovery, and patient care coordination. In finance, AI is used for fraud detection, trading, and risk management.
    AI adoption in education has also increased, with some schools and institutions using AI to personalize learning and improve student outcomes. The benefits of AI adoption in


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill or Trait] who has always been [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Museum of Modern Art. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest and most prestigious fashion events in the world. The city is also home to the French Parliament, the French National Library, and the French National Museum of Modern Art. Paris is a vibrant and diverse city with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and personalized interactions. This could lead to more efficient and effective use of AI in various fields, such as healthcare, finance, and education.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines to ensure that AI is used in a responsible and ethical manner.
    
    3. Greater reliance on AI for decision-making: AI is likely to become more integrated into decision-making processes
    


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
    Generated text:  [First Name] and I am a [Position/Role] at [Company Name]. I am a [Short Biography/Summary of Experience] and I specialize in [Your Area of Expertise] and enjoy [Your Goal/What You Want to Do Next]. My passion is to [Your Passion], and I am always looking to learn new things and improve myself. I am [Your Height/Weight/Height/Weight] and [Your Gender/Any Additional Attributes]. How can I be a great fit for your team? If you would like to chat and discuss how I can contribute to your team, please feel free to reach out
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To generate this answer, I consulted various sources to gather information about the location of Paris, including its official name, official capital, and political importance. I also considered any notable landmarks or tourist attractions in Paris to provide context for the statement. This process resulted in a concise and factual statement about the capital city of France. 
    
    Statement: Paris, officially known as "Le Capitale de l'Industrie" (The Capital of Industry), is the capital city of France. It serves as the political, economic, and cultural center of the nation. The city is home to many iconic landmarks, including the Eiff
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities, but also comes with challenges and uncertainties. Here are some possible future trends in AI:
    
    1. **Advancements in Computer Vision**: AI-driven vision systems will become more capable of understanding and interpreting visual data, leading to new applications in fields such as autonomous vehicles, security, and medical imaging.
    
    2. **Machine Learning and Deep Learning**: These technologies are likely to continue to improve, leading to more efficient and accurate predictive models. Deep learning, specifically, will become even more powerful, enabling faster and more accurate learning of complex patterns in data.
    
    3. **AI for Healthcare**: AI will play an increasingly important role in


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

    ].

     I

    'm

     a

     [

    job

     title

     or

     hobby

    ]

     enthusiast

    ,

     particularly

     skilled

     in

     [

    specific

     interest

     or

     hobby

    ],

     and

     I

     enjoy

     sharing

     my

     knowledge

     and

     experiences

     with

     others

    .

     I

     believe

     in

     [

    phil

    osoph

    y

     or

     belief

     system

    ]

     and

     am

     always

     eager

     to

     learn

     from

     new

     perspectives

     and

     experiences

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Your

     Name

    ]:

     Hi

     there

    !

     I

    'm

     [

    insert

     name

    ]

     from

     [

    insert

     location

    ]

     with

     [

    insert

     job

     title

     or

     hobby

    ].

     I

    'm

     passionate

     about

     [

    insert

     specific

     interest

     or

     hobby

    ].

     And

     yes

    ,

     I

     am

     a

     natural

     teacher

    !

     I

     love

     sharing

     my

     knowledge

     and

     expertise

     with

     others

    ,

     and

     I

     believe

     in

     [

    phil

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     the

     country

    .

     It

     is

     home

     to

     over

     

    1

    0

     million

     people

    ,

     making

     it

     the

     world

    's

     

    1

    5

    th

     most

     populous

     city

     by

     population

    .

     The

     city

     is

     known

     for

     its

     stunning

     architecture

    ,

     diverse

     cultural

     scene

    ,

     and

     rich

     history

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     is

     a

     major

     economic

     and

     cultural

     center

    ,

     known

     for

     its

     fashion

    ,

     art

    ,

     and

     food

     industries

    .

     The

     city

     is

     also

     known

     for

     its

     vibrant

     nightlife

     and

     diverse

     cultural

     scene

    .

     It

     is

     an

     important

     center

     for

     international

     trade

    ,

     diplomacy

    ,

     and

     education

    ,

     and

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

    .

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

    ,

     and

     it

    's

     likely

     to

     continue

     evolving

     rapidly

    .

     Here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     emerge

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     autonomy

    :

     With

     the

     development

     of

     deep

     learning

     and

     neural

     networks

    ,

     there

     is

     a

     growing

     possibility

     for

     machines

     to

     become

     more

     capable

     of

     making

     decisions

     and

     taking

     action

     on

     their

     own

    .

     This

     could

     lead

     to

     autonomous

     vehicles

    ,

     smart

     homes

    ,

     and

     other

     applications

     that

     rely

     on

     autonomous

     decision

    -making

    .
    


    2

    .

     Enhanced

     creativity

    :

     AI

     is

     becoming

     more

     adept

     at

     generating

     new

     and

     creative

     ideas

    .

     This

     could

     lead

     to

     new

     areas

     of

     research

     and

     development

    ,

     such

     as

     AI

     for

     creativity

     and

     AI

     for

     problem

    -solving

    .
    


    3

    .

     Improved

     ethical

     considerations

    :

     As

    



```python
llm.shutdown()
```
