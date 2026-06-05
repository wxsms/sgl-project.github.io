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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.63it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.98it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.98it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.98it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.98it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.98it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.75it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.51it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.51it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.51it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.51it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.51it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.51it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 19.73it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.36it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.36it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.36it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.36it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.36it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 23.36it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 23.36it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 29.71it/s]

    Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 34.09it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 38.45it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 38.45it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 38.45it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 38.45it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 38.45it/s]

    Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 38.45it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 38.45it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 42.78it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 42.78it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 42.78it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 42.78it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 42.78it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 42.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=48.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=48.79 GB):   2%|▏         | 1/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=48.76 GB):   2%|▏         | 1/58 [00:00<00:07,  7.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=48.76 GB):   3%|▎         | 2/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=48.75 GB):   3%|▎         | 2/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=48.75 GB):   5%|▌         | 3/58 [00:00<00:07,  7.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=48.75 GB):   5%|▌         | 3/58 [00:00<00:07,  7.42it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=48.75 GB):   7%|▋         | 4/58 [00:00<00:07,  7.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=48.75 GB):   7%|▋         | 4/58 [00:00<00:07,  7.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=48.75 GB):   9%|▊         | 5/58 [00:00<00:06,  7.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=48.75 GB):   9%|▊         | 5/58 [00:00<00:06,  7.91it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=48.75 GB):  10%|█         | 6/58 [00:00<00:06,  8.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=48.74 GB):  10%|█         | 6/58 [00:00<00:06,  8.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=48.74 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=48.73 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.51it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=48.73 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=48.73 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=48.73 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=48.73 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=48.72 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.43it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=48.72 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=48.72 GB):  21%|██        | 12/58 [00:01<00:04, 10.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=48.72 GB):  21%|██        | 12/58 [00:01<00:04, 10.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=48.72 GB):  21%|██        | 12/58 [00:01<00:04, 10.03it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=48.72 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=48.71 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=48.61 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=48.61 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=48.43 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.30it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=48.23 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=48.23 GB):  31%|███       | 18/58 [00:01<00:03, 11.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=48.04 GB):  31%|███       | 18/58 [00:01<00:03, 11.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=47.86 GB):  31%|███       | 18/58 [00:01<00:03, 11.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=47.86 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=47.70 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.81it/s]

    Capturing num tokens (num_tokens=960 avail_mem=47.58 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.81it/s] Capturing num tokens (num_tokens=896 avail_mem=47.48 GB):  34%|███▍      | 20/58 [00:02<00:02, 12.81it/s]Capturing num tokens (num_tokens=832 avail_mem=47.37 GB):  34%|███▍      | 20/58 [00:02<00:02, 12.81it/s]Capturing num tokens (num_tokens=832 avail_mem=47.37 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.61it/s]Capturing num tokens (num_tokens=768 avail_mem=47.35 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.61it/s]Capturing num tokens (num_tokens=704 avail_mem=47.35 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.61it/s]Capturing num tokens (num_tokens=640 avail_mem=47.35 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.61it/s]Capturing num tokens (num_tokens=576 avail_mem=47.35 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.61it/s]Capturing num tokens (num_tokens=512 avail_mem=47.33 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.61it/s]Capturing num tokens (num_tokens=512 avail_mem=47.33 GB):  50%|█████     | 29/58 [00:02<00:01, 25.88it/s]Capturing num tokens (num_tokens=480 avail_mem=47.35 GB):  50%|█████     | 29/58 [00:02<00:01, 25.88it/s]

    Capturing num tokens (num_tokens=448 avail_mem=47.34 GB):  50%|█████     | 29/58 [00:02<00:01, 25.88it/s]Capturing num tokens (num_tokens=416 avail_mem=65.68 GB):  50%|█████     | 29/58 [00:02<00:01, 25.88it/s]Capturing num tokens (num_tokens=416 avail_mem=65.68 GB):  55%|█████▌    | 32/58 [00:02<00:01, 23.52it/s]Capturing num tokens (num_tokens=384 avail_mem=65.68 GB):  55%|█████▌    | 32/58 [00:02<00:01, 23.52it/s]Capturing num tokens (num_tokens=352 avail_mem=65.67 GB):  55%|█████▌    | 32/58 [00:02<00:01, 23.52it/s]Capturing num tokens (num_tokens=320 avail_mem=65.57 GB):  55%|█████▌    | 32/58 [00:02<00:01, 23.52it/s]Capturing num tokens (num_tokens=288 avail_mem=65.57 GB):  55%|█████▌    | 32/58 [00:02<00:01, 23.52it/s]

    Capturing num tokens (num_tokens=256 avail_mem=65.57 GB):  55%|█████▌    | 32/58 [00:02<00:01, 23.52it/s]Capturing num tokens (num_tokens=256 avail_mem=65.57 GB):  64%|██████▍   | 37/58 [00:02<00:00, 29.71it/s]Capturing num tokens (num_tokens=240 avail_mem=65.54 GB):  64%|██████▍   | 37/58 [00:02<00:00, 29.71it/s]Capturing num tokens (num_tokens=224 avail_mem=65.52 GB):  64%|██████▍   | 37/58 [00:02<00:00, 29.71it/s]Capturing num tokens (num_tokens=208 avail_mem=65.52 GB):  64%|██████▍   | 37/58 [00:02<00:00, 29.71it/s]Capturing num tokens (num_tokens=192 avail_mem=65.51 GB):  64%|██████▍   | 37/58 [00:02<00:00, 29.71it/s]Capturing num tokens (num_tokens=176 avail_mem=65.37 GB):  64%|██████▍   | 37/58 [00:02<00:00, 29.71it/s]Capturing num tokens (num_tokens=176 avail_mem=65.37 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.37it/s]Capturing num tokens (num_tokens=160 avail_mem=64.98 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.37it/s]Capturing num tokens (num_tokens=144 avail_mem=64.98 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.37it/s]Capturing num tokens (num_tokens=128 avail_mem=64.07 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.37it/s]Capturing num tokens (num_tokens=112 avail_mem=63.12 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.37it/s]

    Capturing num tokens (num_tokens=96 avail_mem=63.11 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.37it/s] Capturing num tokens (num_tokens=96 avail_mem=63.11 GB):  81%|████████  | 47/58 [00:02<00:00, 37.17it/s]Capturing num tokens (num_tokens=80 avail_mem=63.11 GB):  81%|████████  | 47/58 [00:02<00:00, 37.17it/s]Capturing num tokens (num_tokens=64 avail_mem=63.11 GB):  81%|████████  | 47/58 [00:02<00:00, 37.17it/s]Capturing num tokens (num_tokens=48 avail_mem=63.10 GB):  81%|████████  | 47/58 [00:02<00:00, 37.17it/s]Capturing num tokens (num_tokens=32 avail_mem=63.10 GB):  81%|████████  | 47/58 [00:02<00:00, 37.17it/s]Capturing num tokens (num_tokens=28 avail_mem=63.10 GB):  81%|████████  | 47/58 [00:02<00:00, 37.17it/s]Capturing num tokens (num_tokens=28 avail_mem=63.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 40.16it/s]Capturing num tokens (num_tokens=24 avail_mem=63.09 GB):  90%|████████▉ | 52/58 [00:02<00:00, 40.16it/s]Capturing num tokens (num_tokens=20 avail_mem=63.09 GB):  90%|████████▉ | 52/58 [00:02<00:00, 40.16it/s]Capturing num tokens (num_tokens=16 avail_mem=63.09 GB):  90%|████████▉ | 52/58 [00:02<00:00, 40.16it/s]Capturing num tokens (num_tokens=12 avail_mem=63.08 GB):  90%|████████▉ | 52/58 [00:02<00:00, 40.16it/s]

    Capturing num tokens (num_tokens=8 avail_mem=63.08 GB):  90%|████████▉ | 52/58 [00:02<00:00, 40.16it/s] Capturing num tokens (num_tokens=8 avail_mem=63.08 GB):  98%|█████████▊| 57/58 [00:02<00:00, 42.58it/s]Capturing num tokens (num_tokens=4 avail_mem=63.08 GB):  98%|█████████▊| 57/58 [00:02<00:00, 42.58it/s]Capturing num tokens (num_tokens=4 avail_mem=63.08 GB): 100%|██████████| 58/58 [00:02<00:00, 20.17it/s]


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
    Generated text:  Lisa. I am a 16-year-old high school student. I want to make more friends. I'm very shy. But I'm willing to make new friends. I want to make friends with friends who are also interested in the same things as me. I hope to make friends with kids who like to listen to music, eat with me and do sports. I am shy and I am a bit silly. But I really hope to make new friends. I have lots of friends. I can go to a lot of fun parties. I can also go to fun events. But sometimes, I feel very shy and nervous when I
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to introduce a bill that requires all citizens to use a particular style of writing for all official communications. It will not be easy, because many different styles are actually used in the United States. Each state has its own writing style, so there will be a lot of confusion. The president is also concerned that a few wealthy and influential citizens will dominate the debate about the style of writing. To help avoid this problem, the president will require that the bill be approved by more than two-thirds of the states in which the bill would have effect. The president is not optimistic about the results of the debate. He says that if only one state
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. London C. Rome D. Moscow
    
    Paris
    
    What is the primary reason for the transition from a feudal system to a constitutional monarchy in France?
    
    A. The need for trade routes to flourish
    B. The need for trade routes to flourish and grow
    C. The need for trade routes to flourish and grow and the desire to replace the absolute monarchy of Louis XIV with a constitutional monarchy
    D. The need to replace the absolute monarchy of Louis XIV with a constitutional monarchy
    
    C. The need for trade routes to flourish and grow and the desire to replace the absolute monarchy of Louis XIV with a constitutional monarchy
    ===============================
    Prompt: The future of AI is
    Generated text:  about more than creating advanced machine learning models that will solve complex problems, as it is now being increasingly used to perform a variety of tasks for which there are no direct equivalents. As the data available to the AI is growing, and the volume of data being generated increases, the AI industry needs to be able to handle this new data flow. To handle this new data flow, we have to use machine learning techniques for data stream processing and deep learning for prediction. In addition, a new model for AI for data analysis, or the de facto standard for AI is the AI pipeline. This is a very common and well-known term that refers to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is the largest city in Europe by population and is a major center for business, finance, and education. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: AI is expected to continue to automate a wide range of tasks, from manufacturing and transportation to customer service and healthcare. This will lead to increased efficiency and productivity, as machines can perform tasks that were previously done by humans.
    
    2. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making, allowing machines to make more informed and nuanced decisions. This will lead to more accurate and reliable AI systems, as well as better decision-making in various fields.
    
    3. Greater reliance on AI for decision-making: AI is likely to become more
    


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
    Generated text:  [Your Name]. I'm a [Your Profession/Title] with a passion for [Your Speciality/Interest/Claim]. I believe that [Your Speciality/Interest/Claim] is what makes you the best person to work with, and that's why I'm excited to learn more about our team and how I can contribute to your success. Are you interested in learning more about [Your Profession/Title] and your speciality/interest/claim? I'd love to chat about it further. [Your Name] [Your Contact Information] [Your Email Address]
    Always remember to keep your introduction friendly and informative. The more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France. It is located in the northwestern region of France and is the largest city in the country. The city is known for its rich history, iconic landmarks, and diverse cultural scene, including its famous Eiffel Tower, Notre-Dame Cathedral, and numerous museums. Paris is also famous for its fashion, art, and cuisine. Its nickname is "The City of Light," and it has a population of over 2 million people. It is one of the world's most popular tourist destinations. Paris is also considered one of the oldest cities in the world, having been inhabited since the 
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some potential trends include:
    
      1. Increasing automation and productivity: AI has the potential to automate a wide range of tasks and increase productivity, making work more efficient and enjoyable for many people.
      2. Integration with other technologies: AI is becoming more integrated with other technologies, such as machine learning and blockchain, creating new possibilities for new applications and potential risks.
      3. Improved privacy and security: As AI systems become more advanced, there is a risk of increasing levels of data collection and processing, which could lead to increased surveillance and privacy risks.
      4. Increased focus on ethical and social considerations


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

     am

     [

    Age

    ].

     I

     live

     in

     [

    City

    /

    State

    ].

     I

     love

     [

    interest

     or

     hobby

    ],

     and

     I

     love

     [

    reason

     why

     I

     love

     it

    ].

     I

     enjoy

     spending

     [

    amount

     of

     time

     per

     day

    /

    week

    /month

    ]

     with

     [

    person

    /g

    roup

     of

     people

    ],

     and

     I

     am

     always

     looking

     for

     ways

     to

     [

    some

     positive

     change

     or

     impact

     on

     the

     world

    ].

     How

     can

     I

     say

     "

    hello

    "

     to

     you

    ?

     [

    Speak

     with

     enthusiasm

     and

     excitement

    ]

     "

    Hello

    ,

     my

     name

     is

     [

    Name

    ]

     and

     I

     am

     [

    Age

    ].

     I

     live

     in

     [

    City

    /

    State

    ].

     I

     love

     [

    interest

     or

     hobby

    ],

     and

     I

     love

     [

    reason

     why

     I

     love

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     beautiful

     city

     with

     a

     rich

     history

    ,

     many

     museums

    ,

     and

     a

     lively

     nightlife

     scene

    .

     The

     city

     is

     known

     for

     its

     romantic

     architecture

    ,

     vibrant

     food

     scene

    ,

     and

     annual

     festivities

     like

     the

     Les

     B

    ains

     des

     Ang

    l

    ais

     and

     the

     St

    .

     Louis

     World

     Fair

    .

     It

    's

     a

     fascinating

     city

     with

     a

     unique

     blend

     of

     French

     and

     European

     cultures

    .

     
    


    Note

     that

     this

     statement

     is

     factual

     and

     does

     not

     contain

     any

     political

    ,

     religious

    ,

     or

     sensitive

     content

    .

     It

     should

     be

     accurate

     for

     a

     general

     understanding

     of

     Paris

    .
    


    Paris

    ,

     the

     beloved

     capital

     of

     France

    ,

     is

     renowned

     for

     its

     exquisite

     architecture

    ,

     vibrant

     arts

     scene

    ,

     and

     annual

     festivals

    .

     Its

     romantic

     history

     and

     cultural

     blend

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     reliance

     on

     machine

     learning

    :

     With

     advancements

     in

     machine

     learning

    ,

     AI

     will

     become

     more

     efficient

     and

     accurate

     at

     recognizing

     patterns

     and

     making

     decisions

    .
    


    2

    .

     Greater

     integration

     of

     AI

     into

     existing

     systems

    :

     AI

     will

     become

     more

     integrated

     into

     existing

     systems

    ,

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     retail

    ,

     to

     increase

     efficiency

     and

     reduce

     errors

    .
    


    3

    .

     Development

     of

     ethical

     AI

    :

     AI

     will

     be

     used

     to

     develop

     ethical

     guidelines

     and

     regulations

    ,

     ensuring

     that

     AI

     systems

     are

     designed

     and

     deployed

     in

     a

     responsible

     manner

    .
    


    4

    .

     Greater

     focus

     on

     low

    -cost

     AI

    :

     As

     AI

     becomes

     more

     affordable

    ,

     there

     will

     be

     a

     greater

     focus

     on

     developing

    



```python
llm.shutdown()
```
