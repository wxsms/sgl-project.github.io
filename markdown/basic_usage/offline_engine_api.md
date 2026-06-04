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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.54it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.54it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:20,  2.54it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:20,  2.54it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.45it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.45it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.78it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.78it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.78it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.78it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.78it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.47it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.47it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.47it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.47it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.47it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.32it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.32it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.32it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.32it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.32it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.32it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 19.51it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 19.51it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 19.51it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 19.51it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 19.51it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.24it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.24it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.24it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.24it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.24it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 23.24it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 23.24it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 38.59it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 38.59it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 38.59it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 40.34it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 40.34it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 40.34it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 40.34it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:06<00:00, 40.34it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:06<00:00, 40.34it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:06<00:00, 40.34it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 43.98it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 43.98it/s]

    Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 43.98it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 43.98it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 43.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.97 GB):   2%|▏         | 1/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.93 GB):   2%|▏         | 1/58 [00:00<00:07,  7.37it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.93 GB):   3%|▎         | 2/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.93 GB):   3%|▎         | 2/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.93 GB):   5%|▌         | 3/58 [00:00<00:07,  7.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.93 GB):   5%|▌         | 3/58 [00:00<00:07,  7.43it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.93 GB):   7%|▋         | 4/58 [00:00<00:07,  7.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.93 GB):   7%|▋         | 4/58 [00:00<00:07,  7.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.93 GB):   9%|▊         | 5/58 [00:00<00:06,  7.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.92 GB):   9%|▊         | 5/58 [00:00<00:06,  7.80it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.92 GB):  10%|█         | 6/58 [00:00<00:06,  8.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.91 GB):  10%|█         | 6/58 [00:00<00:06,  8.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.91 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.91 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.50it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.91 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.91 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.91 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.91 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.90 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=53.90 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.90 GB):  21%|██        | 12/58 [00:01<00:04,  9.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.89 GB):  21%|██        | 12/58 [00:01<00:04,  9.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.89 GB):  21%|██        | 12/58 [00:01<00:04,  9.93it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.89 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.63it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.89 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.63it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.89 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.63it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.89 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.88 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.62it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=53.88 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.88 GB):  31%|███       | 18/58 [00:01<00:03, 12.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.88 GB):  31%|███       | 18/58 [00:01<00:03, 12.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.87 GB):  31%|███       | 18/58 [00:01<00:03, 12.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.87 GB):  34%|███▍      | 20/58 [00:01<00:02, 13.25it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.85 GB):  34%|███▍      | 20/58 [00:01<00:02, 13.25it/s]

    Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.25it/s] Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.01it/s]Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.01it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.01it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.01it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.68it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.68it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.68it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.01it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.01it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.01it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  50%|█████     | 29/58 [00:02<00:01, 16.18it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 16.18it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 16.18it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 16.59it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 16.59it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 16.59it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.78it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.78it/s]

    Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.78it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  60%|██████    | 35/58 [00:02<00:01, 16.97it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 16.97it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 16.97it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.57it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.57it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.57it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.30it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.30it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.30it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:00, 17.36it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:00, 17.36it/s]

    Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  71%|███████   | 41/58 [00:03<00:00, 17.36it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 17.18it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 17.18it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 17.18it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:03<00:00, 16.75it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:03<00:00, 16.75it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 16.75it/s] Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 16.44it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 16.44it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:03<00:00, 16.44it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 16.23it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 16.23it/s]

    Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 16.23it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:03<00:00, 16.11it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 16.11it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 16.11it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.11it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.11it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.11it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  95%|█████████▍| 55/58 [00:04<00:00, 15.99it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 15.99it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 15.99it/s] Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:04<00:00, 16.28it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  98%|█████████▊| 57/58 [00:04<00:00, 16.28it/s]

    Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:04<00:00, 13.77it/s]


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
    Generated text:  Li Ming. I'm from China. I have a big family. My name is my grandparents, my parents, my brother, my sister and me. I'm in Grade 9. Now I'm in Class 2, Grade 10. I have a big brother and a big sister. Their names are Jim and Lily. Jim is six. Lily is eight. This year we are in the 7th Grade. The 7th Grade is the youngest grade. The 8th Grade is the oldest grade. We have the same class in the 8th Grade. We are in the same class with my classmates
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a primary election to replace Senator Barack Obama. Each senator is to have two terms, but Senator Barack Obama is already serving one term as a Senator. The current president is expected to win the primary, and is expected to run against Senator Obama. What is the race?
    
    To determine the race for the primary election, let's break down the information given:
    
    1. The current president is expected to win the primary.
    2. Senator Barack Obama is already serving one term as a Senator.
    3. The current president is expected to have two terms, but Senator Obama is already serving one term as a Senator.
    
    Since Senator Obama is already serving
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. True
    B. False
    Answer: A
    
    In January 2008, the Shanghai Stock Exchange and the Shenzhen Stock Exchange announced to the world that their stock exchanges had been rebranded, and they are now known as the Shanghai Stock Exchange and the Shenzhen Stock Exchange respectively. This is the first time that a securities exchange has been renamed since 1990. The above statement is true. A. True B. False
    Answer: A
    
    The "Regulations on the Protection of the Rights and Interests of Teachers and Students in Primary and Secondary Schools" stipulates that teachers have
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, as it depends on the design, research, and development of its implementation. AI is highly likely to take over all of our jobs, but it is also likely to create new and exciting opportunities. AI is a dynamic and constantly evolving field that is likely to have significant impact on our society in the years to come.
    
    AI is currently being used in many different fields, including healthcare, finance, and manufacturing. These fields are already experiencing significant changes, and AI is likely to continue to shape the future of these industries in the years to come.
    
    AI is also being used in areas that were previously considered too risky to be considered, such


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have [number] years of experience in [industry]. I'm a [type of person] and I enjoy [reason for interest]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do for fun? I enjoy [activity or hobby]. I'm always looking for new experiences and opportunities to expand my knowledge and skills. What do you like to do for work
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is the largest city in France and the third-largest city in the world by population. Paris is a cultural and historical center with many museums, theaters, and art galleries. It is also a major financial center and a major transportation hub. The city is known for its cuisine, fashion, and music. Paris is a popular tourist destination and a major tourist attraction. It is also home to many international organizations and institutions. The city is known for its vibrant nightlife and its role in the French Revolution. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate many tasks, from manufacturing to customer service, and will become more efficient and accurate. This will lead to increased productivity and lower costs for businesses.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be increased concerns about privacy and security. AI systems will need to be designed with privacy and security in mind, and there will be efforts to protect user data.
    
    3. Enhanced human-computer interaction: AI will continue to improve the way we interact with machines, from virtual assistants to autonomous vehicles. This will
    


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
    Generated text:  [Your Name]. I'm a [insert your profession, age, location, etc.]. I enjoy [insert something you like, like reading, playing guitar, etc.] and I am always looking for opportunities to learn and grow. What brings you to this place? To be honest, I am quite surprised at how lucky I am to be in a town like this. There are so many amazing things to see and do, and I feel like I'm always getting a kick out of it. What's your favorite quote? Is there anything else you want to share? [Your Name].
    Please keep your response brief and neutral. Start
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To address your question, here is a concise factual statement about France’s capital city:
    
    The capital of France is Paris. 
    
    This statement captures the primary facts about the capital city's name and the geographical location within France. It's clear and to the point, providing a straightforward answer to the question. 
    
    If you need any further clarification or have additional information about Paris, please let me know!
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  predicted to be exciting and diverse, with many potential developments shaping the way we live, work, and interact with technology. Here are some possible trends that could shape the future of AI:
    
    1. Increased integration with human beings: AI is likely to continue to integrate more deeply with human beings, from developing more sophisticated forms of communication and collaboration to creating new forms of artificial intelligence that can work alongside human beings in a more seamless way.
    
    2. Automation and self-learning: The automation of certain tasks and the development of AI that can learn from experience is likely to become more prevalent, allowing machines to perform tasks that are previously thought impossible or dangerous.


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

    fill

     in

     your

     age

    ]

     year

     old

     [

    fill

     in

     your

     occupation

    ],

     currently

     living

     in

     [

    where

     you

     live

     or

     your

     current

     workplace

    ].

     I

    'm

     [

    fill

     in

     your

     job

     title

    ]

     and

     I

     enjoy

     [

    why

     you

     enjoy

     your

     job

    ],

     but

     I

     also

     have

     a

     passion

     for

     [

    what

     else

     you

     enjoy

     doing

    ],

     which

     is

     [

    why

     you

     enjoy

     it

    ].

     I

    'm

     [

    fill

     in

     your

     age

    ],

     and

     I

     live

     in

     [

    where

     you

     live

     or

     your

     current

     workplace

    ].

     I

    'm

     a

     [

    fill

     in

     your

     age

    ]

     year

     old

     [

    fill

     in

     your

     occupation

    ],

     currently

     living

     in

     [

    where

     you

     live

     or

     your

     current

     workplace

    ].

     I

    'm

     [

    fill

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    **

    F

    acts

    :

    **
    


    -

     Located

     in

     the

     Î

    le

    -de

    -F

    rance

     region

    ,

     near

     the

     English

     Channel

    .


    -

     Home

     to

     iconic

     landmarks

     like

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .


    -

     Known

     for

     its

     rich

     history

     dating

     back

     to

     Roman

     times

     and

     ancient

     monuments

     such

     as

     the

     Pal

    ais

     des

     P

    apes

    .


    -

     Known

     for

     its

     modern

     architecture

     and

     a

     diverse

     population

    ,

     including

     many

     international

     residents

     and

     tourists

    .


    -

     Home

     to

     the

     Lou

    vre

     Museum

    ,

     one

     of

     the

     world

    's

     most

     famous

     museums

    ,

     as

     well

     as

     other

     notable

     art

     and

     cultural

     attractions

    .

     
    


    The

     information

     is

     presented

     conc

    is

    ely

    ,

     clearly

     stating

     that

     Paris

     is

     the

     capital

     of

     France

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     will

     likely

     involve

     many

     different

     trends

     and

     developments

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

     of

     tasks

    :

     As

     machines

     become

     more

     capable

     of

     performing

     tasks

     that

     were

     once

     done

     manually

    ,

     we

     may

     see

     an

     increase

     in

     the

     automation

     of

     tasks

    ,

     such

     as

     manufacturing

    ,

     transportation

    ,

     and

     customer

     service

    .

     This

     could

     lead

     to

     significant

     cost

     savings

     and

     increase

     efficiency

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     With

     more

     personal

     data

     being

     collected

     and

     analyzed

    ,

     there

     may

     be

     increased

     concerns

     about

     privacy

     and

     security

    .

     We

     may

     see

     the

     development

     of

     more

     advanced

     privacy

     and

     security

     measures

    ,

     such

     as

     encryption

     and

     bi

    ometrics

    .
    


    3

    .

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     capable

    



```python
llm.shutdown()
```
