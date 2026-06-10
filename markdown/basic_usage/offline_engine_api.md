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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.61it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:05,  1.18s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:05,  1.18s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:05,  1.18s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  6.96it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  6.96it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.96it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.96it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.96it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.63it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.63it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.63it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.63it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.63it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:05<00:03, 10.63it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:05<00:03, 10.63it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 26.18it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 26.18it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 26.18it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 26.18it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 26.18it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 26.18it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 26.18it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 26.18it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 34.57it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 44.29it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 52.66it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 52.66it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 52.66it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 52.66it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 52.66it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 52.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.81 GB):   2%|▏         | 1/58 [00:00<00:07,  7.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.78 GB):   2%|▏         | 1/58 [00:00<00:07,  7.54it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.78 GB):   3%|▎         | 2/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.78 GB):   3%|▎         | 2/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.78 GB):   5%|▌         | 3/58 [00:00<00:07,  7.67it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.78 GB):   5%|▌         | 3/58 [00:00<00:07,  7.67it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.78 GB):   7%|▋         | 4/58 [00:00<00:06,  7.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.78 GB):   7%|▋         | 4/58 [00:00<00:06,  7.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.78 GB):   9%|▊         | 5/58 [00:00<00:06,  8.08it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.77 GB):   9%|▊         | 5/58 [00:00<00:06,  8.08it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=38.77 GB):  10%|█         | 6/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.76 GB):  10%|█         | 6/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.76 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.76 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.76 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.77it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.76 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.49it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.75 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.75 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.49it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.75 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.74 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.74 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.74 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.74 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.47it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=38.73 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.47it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.73 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.73 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.73 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.39it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.73 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.73 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.72 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.72 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.72 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.70 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.13it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.70 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.88it/s]Capturing num tokens (num_tokens=960 avail_mem=38.72 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.88it/s] Capturing num tokens (num_tokens=896 avail_mem=38.71 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.88it/s]Capturing num tokens (num_tokens=896 avail_mem=38.71 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.12it/s]Capturing num tokens (num_tokens=832 avail_mem=38.71 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.12it/s]Capturing num tokens (num_tokens=768 avail_mem=38.71 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.12it/s]

    Capturing num tokens (num_tokens=768 avail_mem=38.71 GB):  43%|████▎     | 25/58 [00:02<00:02, 16.12it/s]Capturing num tokens (num_tokens=704 avail_mem=38.70 GB):  43%|████▎     | 25/58 [00:02<00:02, 16.12it/s]Capturing num tokens (num_tokens=640 avail_mem=38.70 GB):  43%|████▎     | 25/58 [00:02<00:02, 16.12it/s]Capturing num tokens (num_tokens=640 avail_mem=38.70 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.04it/s]Capturing num tokens (num_tokens=576 avail_mem=38.70 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.04it/s]Capturing num tokens (num_tokens=512 avail_mem=38.68 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.04it/s]

    Capturing num tokens (num_tokens=480 avail_mem=38.70 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.04it/s]Capturing num tokens (num_tokens=480 avail_mem=38.70 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.30it/s]Capturing num tokens (num_tokens=448 avail_mem=38.70 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.30it/s]Capturing num tokens (num_tokens=416 avail_mem=38.70 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.30it/s]Capturing num tokens (num_tokens=384 avail_mem=38.69 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.30it/s]Capturing num tokens (num_tokens=384 avail_mem=38.69 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.32it/s]Capturing num tokens (num_tokens=352 avail_mem=38.69 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.32it/s]

    Capturing num tokens (num_tokens=320 avail_mem=38.68 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.32it/s]Capturing num tokens (num_tokens=288 avail_mem=38.68 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.32it/s]Capturing num tokens (num_tokens=288 avail_mem=38.68 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.78it/s]Capturing num tokens (num_tokens=256 avail_mem=38.68 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.78it/s]Capturing num tokens (num_tokens=240 avail_mem=38.67 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.78it/s]Capturing num tokens (num_tokens=224 avail_mem=38.67 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.78it/s]

    Capturing num tokens (num_tokens=224 avail_mem=38.67 GB):  67%|██████▋   | 39/58 [00:02<00:00, 20.09it/s]Capturing num tokens (num_tokens=208 avail_mem=38.67 GB):  67%|██████▋   | 39/58 [00:02<00:00, 20.09it/s]Capturing num tokens (num_tokens=192 avail_mem=38.67 GB):  67%|██████▋   | 39/58 [00:02<00:00, 20.09it/s]Capturing num tokens (num_tokens=176 avail_mem=38.66 GB):  67%|██████▋   | 39/58 [00:02<00:00, 20.09it/s]Capturing num tokens (num_tokens=176 avail_mem=38.66 GB):  72%|███████▏  | 42/58 [00:02<00:00, 20.18it/s]Capturing num tokens (num_tokens=160 avail_mem=38.66 GB):  72%|███████▏  | 42/58 [00:02<00:00, 20.18it/s]Capturing num tokens (num_tokens=144 avail_mem=38.66 GB):  72%|███████▏  | 42/58 [00:03<00:00, 20.18it/s]

    Capturing num tokens (num_tokens=128 avail_mem=38.65 GB):  72%|███████▏  | 42/58 [00:03<00:00, 20.18it/s]Capturing num tokens (num_tokens=128 avail_mem=38.65 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.14it/s]Capturing num tokens (num_tokens=112 avail_mem=38.65 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.14it/s]Capturing num tokens (num_tokens=96 avail_mem=38.65 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.14it/s] Capturing num tokens (num_tokens=80 avail_mem=38.65 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.14it/s]

    Capturing num tokens (num_tokens=80 avail_mem=38.65 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.09it/s]Capturing num tokens (num_tokens=64 avail_mem=38.64 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.09it/s]Capturing num tokens (num_tokens=48 avail_mem=38.64 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.09it/s]Capturing num tokens (num_tokens=32 avail_mem=38.64 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.09it/s]Capturing num tokens (num_tokens=32 avail_mem=38.64 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.50it/s]Capturing num tokens (num_tokens=28 avail_mem=38.63 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.50it/s]Capturing num tokens (num_tokens=24 avail_mem=38.63 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.50it/s]

    Capturing num tokens (num_tokens=20 avail_mem=38.62 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.50it/s]Capturing num tokens (num_tokens=20 avail_mem=38.62 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.93it/s]Capturing num tokens (num_tokens=16 avail_mem=38.62 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.93it/s]Capturing num tokens (num_tokens=12 avail_mem=38.62 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.93it/s]Capturing num tokens (num_tokens=8 avail_mem=38.62 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.93it/s] Capturing num tokens (num_tokens=8 avail_mem=38.62 GB):  98%|█████████▊| 57/58 [00:03<00:00, 22.20it/s]Capturing num tokens (num_tokens=4 avail_mem=38.61 GB):  98%|█████████▊| 57/58 [00:03<00:00, 22.20it/s]

    Capturing num tokens (num_tokens=4 avail_mem=38.61 GB): 100%|██████████| 58/58 [00:03<00:00, 15.69it/s]


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
    Generated text:  John. I like to listen to music. But I don't have a music library. I find music through my computer. I like to use some music software. I use it to listen to music on my computer and to record music. I like to know about music. I often use music reviews to help me decide what to listen to next. I'm not very good at reading music. I only know the names of the songs on my computer and I'm not very good at reading them. I want to know more about music. My favorite musician is John Denver. He wrote songs for me. I like to sing along with him
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person.
    
    Does it follow that "There is a president of the United States."?
    Options are: a). yes; b). it is not possible to tell; c). no;
    
    a). Yes
    
    The statement "There is a president of the United States" logically follows that "There is a president of the United States" because a president is a person. The phrase "a president of the United States" is the direct translation of the English sentence "There is a president of the United States, " which is a grammatically correct and meaningful statement. So, based on the definition and translation of the sentence, we can confidently
    ===============================
    Prompt: The capital of France is
    Generated text: : ____
    A. Paris
    B. Lyon
    C. Lille
    D. Marseille
    
    To determine the capital of France, let's consider the following points:
    
    1. The capital of France is often referred to as "Paris," but this is not the correct answer.
    2. The capital of France is actually Lyon.
    3. The capital of France is not Lyon, but rather it is the capital of the department of the same name, which is the second largest department in France.
    
    Therefore, the correct answer is:
    
    \boxed{A} Lyon
    ===============================
    Prompt: The future of AI is
    Generated text:  a bright one with many possibilities, both for good and for bad. The latest research shows that a number of artificial intelligence systems are already outperforming or outperforming the best human AI systems. And as AI evolves, it’s likely to be used for more complex tasks than we can even imagine. But the path ahead is fraught with challenges and uncertainties.
    To help us navigate this complex landscape, we’ve brought together experts in AI and machine learning to provide a clear picture of what we can expect, and what we can’t. This month, we’re joined by Andrew Szokoly, Senior Director of Research at the Future of Life


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Gender] [Gender Identity] who has always been passionate about [What you love to do]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I am [What you do best]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and fashion, and is home to many famous museums, theaters, and restaurants. The city is known for its romantic and romantic atmosphere, and is a popular tourist destination for visitors from around the world. Paris is a city of contrasts, with its modern skyscrapers and historical architecture blending seamlessly into the surrounding landscape. Its status as the world's
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn and adapt to new situations. This will enable AI to perform tasks that are currently only possible with human intelligence, such as decision-making, creativity, and problem-solving.
    
    2. Enhanced privacy and security: As AI systems become more advanced, they will need to be designed with greater privacy and security in mind. This will require the development of new algorithms and techniques to protect sensitive data and prevent cyber attacks.
    
    3. Increased use of AI in healthcare: AI will play an increasingly important role in
    


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
    Generated text:  [Name], and I am a [Industry] [Skill] with [Number of Years] years of experience in [Industry]. I've always been interested in [What I love to do], and I've always wanted to [What I want to achieve], so I've decided to turn my passion into a career. I'm excited to be a part of your team, and I'm looking forward to learning more about our industry and what you do. What's your name? Can you tell me a little bit about your background and experience? Remember, it's important to be neutral and friendly in your introduction. Good luck! [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital and most populous city of France and the largest city in both Europe and the world. Located in the south of the country on the Île de France, it has a population of 1943,227, according to the 2021 census. It is also the seat of the Senate and the National Assembly, and is the second most populous city in Europe after Rome. The city is known for its museums, art, and historical sites, including the Louvre and the Musée d'Orsay, and is home to many cultural institutions. Its rich history and iconic landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but several trends are likely to shape its development:
    
    1. Increased AI autonomy: AI systems will become more capable of making decisions without human intervention, leading to more autonomous and ethical AI.
    
    2. AI ethics and privacy concerns: As AI systems become more sophisticated and powerful, they will be more likely to perpetuate existing inequalities and privacy issues. There will be a need for new ethical frameworks and regulations to govern AI development and deployment.
    
    3. AI for healthcare and patient care: AI will play an increasingly important role in medical diagnostics, treatment planning, and personalized medicine, improving patient outcomes and reducing costs.
    
    4. AI in finance


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

    's

     name

    ],

     and

     I

    'm

     a

    /an

     [

    insert

     character

    's

     occupation

     or

     profession

    ].

     I

    'm

     [

    insert

     character

    's

     age

    ],

     and

     I

    'm

     [

    insert

     character

    's

     personality

     traits

    ,

     such

     as

     intro

    verted

    ,

     ext

    ro

    verted

    ,

     logical

    ,

     etc

    .

    ].

     I

    'm

     [

    insert

     character

    's

     current

     profession

     or

     career

    ],

     and

     I

     enjoy

     [

    insert

     character

    's

     hobby

     or

     interest

    ].

     I

     like

     to

     [

    insert

     character

    's

     hobbies

     or

     interests

    ,

     such

     as

     reading

    ,

     music

    ,

     sports

    ,

     etc

    .

    ].

     I

    'm

     [

    insert

     character

    's

     favorite

     thing

     to

     do

    ],

     and

     I

    'm

     always

     looking

     for

     [

    insert

     character

    's

     goal

     or

     dream

    ].

     I

    'm

     [

    insert

     character

    's

     age

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     Se

    ine

     Valley

    ,

     the

     heart

     of

     the

     city

    .

     It

     is

     a

     historical

     and

     cultural

     center

    ,

     with

     over

     

    3

     million

     inhabitants

    .

     The

     city

     is

     famous

     for

     its

     art

    ,

     architecture

    ,

     food

    ,

     and

     fashion

    .

     Its

     skyline

     is

     composed

     of

     tall

     buildings

     and

     towering

     towers

    ,

     and

     its

     city

     centre

     is

     a

     hub

     of

     activity

    .

     Paris

     is

     a

     vibrant

     and

     dynamic

     city

    ,

     known

     for

     its

     artistic

     and

     intellectual

     atmosphere

    .

     Its

     culture

     is

     rich

     and

     diverse

    ,

     and

     it

     is

     home

     to many

     renowned

     institutions

     such

     as

     the

     Lou

    vre

     Museum

    ,

     the

     Notre

     Dame

     Cathedral

    ,

     and

     the

     E

    iff

    el

     Tower

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

     rich

     history

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

    ,

     and

     there

     are

     many

     possible

     trends

     that

     could

     shape

     its

     development

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     that

     we

     can

     expect

     to

     see

    :
    


    1

    .

     Increased

     Integration

     with

     Human

     Intelligence

    :

     As

     AI

     becomes

     more

     advanced

     and

     capable

    ,

     we

     may

     see

     an

     increase

     in

     integration

     between

     AI

     and

     human

     intelligence

    .

     For

     example

    ,

     AI

     could

     become

     more

     adept

     at

     understanding

     and

     predicting

     human

     emotions

    ,

     preferences

    ,

     and

     behaviors

    .
    


    2

    .

     More

     Util

    ization

     of

     AI

     for

     Medical

     Diagnosis

     and

     Treatment

    :

     As

     AI

     becomes

     more

     advanced

     and

     capable

    ,

     we

     may

     see

     an

     increase

     in

     its

     use

     for

     medical

     diagnosis

     and

     treatment

    .

     AI

     could

     be

     used

     to

     analyze

     medical

     images

    ,

     make

     predictions

     about

     patient

     outcomes

    ,

    



```python
llm.shutdown()
```
