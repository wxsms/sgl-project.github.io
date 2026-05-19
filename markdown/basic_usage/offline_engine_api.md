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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.33it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.33it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.33it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.33it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.33it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.17it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.17it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.17it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.17it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:03, 11.17it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 15.03it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 15.03it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:02, 15.03it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 15.03it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 15.03it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 22.95it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 22.95it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 22.95it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 22.95it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 22.95it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 22.95it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 28.30it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 28.30it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 28.30it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 28.30it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 28.30it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 28.30it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 32.71it/s]

    Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 38.33it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 38.33it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 38.33it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 38.33it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 38.33it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 38.33it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 38.33it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 38.33it/s]

    Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 44.90it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 44.90it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 44.90it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 44.90it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 44.90it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 44.90it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 44.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.38 GB):   2%|▏         | 1/58 [00:00<00:07,  7.17it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.35 GB):   2%|▏         | 1/58 [00:00<00:07,  7.17it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.35 GB):   3%|▎         | 2/58 [00:00<00:07,  7.18it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.34 GB):   3%|▎         | 2/58 [00:00<00:07,  7.18it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.34 GB):   5%|▌         | 3/58 [00:00<00:07,  7.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.34 GB):   5%|▌         | 3/58 [00:00<00:07,  7.43it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.34 GB):   7%|▋         | 4/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.34 GB):   7%|▋         | 4/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.34 GB):   9%|▊         | 5/58 [00:00<00:06,  7.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.34 GB):   9%|▊         | 5/58 [00:00<00:06,  7.96it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=56.34 GB):  10%|█         | 6/58 [00:00<00:06,  8.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.33 GB):  10%|█         | 6/58 [00:00<00:06,  8.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.33 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.32 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.67it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=56.32 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.32 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.32 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.31 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.53it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=56.31 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.31 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.31 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.31 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.31 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.09it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=56.30 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.30 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.30 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.30 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.30 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.29 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.38it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=56.29 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.29 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.46it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.29 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.27 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.27 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.73it/s]Capturing num tokens (num_tokens=960 avail_mem=56.28 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.73it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=56.28 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.73it/s]Capturing num tokens (num_tokens=832 avail_mem=56.28 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.73it/s]Capturing num tokens (num_tokens=832 avail_mem=56.28 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.56it/s]Capturing num tokens (num_tokens=768 avail_mem=56.27 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.56it/s]Capturing num tokens (num_tokens=704 avail_mem=56.27 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.56it/s]Capturing num tokens (num_tokens=640 avail_mem=56.22 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.56it/s]

    Capturing num tokens (num_tokens=640 avail_mem=56.22 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.80it/s]Capturing num tokens (num_tokens=576 avail_mem=55.98 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.80it/s]Capturing num tokens (num_tokens=512 avail_mem=55.97 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.80it/s]Capturing num tokens (num_tokens=480 avail_mem=55.98 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.80it/s]Capturing num tokens (num_tokens=480 avail_mem=55.98 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.88it/s]Capturing num tokens (num_tokens=448 avail_mem=55.98 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.88it/s]Capturing num tokens (num_tokens=416 avail_mem=55.98 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.88it/s]

    Capturing num tokens (num_tokens=384 avail_mem=55.98 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.88it/s]Capturing num tokens (num_tokens=384 avail_mem=55.98 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.81it/s]Capturing num tokens (num_tokens=352 avail_mem=55.97 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.81it/s]Capturing num tokens (num_tokens=320 avail_mem=55.97 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.81it/s]Capturing num tokens (num_tokens=288 avail_mem=55.97 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.81it/s]Capturing num tokens (num_tokens=288 avail_mem=55.97 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.61it/s]Capturing num tokens (num_tokens=256 avail_mem=55.96 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.61it/s]

    Capturing num tokens (num_tokens=240 avail_mem=55.96 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.61it/s]Capturing num tokens (num_tokens=224 avail_mem=55.96 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.61it/s]Capturing num tokens (num_tokens=224 avail_mem=55.96 GB):  67%|██████▋   | 39/58 [00:02<00:00, 22.09it/s]Capturing num tokens (num_tokens=208 avail_mem=55.95 GB):  67%|██████▋   | 39/58 [00:02<00:00, 22.09it/s]Capturing num tokens (num_tokens=192 avail_mem=55.95 GB):  67%|██████▋   | 39/58 [00:02<00:00, 22.09it/s]Capturing num tokens (num_tokens=176 avail_mem=55.95 GB):  67%|██████▋   | 39/58 [00:02<00:00, 22.09it/s]Capturing num tokens (num_tokens=176 avail_mem=55.95 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.94it/s]Capturing num tokens (num_tokens=160 avail_mem=55.95 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.94it/s]

    Capturing num tokens (num_tokens=144 avail_mem=55.94 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.94it/s]Capturing num tokens (num_tokens=128 avail_mem=55.94 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.94it/s]Capturing num tokens (num_tokens=128 avail_mem=55.94 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.63it/s]Capturing num tokens (num_tokens=112 avail_mem=55.94 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.63it/s]Capturing num tokens (num_tokens=96 avail_mem=55.93 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.63it/s] Capturing num tokens (num_tokens=80 avail_mem=55.93 GB):  78%|███████▊  | 45/58 [00:03<00:00, 22.63it/s]

    Capturing num tokens (num_tokens=80 avail_mem=55.93 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.46it/s]Capturing num tokens (num_tokens=64 avail_mem=55.93 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.46it/s]Capturing num tokens (num_tokens=48 avail_mem=55.92 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.46it/s]Capturing num tokens (num_tokens=32 avail_mem=55.92 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.46it/s]Capturing num tokens (num_tokens=32 avail_mem=55.92 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.45it/s]Capturing num tokens (num_tokens=28 avail_mem=55.91 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.45it/s]Capturing num tokens (num_tokens=24 avail_mem=55.91 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.45it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.91 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.45it/s]Capturing num tokens (num_tokens=20 avail_mem=55.91 GB):  93%|█████████▎| 54/58 [00:03<00:00, 22.72it/s]Capturing num tokens (num_tokens=16 avail_mem=55.91 GB):  93%|█████████▎| 54/58 [00:03<00:00, 22.72it/s]Capturing num tokens (num_tokens=12 avail_mem=55.90 GB):  93%|█████████▎| 54/58 [00:03<00:00, 22.72it/s]Capturing num tokens (num_tokens=8 avail_mem=55.90 GB):  93%|█████████▎| 54/58 [00:03<00:00, 22.72it/s] Capturing num tokens (num_tokens=8 avail_mem=55.90 GB):  98%|█████████▊| 57/58 [00:03<00:00, 22.44it/s]Capturing num tokens (num_tokens=4 avail_mem=55.90 GB):  98%|█████████▊| 57/58 [00:03<00:00, 22.44it/s]

    Capturing num tokens (num_tokens=4 avail_mem=55.90 GB): 100%|██████████| 58/58 [00:03<00:00, 16.51it/s]


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
    Generated text:  Kelsey. I am the founder of my own eLearning platform which includes modules for teachers, students, and learners. I have developed an online course management system to ensure students are always up to date and have access to resources. I am committed to creating a positive and inclusive learning environment, and I encourage students to provide feedback on the platform. I have a background in both education and technology, and I believe that technology can be a powerful tool to enhance learning and education. I am always open to new ideas and approaches to learning, and I am excited about the opportunities that my platform provides. How can I connect with you and provide feedback
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person who is elected by the people, and they are the head of the government. Is this statement correct?
    A. Correct
    B. Incorrect
    C. Can't tell
    D. I don't know
    Answer:
    A
    
    According to the latest data released by the United Nations Economic Commission for Europe (UNECE), in 2019, the total number of workers in EU member states reached 82.123 million. It is estimated that the proportion of women workers in the EU workforce is 44.6%. Based on this information, which of the following options is correct? A) 
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    B. London
    C. Rome
    D. Madrid
    
    The capital of France is Paris. 
    
    Therefore, the correct answer is:
    
    A. Paris
    
    The other options are incorrect because:
    
    B. London is the capital of England.
    C. Rome is the capital of Italy.
    D. Madrid is the capital of Spain. 
    
    None of these cities are the capital of France. Madrid is in Spain, and Paris is in France. London, Rome, and Madrid are in different countries. The correct capital of France is Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  here. It is here to make our lives better, and it is here to make our lives more interesting.
    What is the most likely to be used to control AI?
    Options are: A. computer; B. dictionary; C. book; D. database; E. none of the above;
    Answer:
    
    A. computer
    
    The most likely to be used to control AI is a computer. Computers are programmed to process information, analyze data, and make decisions. They can be used to create and control AI systems, allowing them to perform tasks in a controlled manner. However, the other options listed, such as a dictionary, book,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your profession or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do for fun? I enjoy [insert a short description of your favorite hobby or activity here]. I'm always looking for new experiences and opportunities to expand my horizons. What's your favorite hobby or activity? I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the third-largest city in the world by population. The city is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich history, including the influence of French colonialism in the Americas and its role in the French Revolution and Napoleonic Wars. The city is a major center for art, culture, and commerce, and is a popular tourist destination. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to the needs of their users.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and responsible development. This could lead to more stringent regulations and standards for AI development
    


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
    Generated text:  [insert your name here], and I am a [insert your occupation here]. I'm currently pursuing my [insert your career goal here]. This is my first time in the field of [insert your field of interest here]. 
    
    I am always looking for opportunities to learn and grow, and I am confident that I will make valuable contributions to [insert your field of interest here]. I am always up-to-date with the latest trends and advancements in my field, and I am eager to share my knowledge and insights with others. 
    
    I am a team player, and I am always willing to help others. I am a problem solver, and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the 12th-largest in the world by population. Located on the Loire River, Paris is a historic and cultural center in Northwestern France. The city is known for its stunning architecture, vibrant arts scene, and annual celebrations such as the Eiffel Tower ribbon cutting and the World Cup. Paris is a major transportation hub and a major tourist destination. The city is also home to numerous museums, theaters, and restaurants, making it a popular destination for visitors and residents alike. Despite its historical significance, Paris is also a multicultural city with over 350 languages spoken and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with many exciting developments on the horizon. Here are some possible trends that could shape the AI landscape in the coming years:
    
    1. More advanced AI: AI technology is getting better at natural language processing, image recognition, and other tasks that were previously considered impossible to do. This means that we can expect more complex and sophisticated AI systems to be developed in the future.
    
    2. Improved personalization: As AI systems become more advanced, they will be able to learn from user behavior and preferences to provide more personalized experiences. This will lead to better user experiences and a more efficient use of resources.
    
    3. Autonomous AI: Autonomous AI will


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

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     who

     was

     born

     in

     [

    Birth

    place

    ]

     and

     grew

     up

     in

     [

    Place

    ].

     I

    'm

     a

     [

    occupation

    ]

     who

     loves

     [

    ex

    cellent

     skill

     or

     hobby

    ]

     and

     [

    description

     of

     something

     you

     enjoy

     doing

    ].

     I

    'm

     also

     a

     [

    occupation

    ]

     who

     has

     always

     been

     [

    interest

    ].

     I

    'm

     a

     [

    occupation

    ]

     who

     is

     [

    description

     of

     a

     personal

     trait

     or

     quality

    ].

     [

    Name

    ],

     your

     character

    's

     background

     and

     personality

     are

     important

     to

     me

    ,

     and

     I

     want

     to

     share

     them

     with

     you

    .

     Thanks

    !

     How

     can

     I

     assist

     you

     with

     your

     self

    -int

    roduction

    ?

     Here

     are

     some

     suggestions

    :
    


    1

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     historic

     landmarks

    ,

     vibrant

     culture

    ,

     and

     luxurious

     fashion

    .

     Paris

     is

     home

     to

     the

     Lou

    vre

     Museum

     and

     the

     E

    iff

    el

     Tower

    ,

     as

     well

     as

     other

     famous

     attractions

     like

     the

     Palace

     of

     Vers

    ailles

    ,

     the

     Palace

     of

     Orange

    ,

     and

     the

     Lou

    vre

     Pyramid

    .

     The

     city

     also

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     the

     Renaissance

    .

     Paris

     is

     the

     third

     most

     populous

     city

     in

     the

     world

     and

     is

     known

     for

     its

     rich

     culture

    ,

     arts

    ,

     and

     cuisine

    .

     It

     is

     a

     major

     tourist

     destination

     and

     attracts

     millions

     of

     visitors

     each

     year

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     trends

     that

     could

     change

     the

     world

     as

     we

     know

     it

    .

     Here

     are

     some

     of

     the

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     With

     the

     advancement

     of

     machine

     learning

     and

     deep

     learning

     techniques

    ,

     AI

     systems

     are

     becoming

     increasingly

     intelligent

     and

     capable

     of

     performing

     complex

     tasks

     that

     were

     previously

     thought

     to

     be

     impossible

    .
    


    2

    .

     Automation

     of

     human

     jobs

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     are

     likely

     to

     automate

     many

     of

     the

     jobs

     that

     humans

     currently

     do

    .

     This

     could

     lead

     to

     a

     significant

     reduction

     in

     the

     need

     for

     human

     workers

     and

     could

     create

     new

     job

     opportunities

    .
    


    3

    .

     Integration

     with

     other

     technologies

    :

     As

     AI

     continues

     to

     advance

    



```python
llm.shutdown()
```
