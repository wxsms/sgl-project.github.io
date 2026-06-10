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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=53.23 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=53.23 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it]Capturing batches (bs=120 avail_mem=53.13 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it]Capturing batches (bs=112 avail_mem=53.13 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it]Capturing batches (bs=104 avail_mem=53.13 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it]Capturing batches (bs=96 avail_mem=53.13 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it] Capturing batches (bs=96 avail_mem=53.13 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.15it/s]Capturing batches (bs=88 avail_mem=53.12 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.15it/s]Capturing batches (bs=80 avail_mem=53.12 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.15it/s]Capturing batches (bs=72 avail_mem=53.12 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.15it/s]

    Capturing batches (bs=64 avail_mem=53.12 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.15it/s]Capturing batches (bs=64 avail_mem=53.12 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.51it/s]Capturing batches (bs=56 avail_mem=53.12 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.51it/s]Capturing batches (bs=48 avail_mem=53.12 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.51it/s]Capturing batches (bs=40 avail_mem=53.12 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.51it/s]Capturing batches (bs=40 avail_mem=53.12 GB):  60%|██████    | 12/20 [00:01<00:00, 11.67it/s]Capturing batches (bs=32 avail_mem=53.12 GB):  60%|██████    | 12/20 [00:01<00:00, 11.67it/s]

    Capturing batches (bs=24 avail_mem=53.11 GB):  60%|██████    | 12/20 [00:01<00:00, 11.67it/s]Capturing batches (bs=16 avail_mem=53.11 GB):  60%|██████    | 12/20 [00:01<00:00, 11.67it/s]Capturing batches (bs=16 avail_mem=53.11 GB):  75%|███████▌  | 15/20 [00:01<00:00, 13.15it/s]Capturing batches (bs=12 avail_mem=53.11 GB):  75%|███████▌  | 15/20 [00:01<00:00, 13.15it/s]Capturing batches (bs=8 avail_mem=52.84 GB):  75%|███████▌  | 15/20 [00:01<00:00, 13.15it/s] 

    Capturing batches (bs=4 avail_mem=52.82 GB):  75%|███████▌  | 15/20 [00:01<00:00, 13.15it/s]Capturing batches (bs=4 avail_mem=52.82 GB):  90%|█████████ | 18/20 [00:01<00:00, 15.44it/s]Capturing batches (bs=2 avail_mem=52.13 GB):  90%|█████████ | 18/20 [00:01<00:00, 15.44it/s]Capturing batches (bs=1 avail_mem=52.13 GB):  90%|█████████ | 18/20 [00:01<00:00, 15.44it/s]Capturing batches (bs=1 avail_mem=52.13 GB): 100%|██████████| 20/20 [00:01<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.55it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.33it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 22.67it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.27 GB):   2%|▏         | 1/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.24 GB):   2%|▏         | 1/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=51.24 GB):   2%|▏         | 1/58 [00:00<00:07,  7.65it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=51.24 GB):   5%|▌         | 3/58 [00:00<00:04, 13.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.23 GB):   5%|▌         | 3/58 [00:00<00:04, 13.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=51.23 GB):   5%|▌         | 3/58 [00:00<00:04, 13.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=51.23 GB):   9%|▊         | 5/58 [00:00<00:04, 12.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=51.23 GB):   9%|▊         | 5/58 [00:00<00:04, 12.31it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=51.22 GB):   9%|▊         | 5/58 [00:00<00:04, 12.31it/s]Capturing num tokens (num_tokens=5120 avail_mem=51.22 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=51.22 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=51.21 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.91it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=51.21 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=51.21 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.55it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=51.21 GB):  16%|█▌        | 9/58 [00:01<00:03, 12.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=51.21 GB):  19%|█▉        | 11/58 [00:01<00:09,  5.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=51.20 GB):  19%|█▉        | 11/58 [00:01<00:09,  5.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=51.20 GB):  19%|█▉        | 11/58 [00:01<00:09,  5.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=51.20 GB):  19%|█▉        | 11/58 [00:01<00:09,  5.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.20 GB):  19%|█▉        | 11/58 [00:01<00:09,  5.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.20 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=51.19 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=51.19 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=51.18 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.00it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=51.18 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.18 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.18 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=51.16 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.47it/s]Capturing num tokens (num_tokens=960 avail_mem=51.18 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.47it/s] Capturing num tokens (num_tokens=896 avail_mem=51.17 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.47it/s]

    Capturing num tokens (num_tokens=896 avail_mem=51.17 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.97it/s]Capturing num tokens (num_tokens=832 avail_mem=74.38 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.97it/s]Capturing num tokens (num_tokens=768 avail_mem=74.38 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.97it/s]Capturing num tokens (num_tokens=704 avail_mem=74.38 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.97it/s]Capturing num tokens (num_tokens=640 avail_mem=74.38 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.97it/s]Capturing num tokens (num_tokens=640 avail_mem=74.38 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.85it/s]Capturing num tokens (num_tokens=576 avail_mem=74.37 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.85it/s]Capturing num tokens (num_tokens=512 avail_mem=74.36 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.85it/s]Capturing num tokens (num_tokens=480 avail_mem=74.37 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.85it/s]Capturing num tokens (num_tokens=448 avail_mem=74.37 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.85it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.37 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.71it/s]Capturing num tokens (num_tokens=416 avail_mem=74.37 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.71it/s]Capturing num tokens (num_tokens=384 avail_mem=74.36 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.71it/s]Capturing num tokens (num_tokens=352 avail_mem=74.36 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.36 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.36 GB):  60%|██████    | 35/58 [00:02<00:00, 25.27it/s]Capturing num tokens (num_tokens=288 avail_mem=74.35 GB):  60%|██████    | 35/58 [00:02<00:00, 25.27it/s]Capturing num tokens (num_tokens=256 avail_mem=74.35 GB):  60%|██████    | 35/58 [00:02<00:00, 25.27it/s]Capturing num tokens (num_tokens=240 avail_mem=74.35 GB):  60%|██████    | 35/58 [00:02<00:00, 25.27it/s]Capturing num tokens (num_tokens=224 avail_mem=74.34 GB):  60%|██████    | 35/58 [00:02<00:00, 25.27it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.34 GB):  67%|██████▋   | 39/58 [00:02<00:00, 28.37it/s]Capturing num tokens (num_tokens=208 avail_mem=74.34 GB):  67%|██████▋   | 39/58 [00:02<00:00, 28.37it/s]Capturing num tokens (num_tokens=192 avail_mem=74.34 GB):  67%|██████▋   | 39/58 [00:02<00:00, 28.37it/s]Capturing num tokens (num_tokens=176 avail_mem=74.34 GB):  67%|██████▋   | 39/58 [00:02<00:00, 28.37it/s]Capturing num tokens (num_tokens=160 avail_mem=74.33 GB):  67%|██████▋   | 39/58 [00:02<00:00, 28.37it/s]Capturing num tokens (num_tokens=160 avail_mem=74.33 GB):  74%|███████▍  | 43/58 [00:02<00:00, 30.81it/s]Capturing num tokens (num_tokens=144 avail_mem=74.33 GB):  74%|███████▍  | 43/58 [00:02<00:00, 30.81it/s]Capturing num tokens (num_tokens=128 avail_mem=74.33 GB):  74%|███████▍  | 43/58 [00:02<00:00, 30.81it/s]Capturing num tokens (num_tokens=112 avail_mem=74.33 GB):  74%|███████▍  | 43/58 [00:02<00:00, 30.81it/s]Capturing num tokens (num_tokens=96 avail_mem=74.32 GB):  74%|███████▍  | 43/58 [00:02<00:00, 30.81it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=74.32 GB):  81%|████████  | 47/58 [00:02<00:00, 32.00it/s]Capturing num tokens (num_tokens=80 avail_mem=74.32 GB):  81%|████████  | 47/58 [00:02<00:00, 32.00it/s]Capturing num tokens (num_tokens=64 avail_mem=74.32 GB):  81%|████████  | 47/58 [00:02<00:00, 32.00it/s]Capturing num tokens (num_tokens=48 avail_mem=74.31 GB):  81%|████████  | 47/58 [00:02<00:00, 32.00it/s]Capturing num tokens (num_tokens=32 avail_mem=74.31 GB):  81%|████████  | 47/58 [00:02<00:00, 32.00it/s]Capturing num tokens (num_tokens=32 avail_mem=74.31 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.43it/s]Capturing num tokens (num_tokens=28 avail_mem=74.31 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.43it/s]Capturing num tokens (num_tokens=24 avail_mem=74.30 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.43it/s]Capturing num tokens (num_tokens=20 avail_mem=74.30 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.43it/s]Capturing num tokens (num_tokens=16 avail_mem=74.30 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.43it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.29 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.43it/s]Capturing num tokens (num_tokens=12 avail_mem=74.29 GB):  97%|█████████▋| 56/58 [00:02<00:00, 36.31it/s]Capturing num tokens (num_tokens=8 avail_mem=74.29 GB):  97%|█████████▋| 56/58 [00:02<00:00, 36.31it/s] Capturing num tokens (num_tokens=4 avail_mem=74.29 GB):  97%|█████████▋| 56/58 [00:02<00:00, 36.31it/s]Capturing num tokens (num_tokens=4 avail_mem=74.29 GB): 100%|██████████| 58/58 [00:02<00:00, 19.85it/s]


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
    Generated text:  Tom and I am a writer and a software engineer. I have been working in software engineering for about two years now. I have a passion for learning new things and exploring new ideas. My previous job at a tech company was where I got to work with a team of experienced developers. During my time there, I was always pushing myself and learning new technologies. I enjoy creating my own solutions and developing my own software. I am also interested in designing and developing systems that can be used by people. I have a knack for coming up with unique and innovative ideas that can solve problems. Currently, I am working on a project to create an
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader, who assumes the presidency of the United States at the direction of the legislative branch of the government of the United States. The position is not incumbent. The position is not impeachable. The position has no term of service. The position does not have any leader in succession. The president is also the head of the executive branch. The position has no power to veto. The president is a president. The position has no official leader. The president has no official leader. The president is not a president.
    
    What is a valid term of service for the president of the United States? The president of the United States has no term
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Brussels
    C. London
    D. Rome
    
    The capital of France is:
    
    A. Paris
    
    Paris is the capital city of France. Brussels is the capital of Belgium, London is the capital city of the United Kingdom, and Rome is the capital city of Italy. Rome is known for its historical significance, but it is not the capital of France. The capital of France is Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of researchers, and there are several challenges that need to be overcome in order to achieve a better understanding of the effects of AI on human society. One of the most significant challenges is the potential for AI to perpetuate and exacerbate existing social and economic inequalities. Additionally, there are concerns about the potential for AI to have unintended consequences and impact on individuals and communities. To address these challenges, researchers have developed a range of methods for ethical and responsible use of AI, such as considering the potential for bias, unintended consequences, and the impact on marginalized populations. These methods have helped to pave the way for a more informed and responsible


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
    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    (Note: The statement should be a single, clear sentence that captures the essence of Paris's importance and cultural significance.) 
    
    The capital of France is Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    This statement succinctly captures the essence of Paris's importance and cultural significance, highlighting its iconic landmarks, cathedral, and diverse cultural scene. 
    
    The statement is concise, factual, and provides a clear understanding of the capital city's significance. 
    
    The capital of France is Paris,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there will be a greater emphasis on ethical considerations and the development of AI that is designed to be fair, transparent, and accountable.
    
    2. AI will become more integrated into everyday life: As AI becomes more integrated into our daily lives, we may see more widespread adoption of AI-powered technologies, such as voice assistants, self-driving cars, and smart home devices.
    
    3
    


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
    Generated text:  [Name] and I'm [Age]. I'm a [Professional/Personal/Technical/Other] with [Number of years working]. I've always been [a/an] and [a/an]. I'm [an employee, volunteer, etc.]. I'm a [dedicated, enthusiastic, relaxed, etc.]. I'm a [customer-focused, problem-solver, etc.]. I'm a [unique, quirky, interesting, etc.]. I'm a [best friend, teacher, leader, etc.]. I'm a [kind, compassionate, responsible, etc.]. I'm a [fun, silly,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. Other notable attractions include the Louvre Museum, Elysée Palace, and the Arc de Triomphe. Paris is a cultural and political center known for its rich history and diverse culture. It was founded by the ancient Gauls and was the first major city in the world to be written about in Latin. With a population of over 2 million people, it is the largest city in France by area and is home to many of the country’s major museums and attractions. Paris has a rich history and is a popular tourist destination. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and innovative, with many potential trends that could shape how it is used and developed. Here are some possible future trends in AI:
    
    1. Increased AI Transparency: As AI systems become more complex and sophisticated, there may be a need for greater transparency and explainability of their decisions. This could involve tools that enable users to understand how a system arrived at a particular decision, rather than relying on machine-generated output.
    
    2. Enhanced AI Ethics: As AI becomes more integrated into everyday life, there may be a need for greater ethical consideration of its use. This could involve frameworks for evaluating the potential impact of AI on individuals and society, and


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

    ].

     I

    'm

     a

     friendly

     and

     engaging

     individual

     who

     is

     always

     looking

     for

     ways

     to

     make

     people

     smile

    .

     I

     enjoy

     helping

     people

     feel

     at

     ease

     and

     create

     a

     positive

     impact

     in

     their

     lives

    .

     I

     have

     a

     lot

     of

     energy

     and

     I

    'm

     always

     ready

     to

     learn

     something

     new

    .

     If

     you

    're

     looking

     for

     someone

     to

     talk

     to

     and

     have

     a

     good

     time

    ,

     I

    'm

     your

     guy

    !

     

    😊

    😊

    😊

    
    


    That

     sounds

     like

     a

     great

     personality

    !

     Can

     you

     tell

     me

     more

     about

     your

     background

     and

     how

     you

     got

     started

     in

     this

     field

    ?

     Absolutely

    !

     I

     started

     my

     career

     as

     a

     Customer

     Service

     Representative

     at

     [

    Company

     Name

    ]

     a

     few

     years

     ago

    .

     At

     first

    ,

     I

     was

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     European

     Union

     and

     one

     of

     the

     most

     important

     cities

     in

     the

     world

    ,

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     art

     scene

    .

     Paris

     is

     famous

     for

     its

     museums

    ,

     landmarks

    ,

     and

     vibrant

     nightlife

    ,

     as

     well

     as

     its

     famous

     landmarks

     such

     as

     the

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

     Lou

    vre

     Museum

    .

     Paris

     has

     a

     long

     and

     stor

    ied

     history

     dating

     back

     to

     the

     Romans

    ,

     and

     it

     has

     played

     a

     major

     role

     in

     the

     development

     of

     French

     culture

     and

     politics

    .

     Today

    ,

     Paris

     is

     a

     major

     center

     for

     business

    ,

     finance

    ,

     and

     tourism

    ,

     and

     it

     continues

     to

     be

     a

     major

     cultural

     center

     in

     Europe

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     rapidly

     evolving

    ,

     and

     there

     are

     several

     possible

     trends

     that

     we

     can

     expect

     to

     see

     in

     the

     coming

     years

    .

     Here

     are

     some

     of

     the

     most

     likely

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     responsible

     AI

    :

     One

     of

     the

     most

     pressing

     concerns

     in

     the

     AI

     industry

     is

     the

     ethical

     implications

     of

     AI

    .

     As

     AI

     becomes

     more

     ubiquitous

    ,

     there

     will

     be

     an

     increasing

     need

     to

     ensure

     that

     it

     is

     used

     in

     ways

     that

     are

     fair

    ,

     transparent

    ,

     and

     responsible

    .

     This

     may

     lead

     to

     changes

     in

     the

     way

     that

     AI

     is

     developed

    ,

     designed

    ,

     and

     deployed

    ,

     as

     well

     as

     new

     approaches

     to

     problem

    -solving

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     is

     already

     being

    



```python
llm.shutdown()
```
