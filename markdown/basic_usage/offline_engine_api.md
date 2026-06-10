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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.62it/s]


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=77.01 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=77.01 GB):   5%|▌         | 1/20 [00:01<00:24,  1.28s/it]Capturing batches (bs=120 avail_mem=76.91 GB):   5%|▌         | 1/20 [00:01<00:24,  1.28s/it]Capturing batches (bs=112 avail_mem=76.91 GB):   5%|▌         | 1/20 [00:01<00:24,  1.28s/it]Capturing batches (bs=104 avail_mem=76.91 GB):   5%|▌         | 1/20 [00:01<00:24,  1.28s/it]Capturing batches (bs=104 avail_mem=76.91 GB):  20%|██        | 4/20 [00:01<00:04,  3.68it/s]Capturing batches (bs=96 avail_mem=76.91 GB):  20%|██        | 4/20 [00:01<00:04,  3.68it/s] Capturing batches (bs=88 avail_mem=76.91 GB):  20%|██        | 4/20 [00:01<00:04,  3.68it/s]Capturing batches (bs=80 avail_mem=76.91 GB):  20%|██        | 4/20 [00:01<00:04,  3.68it/s]

    Capturing batches (bs=80 avail_mem=76.91 GB):  35%|███▌      | 7/20 [00:01<00:01,  6.92it/s]Capturing batches (bs=72 avail_mem=76.91 GB):  35%|███▌      | 7/20 [00:01<00:01,  6.92it/s]Capturing batches (bs=64 avail_mem=76.91 GB):  35%|███▌      | 7/20 [00:01<00:01,  6.92it/s]Capturing batches (bs=56 avail_mem=76.90 GB):  35%|███▌      | 7/20 [00:01<00:01,  6.92it/s]Capturing batches (bs=56 avail_mem=76.90 GB):  50%|█████     | 10/20 [00:01<00:00, 10.13it/s]Capturing batches (bs=48 avail_mem=76.90 GB):  50%|█████     | 10/20 [00:01<00:00, 10.13it/s]Capturing batches (bs=40 avail_mem=76.90 GB):  50%|█████     | 10/20 [00:01<00:00, 10.13it/s]Capturing batches (bs=32 avail_mem=76.90 GB):  50%|█████     | 10/20 [00:01<00:00, 10.13it/s]

    Capturing batches (bs=32 avail_mem=76.90 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.32it/s]Capturing batches (bs=24 avail_mem=76.90 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.32it/s]Capturing batches (bs=16 avail_mem=76.90 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.32it/s]Capturing batches (bs=12 avail_mem=76.90 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.32it/s]Capturing batches (bs=12 avail_mem=76.90 GB):  80%|████████  | 16/20 [00:01<00:00, 15.03it/s]Capturing batches (bs=8 avail_mem=76.90 GB):  80%|████████  | 16/20 [00:01<00:00, 15.03it/s] Capturing batches (bs=4 avail_mem=76.90 GB):  80%|████████  | 16/20 [00:01<00:00, 15.03it/s]

    Capturing batches (bs=2 avail_mem=76.89 GB):  80%|████████  | 16/20 [00:01<00:00, 15.03it/s]Capturing batches (bs=1 avail_mem=76.89 GB):  80%|████████  | 16/20 [00:01<00:00, 15.03it/s]Capturing batches (bs=1 avail_mem=76.89 GB): 100%|██████████| 20/20 [00:01<00:00, 19.13it/s]Capturing batches (bs=1 avail_mem=76.89 GB): 100%|██████████| 20/20 [00:01<00:00, 10.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.56it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]

    Compiling num tokens (num_tokens=48):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=32):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=28):  67%|██████▋   | 39/58 [00:04<00:00, 22.76it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 36.08it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 36.08it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 36.08it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 36.08it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 36.08it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 36.08it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 36.08it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.91 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.88 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.88 GB):   3%|▎         | 2/58 [00:00<00:03, 16.05it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.88 GB):   3%|▎         | 2/58 [00:00<00:03, 16.05it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.87 GB):   3%|▎         | 2/58 [00:00<00:03, 16.05it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.87 GB):   3%|▎         | 2/58 [00:00<00:03, 16.05it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.87 GB):   9%|▊         | 5/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.87 GB):   9%|▊         | 5/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.85 GB):   9%|▊         | 5/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.85 GB):   9%|▊         | 5/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.85 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.85 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.85 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.12it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=60.84 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.84 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.84 GB):  21%|██        | 12/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.84 GB):  21%|██        | 12/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.83 GB):  21%|██        | 12/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.83 GB):  21%|██        | 12/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.83 GB):  21%|██        | 12/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.82 GB):  21%|██        | 12/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.82 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.82 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.82 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.20it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=60.82 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.80 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.20it/s]Capturing num tokens (num_tokens=960 avail_mem=60.81 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.20it/s] Capturing num tokens (num_tokens=960 avail_mem=60.81 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=896 avail_mem=60.81 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=832 avail_mem=60.81 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=768 avail_mem=60.80 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=704 avail_mem=60.80 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=704 avail_mem=60.80 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.85it/s]Capturing num tokens (num_tokens=640 avail_mem=60.80 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.85it/s]

    Capturing num tokens (num_tokens=576 avail_mem=60.79 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.85it/s]Capturing num tokens (num_tokens=512 avail_mem=60.78 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.85it/s]Capturing num tokens (num_tokens=480 avail_mem=60.80 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.85it/s]Capturing num tokens (num_tokens=480 avail_mem=60.80 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.84it/s]Capturing num tokens (num_tokens=448 avail_mem=60.79 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.84it/s]Capturing num tokens (num_tokens=416 avail_mem=60.79 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.84it/s]Capturing num tokens (num_tokens=384 avail_mem=60.79 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.84it/s]Capturing num tokens (num_tokens=352 avail_mem=60.78 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.84it/s]Capturing num tokens (num_tokens=320 avail_mem=60.78 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.84it/s]Capturing num tokens (num_tokens=320 avail_mem=60.78 GB):  60%|██████    | 35/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=288 avail_mem=60.78 GB):  60%|██████    | 35/58 [00:01<00:00, 39.41it/s]

    Capturing num tokens (num_tokens=256 avail_mem=60.77 GB):  60%|██████    | 35/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=240 avail_mem=60.77 GB):  60%|██████    | 35/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=224 avail_mem=60.77 GB):  60%|██████    | 35/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=224 avail_mem=60.77 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=208 avail_mem=60.76 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=192 avail_mem=60.76 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=176 avail_mem=60.76 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.26it/s]

    Capturing num tokens (num_tokens=160 avail_mem=60.76 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=160 avail_mem=60.76 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=144 avail_mem=60.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=128 avail_mem=60.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=112 avail_mem=60.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=96 avail_mem=60.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.02it/s] Capturing num tokens (num_tokens=80 avail_mem=60.74 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=80 avail_mem=60.74 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=64 avail_mem=60.74 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=48 avail_mem=60.73 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=32 avail_mem=60.73 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.83it/s]

    Capturing num tokens (num_tokens=28 avail_mem=60.73 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=24 avail_mem=60.72 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=24 avail_mem=60.72 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=20 avail_mem=60.72 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=16 avail_mem=60.72 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=12 avail_mem=60.72 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=8 avail_mem=60.71 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.94it/s] Capturing num tokens (num_tokens=4 avail_mem=60.71 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=4 avail_mem=60.71 GB): 100%|██████████| 58/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=4 avail_mem=60.71 GB): 100%|██████████| 58/58 [00:01<00:00, 35.35it/s]


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
    Generated text:  Vladyka and I'm a 27-year-old woman. I have a lot of medical issues and I've been living with my mother since childhood. I have lots of hair loss, and I have a lot of hair loss in my armpits, groin, and on my penis. I also have a lot of bald spots on my face, but I haven't noticed them as much as my hair loss. I also have a lot of rashes on my arms and neck, and I have been constantly scratching my skin. I've been having lots of extreme cases of stress in my life, and I've been feeling
    ===============================
    Prompt: The president of the United States is
    Generated text:  from which country?
    
    The president of the United States is from the United States. The current president is Joe Biden. He was elected in 2020 and served as the 46th president of the United States. He was re-elected in 2024 and served as the 48th president of the United States. He is the 45th and 46th president of the United States and is also the longest-serving president in U. S. history. He is a Democrat and has a bicameral legislature, the Senate and the House of Representatives. He is married to former First Lady
    ===============================
    Prompt: The capital of France is
    Generated text:  __________
    A. Paris
    B. Brussels
    C. London
    D. Stockholm
    Answer:
    
    A
    
    According to the "Regulations on the Protection of Computer Information Systems in the People's Republic of China," the establishment of a computer information system must first obtain ____.
    A. A legal entity
    B. An approval document
    C. A safety assessment
    D. A legal registration
    Answer:
    
    D
    
    For the ratio of the minimum value of a set of observed values to the maximum value, if this ratio is greater than 1, then the set of observed values is referred to as ____.
    A. Percentile
    
    ===============================
    Prompt: The future of AI is
    Generated text:  an exciting one, with new technologies making their way into everyday life and businesses around the world. However, it’s also important to be aware of the potential risks and hazards that come with this technology. One of the main risks of AI is the possibility of bias in the algorithms used to train it.
    Bias refers to unfair or inaccurate predictions that can be made by the algorithm. This can happen when the algorithm is not randomly selected or when it is not representative of the entire population. For example, if a person is unfairly selected for a job based on their race or gender, this can lead to a bias in the algorithm and inaccurate predictions.
    


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I've been working here for [number] years. I'm a [job title] at [company name], and I've been working here for [number] years. I'm a [job title] at [company name], and I've been working here for [number] years. I'm a [job title] at [company name], and I've been working here for [number] years. I'm a [job title] at [company name], and I've been
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major cultural and economic center, hosting many world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major hub for international business and diplomacy. It is also known for its rich history, including the influence of French Revolution, Napoleon Bonaparte, and the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and personalized medicine to virtual assistants and chatbots. Additionally, AI is likely to continue to be used for a wide range of applications, from improving healthcare outcomes to enhancing education and transportation systems. As AI becomes more integrated into our daily lives, it is likely to have a significant impact on society and the way we work and live. However, it is also important to consider the potential risks and challenges associated with AI, such as
    


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
    Generated text:  [Your Name] and I'm a [job title, such as "Director of Marketing," "CEO," or "Manager of Sales"]. Throughout my career, I've had the privilege of building and leading successful companies that have helped many businesses grow and succeed. I'm passionate about [mention a personal passion or area of interest] and I strive to make a positive impact on the world through my work. What's your role, interests, or hobbies? How can I become a part of your team? Let's get to know each other better. 🌟✨ #Interactions #SocialMediaInsider #BuildYourNetwork �
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as "The City of Light". It is an historic and cultural center, known for its iconic architecture, vibrant culture, and world-renowned art museums, including the Louvre. The city is home to over 10 million people and is one of the most populous cities in the world. Paris is also known as the "City of Miracles" due to its unique blend of Gothic and modern architectural styles. Additionally, the city has a rich history, with sites such as Notre-Dame Cathedral and the Champs-Elysees being significant landmarks. Despite its urbanization and urban sprawl, Paris remains
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several trends and developments that will shape its capabilities, applications, and ethics. Some possible future trends in AI include:
    
    1. Increased AI-Driven Automation: AI is already being used to automate routine tasks, such as data entry, customer service, and administrative tasks. As the technology continues to evolve, we can expect AI to become even more sophisticated and integrated into our daily lives, allowing us to automate more complex tasks.
    
    2. Enhanced Personalization and Ad-Strategy: AI will enable businesses to offer personalized experiences to customers, from recommending products based on their preferences to recommending content and services based on their browsing history or


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

    Occup

    ation

    ],

     and

     I

    'm

     really

     happy

     to

     meet

     you

    .

     I

    'm

     a

     [

    Brief

     Summary

     of

     Profession

    /

    Background

    ]

     who

     has

     been

     working

     in

     this

     field

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

    'm

     always

     up

    -to

    -date

     with

     the

     latest

     trends

     and

     technologies

     in

     this

     field

    ,

     and

     I

     love

     [

    Name

     of

     Profession

    /

    Background

    ],

     and

     I

     believe

     that

     [

    Name

     of

     Profession

    /

    Background

    ]

     is

     a

     great

     fit

     for

     me

    .

     If

     you

    're

     looking

     for

     someone

     who

     can

     provide

     innovative

     solutions

     to

     complex

     problems

    ,

     I

    'm

     your

     go

    -to

     person

    .

     I

    'm

     looking

     forward

     to

     the

     opportunity

     to

     help

     you

     and

     help

     you

     succeed

    .

     What

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     capital

     city

     of

     France

    ,

     located

     in

     the

     South

     of

     the

     country

    ,

     is

     Paris

    .

     It

     is

     the

     most

     populous

     city

     and

     the

     political

     and

     cultural

     center

     of

     France

    .

     The

     city

     has

     a

     rich

     history

     and

     a

     unique

     cultural

     landscape

    ,

     with

     landmarks

     like

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

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

     and

     its

     artistic

     and

     literary

     traditions

    .

     Despite

     its

     urban

     spraw

    l

    ,

     Paris

     is

     still

     a

     cosm

    opolitan

     and

     vibrant

     city

     with

     a

     diverse

     population

    .

     Its

     influence

     on

     global

     culture

     has

     been

     significant

    ,

     and

     it

     continues

     to

     be

     an

     important

     political

    ,

     economic

    ,

     and

     cultural

     hub

     in

     France

     and

     Europe

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

     with

     many

     potential

     directions

     and

     technologies

    .

     Some

     of

     the

     possible

     future

     trends

     in

     artificial

     intelligence

     include

    :
    


    1

    .

     More

     advanced

     natural

     language

     processing

    :

     With

     the

     increasing

     volume

     of

     data

     and

     the

     growing

     complexity

     of

     language

    ,

     natural

     language

     processing

     will

     become

     more

     sophisticated

    .

     This

     will

     enable

     machines

     to

     understand

     and

     respond

     to

     human

     speech

    ,

     generating

     more

     accurate

     and

     context

    ually

     appropriate

     responses

    .
    


    2

    .

     Enhanced

     machine

     learning

     and

     deep

     learning

    :

     With

     advancements

     in

     machine

     learning

     and

     deep

     learning

    ,

     AI

     systems

     will

     be

     able

     to

     perform

     complex

     tasks

     that

     were

     previously

     only

     possible

     with

     human

     intelligence

    .

     This

     includes

     tasks

     such

     as

     image

     and

     speech

     recognition

    ,

     autonomous

     driving

    ,

     and

     self

    -learning

     algorithms

    .
    


    3

    .

    



```python
llm.shutdown()
```
