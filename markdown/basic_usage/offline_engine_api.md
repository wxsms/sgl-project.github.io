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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:05,  1.20s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:05,  1.20s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:05,  1.20s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.69it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.69it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.69it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.69it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.11it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.11it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  7.11it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  7.11it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  7.11it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.95it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.95it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.95it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.95it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.56it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 13.56it/s]

    Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.97it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.97it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.97it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.97it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.97it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 23.97it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 28.60it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 28.60it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 28.60it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 28.60it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 28.60it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 28.60it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 32.53it/s]

    Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 37.51it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 37.51it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 37.51it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 37.51it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 37.51it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 37.51it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 40.31it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 40.31it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 40.31it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 40.31it/s]

    Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 40.31it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 40.31it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 40.31it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 40.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 47.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.78 GB):   2%|▏         | 1/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.75 GB):   2%|▏         | 1/58 [00:00<00:07,  7.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.75 GB):   3%|▎         | 2/58 [00:00<00:07,  7.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.75 GB):   3%|▎         | 2/58 [00:00<00:07,  7.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.75 GB):   5%|▌         | 3/58 [00:00<00:07,  7.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.74 GB):   5%|▌         | 3/58 [00:00<00:07,  7.54it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.74 GB):   7%|▋         | 4/58 [00:00<00:07,  7.71it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.74 GB):   7%|▋         | 4/58 [00:00<00:07,  7.71it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.74 GB):   9%|▊         | 5/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.74 GB):   9%|▊         | 5/58 [00:00<00:06,  7.97it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=57.74 GB):  10%|█         | 6/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.73 GB):  10%|█         | 6/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.73 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.58it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.73 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.58it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.73 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.73 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.72 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.72 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.72 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.49it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.71 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.71 GB):  21%|██        | 12/58 [00:01<00:04, 10.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.11 GB):  21%|██        | 12/58 [00:01<00:04, 10.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.36 GB):  21%|██        | 12/58 [00:01<00:04, 10.03it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.36 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.35 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.35 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.35 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.34 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.44it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=55.34 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.34 GB):  31%|███       | 18/58 [00:01<00:03, 11.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.34 GB):  31%|███       | 18/58 [00:01<00:03, 11.27it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=55.34 GB):  31%|███       | 18/58 [00:01<00:03, 11.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.34 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.32 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.80it/s]Capturing num tokens (num_tokens=960 avail_mem=54.48 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.80it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=54.48 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.26it/s]Capturing num tokens (num_tokens=896 avail_mem=54.48 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.26it/s]Capturing num tokens (num_tokens=832 avail_mem=54.47 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.26it/s]Capturing num tokens (num_tokens=832 avail_mem=54.47 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.66it/s]Capturing num tokens (num_tokens=768 avail_mem=54.47 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.66it/s]

    Capturing num tokens (num_tokens=704 avail_mem=54.47 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.66it/s]Capturing num tokens (num_tokens=704 avail_mem=54.47 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.46it/s]Capturing num tokens (num_tokens=640 avail_mem=54.46 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.46it/s]Capturing num tokens (num_tokens=576 avail_mem=54.46 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.46it/s]

    Capturing num tokens (num_tokens=576 avail_mem=54.46 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.93it/s]Capturing num tokens (num_tokens=512 avail_mem=54.45 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.93it/s]Capturing num tokens (num_tokens=480 avail_mem=54.46 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.93it/s]Capturing num tokens (num_tokens=480 avail_mem=54.46 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.70it/s]Capturing num tokens (num_tokens=448 avail_mem=54.46 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.70it/s]Capturing num tokens (num_tokens=416 avail_mem=54.46 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.70it/s]

    Capturing num tokens (num_tokens=384 avail_mem=54.46 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.70it/s]Capturing num tokens (num_tokens=384 avail_mem=54.46 GB):  57%|█████▋    | 33/58 [00:02<00:01, 15.21it/s]Capturing num tokens (num_tokens=352 avail_mem=54.45 GB):  57%|█████▋    | 33/58 [00:02<00:01, 15.21it/s]Capturing num tokens (num_tokens=320 avail_mem=54.45 GB):  57%|█████▋    | 33/58 [00:02<00:01, 15.21it/s]Capturing num tokens (num_tokens=288 avail_mem=54.44 GB):  57%|█████▋    | 33/58 [00:02<00:01, 15.21it/s]

    Capturing num tokens (num_tokens=288 avail_mem=54.44 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.33it/s]Capturing num tokens (num_tokens=256 avail_mem=54.44 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.33it/s]Capturing num tokens (num_tokens=240 avail_mem=54.44 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.33it/s]Capturing num tokens (num_tokens=224 avail_mem=54.43 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.33it/s]Capturing num tokens (num_tokens=224 avail_mem=54.43 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.55it/s]Capturing num tokens (num_tokens=208 avail_mem=54.43 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.55it/s]Capturing num tokens (num_tokens=192 avail_mem=54.43 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.55it/s]

    Capturing num tokens (num_tokens=176 avail_mem=54.43 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.55it/s]Capturing num tokens (num_tokens=176 avail_mem=54.43 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.42it/s]Capturing num tokens (num_tokens=160 avail_mem=54.42 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.42it/s]Capturing num tokens (num_tokens=144 avail_mem=54.42 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.42it/s]Capturing num tokens (num_tokens=128 avail_mem=54.42 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.42it/s]Capturing num tokens (num_tokens=128 avail_mem=54.42 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.81it/s]Capturing num tokens (num_tokens=112 avail_mem=54.42 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.81it/s]

    Capturing num tokens (num_tokens=96 avail_mem=54.41 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.81it/s] Capturing num tokens (num_tokens=80 avail_mem=54.41 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.81it/s]Capturing num tokens (num_tokens=80 avail_mem=54.41 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.51it/s]Capturing num tokens (num_tokens=64 avail_mem=54.40 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.51it/s]Capturing num tokens (num_tokens=48 avail_mem=54.40 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.51it/s]Capturing num tokens (num_tokens=32 avail_mem=54.40 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.51it/s]

    Capturing num tokens (num_tokens=32 avail_mem=54.40 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.89it/s]Capturing num tokens (num_tokens=28 avail_mem=54.39 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.89it/s]Capturing num tokens (num_tokens=24 avail_mem=54.39 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.89it/s]Capturing num tokens (num_tokens=20 avail_mem=54.39 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.89it/s]Capturing num tokens (num_tokens=20 avail_mem=54.39 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.35it/s]Capturing num tokens (num_tokens=16 avail_mem=54.39 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.35it/s]Capturing num tokens (num_tokens=12 avail_mem=54.38 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.35it/s]

    Capturing num tokens (num_tokens=8 avail_mem=54.38 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.35it/s] Capturing num tokens (num_tokens=8 avail_mem=54.38 GB):  98%|█████████▊| 57/58 [00:03<00:00, 21.37it/s]Capturing num tokens (num_tokens=4 avail_mem=54.38 GB):  98%|█████████▊| 57/58 [00:03<00:00, 21.37it/s]Capturing num tokens (num_tokens=4 avail_mem=54.38 GB): 100%|██████████| 58/58 [00:04<00:00, 14.36it/s]


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
    Generated text:  Alex and I am 30 years old. I've been living here in Washington D. C. since 2012. I moved to the state of Washington when I was a child. My mom was a plant doctor and I have always had a strong interest in gardening. My father and brothers have always enjoyed gardening as well, so my family has always been very active gardeners. I have been in the garden since childhood.
    
    I am a fan of gardening and love taking photos of the garden and the people I care for. I have a small garden with plants such as marigolds and lavender, along with a small
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. What is the correct way to express this in Japanese?
    
    A. 人選され
    
    B. 人君
    
    C. 人を務め
    
    D. 人を務めない
    
    E. 人を務める
    
    Answer: B
    
    Explanation: The correct way to express "the president of the United States is a man" in Japanese is "人君" (naijū). This word literally translates to "President of the United States", but in Japanese, it is more commonly expressed as "人を務め" (naijū tezu) which means "President of
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the center of the country, in the area of the city of Paris. The national capital is a metropolitan area with the city of Paris, divided by the Seine into a great part and a small part, including the Seine valley, both of which are part of the city of Paris.
    
    Is the following statement correct: The capital of France is located in the center of the country, in the area of the city of Paris.
    To determine if the statement "The capital of France is located in the center of the country, in the area of the city of Paris" is correct, let's break it down:
    
    1.
    ===============================
    Prompt: The future of AI is
    Generated text:  set to revolutionize the way we communicate, work, and interact with technology. It is essential to understand the various types of AI that are currently available and how they can be used to enhance our lives.
    One of the most promising types of AI is Natural Language Processing (NLP), which is used to interpret and understand human language. This technology has the potential to revolutionize various industries, including healthcare, finance, and legal. NLP algorithms are capable of recognizing patterns and extracting insights from large amounts of text data, such as emails, news articles, and social media posts.
    Another type of AI that is gaining popularity is Computer Vision.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short description of your favorite activity]. I'm always looking for ways to improve myself and make the world a better place. What's your favorite book or movie? I love [insert a short
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. Paris is a major tourist destination and a major economic hub, with a diverse population of over 10 million people. The city is home to many famous museums, including the Musée d'Orsay and the Musée Rodin. It is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a vibrant and dynamic city with a rich cultural heritage and a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in various industries, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Integration of AI with other technologies: AI is already being integrated into various technologies, such as smart homes, self-driving cars, and virtual assistants. As AI continues to evolve, it is likely
    


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
    Generated text:  [insert name]. I am a [insert occupation or profession] with a passion for [insert a specific interest or hobby related to [insert a profession], e.g., writing, photography, music, etc.]. I love exploring the world and learning new things, and I'm always on the lookout for fresh ideas and unexpected connections. Whether I'm talking to strangers or coworkers, I never fail to leave a lasting impression. I'm always ready to learn and grow, and I'm always eager to share my experiences with you. 
    
    I believe in taking risks and pursuing my passions, and I'm always looking for opportunities to learn and grow
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France, with a population of over 2. 3 million people. It is the capital city of France and the country's largest city. It is also the country's most populous city and a major international hub. Paris is known for its rich history, beautiful architecture, and diverse culture, including its iconic Eiffel Tower. The city is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many other world-renowned landmarks. Paris is also the birthplace of many world-famous artists, including Pablo Picasso, Dali, and Millet. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid advancements and significant growth, but it is also likely to face challenges and disruptions. Here are some possible future trends in artificial intelligence:
    
    1. Enhanced cognitive capabilities: AI is likely to continue to improve its cognitive capabilities, making it more capable of solving complex problems and making decisions based on data. This will lead to new applications and advancements in AI, such as robotics, autonomous vehicles, and healthcare.
    
    2. Increased privacy concerns: As AI becomes more integrated into our lives, there will be increasing concerns about privacy and data security. We will need to address these concerns through regulations, data protection, and ethical guidelines.
    
    


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

    ]

     and

     I

     am

     a

     [

    Your

     Occupation

    ],

     [

    Your

     Age

    ],

     [

    Your

     Education

     Level

    ],

     [

    Your

     Experience

     Level

    ],

     [

    Your

     Special

    ization

    ]

     and

     [

    Your

     Inter

    ests

    /

    Values

    ].

     I

     am

     a

     [

    Your

     Profession

    ],

     [

    Your

     Specialty

    ],

     [

    Your

     Qual

    ification

    ],

     [

    Your

     Profession

    ],

     and

     [

    Your

     Special

    ization

    ].

     Throughout

     my

     life

    ,

     I

     have

     been

     [

    Your

     Career

     Goal

    ],

     [

    Your

     Long

    evity

     Goal

    ],

     [

    Your

     Spiritual

     Bel

    ief

    ],

     [

    Your

     Accom

    pl

    ishments

    ],

     and

     [

    Your

     Personal

     Legacy

    ].

     I

     am

     [

    Your

     Personality

    ],

     [

    Your

     Personality

     Traits

    ],

     [

    Your

     Personality

     Traits

    ],

     [

    Your

     Personality

     Traits

    ],

     and

     [

    Your

     Personality

     Traits

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     often

     referred

     to

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     iconic

     status

     in

     French

     culture

     and

     its

     romantic

     image

    .

     It

     is

     also

     the

     capital

     of

     the

     Paris

     region

    ,

     a

     UNESCO

     World

     Heritage

     site

     known

     for

     its

     historical

     and

     architectural

     landmarks

    .

     Paris

     is

     home

     to

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     notable

     attractions

    .

     It

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

     and

     is

     a

     major

     center

     of

     business

    ,

     finance

    ,

     and

     culture

     in

     the

     European

     Union

    .

     The

     city

     is

     also

     famous

     for

     its

     fashion

     industry

     and

     the

     annual

     "

    C

    er

    sei

     Festival

    "

     that

     celebrates

     the

     city

    's

     rich

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     exciting

     and

     varied

    .

     Some

     of

     the

     possible

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     there

     will

     likely

     be

     a

     growing

     interest

     in

     how

     to

     ensure

     that

     they

     are

     used

     eth

    ically

     and

     responsibly

    .
    


    2

    .

     Greater

     integration

     with

     natural

     language

     processing

    :

     With

     the

     growing

     importance

     of

     language

     processing

    ,

     there

     is

     likely

     to

     be

     more

     integration

     of

     AI

     systems

     with

     natural

     language

     processing

    ,

     allowing

     them

     to

     understand

     and

     respond

     to

     human

     speech

    .
    


    3

    .

     Increased

     use

     of

     AI

     for

     healthcare

    :

     With

     advancements

     in

     AI

    ,

     there

     is

     likely

     to

     be

     a

     growing

     interest

     in

     using

     AI

     to

     improve

     healthcare

     outcomes

    ,

     such

     as

     in

     diagnosis

     and

     treatment

     of

    



```python
llm.shutdown()
```
