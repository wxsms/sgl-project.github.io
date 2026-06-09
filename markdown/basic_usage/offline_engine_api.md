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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:19,  2.59it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  7.64it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  7.64it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  7.64it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:05,  7.64it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:05,  7.64it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.23it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.23it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.23it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.23it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 11.23it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:03, 11.23it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 16.19it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 16.19it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 16.19it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 16.19it/s]

    Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 16.19it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 16.19it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.35it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.35it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 21.35it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 21.35it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 21.35it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 21.35it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 21.35it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 28.22it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 34.33it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 34.33it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 34.33it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 34.33it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 34.33it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 34.33it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 34.33it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 39.39it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 39.39it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 39.39it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 39.39it/s]

    Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 39.39it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 39.39it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 39.39it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 43.44it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 43.44it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 43.44it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 43.44it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 43.44it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 43.44it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 43.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   2%|▏         | 1/58 [00:00<00:08,  6.94it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:08,  6.94it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.63it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:06,  8.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:06,  8.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:06,  8.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  8.00it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.57it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.47it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:03, 15.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:03, 15.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.04it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  33%|███▎      | 19/58 [00:01<00:02, 18.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 18.81it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:01<00:02, 18.81it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 18.81it/s] Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:01<00:01, 20.35it/s]Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:01<00:01, 20.35it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:01<00:01, 20.35it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:01<00:01, 20.35it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.46it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.46it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.46it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.46it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.39it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.39it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.39it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.39it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 19.17it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 19.17it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 19.17it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 19.17it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.68it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.68it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.68it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.68it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.71it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.71it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.71it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.71it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 21.95it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 21.95it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 21.95it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  69%|██████▉   | 40/58 [00:02<00:00, 21.95it/s]

    Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 22.79it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 22.79it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 22.79it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 22.79it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.74it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.74it/s] Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.74it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.74it/s]

    Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.11it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.11it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.11it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.11it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.88it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.88it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.88it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.88it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.42it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.42it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.42it/s] Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  95%|█████████▍| 55/58 [00:03<00:00, 25.42it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 25.67it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 18.84it/s]


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
    Generated text:  Martin and I’m a PhD student working at the University of Aberdeen in the School of Science and Engineering. My research is focused on the analysis of functional connectivity in high-dimensional spatiotemporal data and the use of machine learning algorithms for the reconstruction of spatiotemporal causal dynamics. My work is supported by the ERC Advanced Grant.
    In my thesis, I developed a new algorithm called Markov Random Fields (MRF) with a Markov chain Monte Carlo (MCMC) sampling method to estimate the full space-time functional connectivity matrix of a system. I used the Markov random fields model to derive a complete description of
    ===============================
    Prompt: The president of the United States is
    Generated text:  paid 200,000 dollars per year, what is this as a percentage of the total income of all corporations in the country?
    To determine what 200,000 dollars is as a percentage of the total income of all corporations in the United States, we need to follow these steps:
    
    1. Identify the total income of all corporations in the United States.
    2. Calculate the percentage of the total income that 200,000 dollars represents.
    
    Step 1: Identify the total income of all corporations in the United States.
    Let's denote the total income of all corporations in the United
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the north of the country. It is the largest city of France. Its area is approximately 112 square kilometers. The city has a population of around 1.8 million people. The city is a very important port city. It is located on the Seine River, which flows into the Seine estuary, which is located in the city. The Seine estuary is a famous area of the city. It is the location of the Seine bridge. In the city center, there is a large Parliament. The Parliament is the highest institution of the French government. It is also the seat of the highest
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting, but there are also some potential downsides. In this post, I will provide you with a list of the top 10 potential downsides of AI, along with a brief overview of each and how to mitigate them.
    1. Lack of Transparency and Accountability
    One of the biggest downsides of AI is that it can be difficult to understand how it is making decisions. This can be problematic if the AI system is involved in ethical or legal issues, such as discrimination or the violation of privacy. For example, if an AI system is used to prioritize customers based on their age, this could lead to discrimination against older customers


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] [Vehicle Name]. I have been driving for [Number of Years] years and have been driving for [Number of Months] months. I have been driving for [Number of Days] days and have been driving for [Number of Weeks] weeks. I have been driving for [Number of Months] months and have been driving for [Number of Weeks] weeks. I have been driving for [Number of Months] months and have been driving for [Number of Weeks] weeks. I have been driving for [Number of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French Riviera. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has played a significant role in French history and continues to be a major cultural and economic center in the world. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated forms of AI, such as those that can understand and respond to human emotions and motivations.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more rigorous testing and evaluation of AI systems, as well
    


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
    Generated text:  [insert fictional character's name] and I am a [insert fictional character's age] year old male. I currently live in [insert fictional city] and I work at [insert fictional company] as a [insert fictional role]. I am [insert fictional interest, if applicable] and [insert fictional hobbies, if applicable]. I'm always looking to learn new things and try new things to stay sharp and fresh in my career. How about you, [insert fictional character's name]?
    
    My name is [insert fictional character's name], and I am a [insert fictional character's age] year old male. I live in [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light.
    
    That's correct! Paris, also known as the City of Light, is the capital of France, and is one of the most iconic and beloved cities in the world. Its vibrant culture, beautiful architecture, and world-renowned museums and landmarks make it one of the most visited cities in the world. Paris is also known for its rich history, including its Notre-Dame Cathedral and its association with the French Revolution. Despite the challenges of living in a major city, Paris remains a beloved and exciting destination for tourists and locals alike. 
    
    Now that you've learned the facts about the capital of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  filled with possibilities and breakthroughs, and as technology continues to evolve, there are several trends that are likely to shape the future of AI in the coming years. Here are some of the most prominent trends:
    
    1. Increased focus on ethical AI: As we become increasingly aware of the potential consequences of AI, there will be an increased focus on developing ethical AI. This means that AI will become more transparent, accountable, and accountable to the ethical standards of society.
    
    2. AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and the potential for this technology to become even more advanced is likely. This could lead


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

     am

     [

    Occup

    ation

     or

     Professional

    ].

     I

     have

     always

     been

     fascinated

     by

     [

    what

     your

     character

     is

     passionate

     about

     or

     exc

    els

     in

    ].

     Whether

     it

    's

     [

    What

     they

     do

     for

     a

     living

    ]

     or

     [

    What

     they

     are

     good

     at

    ],

     I

    'm

     excited

     to

     learn

     more

     about

     you

     and

     help

     you

     grow

    .

     How

     would

     you

     describe

     your

     personal

     style

     and

     what

     motiv

    ates

     you

    ?

     Remember

    ,

     I

    'm

     here

     to

     learn

     more

     about

     you

     and

     help

     you

     achieve

     your

     goals

    .

     Let

    's

     start

     by

     having

     a

     chat

     about

     [

    What

     we

    're

     discussing

     at

     the

     beginning

     of

     the

     conversation

    ].

     How

     would

     you

     describe

     yourself

    ?

     I

     am

     [

    Name

    ],

     a

     [

    Occup

    ation

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     fact

     is

     a

     direct

     answer

     to

     the

     question

     "

    What

     is

     the

     capital

     of

     France

    ?"

     with

     a

     brief

     explanatory

     statement

    .

     
    


    For

     a

     more

     detailed

     answer

    ,

     here

    's

     an

     example

    :

     "

    Paris

     is

     the

     largest

     city

     in

     France

     and

     serves

     as

     its

     capital

    .

     The

     city

     is

     known

     for

     its

     architecture

    ,

     culture

    ,

     and

     history

    ."

     
    


    The

     statement

     is

     concise

    ,

     accurate

    ,

     and

     provides

     a

     clear

     and

     specific

     answer

     to

     the

     question

    .

     However

    ,

     there

     could

     be

     alternative

     ways

     to

     express

     the

     same

     information

    ,

     such

     as

     "

    Paris

    ,

     the

     capital

     of

     France

    ,

     is

     a

     large

     city

     with

     a

     rich

     cultural

     and

     historical

     significance

    ."

     


    Example

    :

     


    -

     "

    Paris

     is

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     complex

    ,

     but

     some

     possible

     trends

     that

     could

     emerge

     include

    :
    


    1

    .

     Improved

     ethical

     guidelines

    :

     There

     will

     be

     increasing

     emphasis

     on

     developing

     guidelines

     for

     how

     AI

     should

     be

     used

     and

     deployed

    .

     This

     could

     include

     guidelines

     for

     data

     privacy

    ,

     bias

    ,

     and

     transparency

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     into

     daily

     life

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     to

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     such

     as

     through

     voice

     assistants

    ,

     smart

     homes

    ,

     and

     other

     advanced

     technologies

    .
    


    3

    .

     AI

     will

     become

     more

     accessible

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     become

     increasingly

     accessible

     to

     people

     with

     varying

     levels

     of

     technical

     expertise

    .

     This

     could

     include

     increased

     access

     to

     AI

    



```python
llm.shutdown()
```
