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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:09<09:11,  9.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:09<09:11,  9.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:09<09:11,  9.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:09<09:11,  9.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:09<09:11,  9.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:09<01:17,  1.46s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:09<01:17,  1.46s/it]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:09<01:17,  1.46s/it]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:09<01:17,  1.46s/it]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:09<01:17,  1.46s/it]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:09<01:17,  1.46s/it]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:09<01:17,  1.46s/it]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:09<01:17,  1.46s/it]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:09<00:22,  2.09it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:09<00:08,  4.57it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:09<00:08,  4.57it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:10<00:08,  4.57it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:10<00:08,  4.57it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:10<00:08,  4.57it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:10<00:08,  4.57it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:10<00:08,  4.57it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:10<00:08,  4.57it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:10<00:08,  4.57it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:10<00:08,  4.57it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:10<00:03,  7.86it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:10<00:01, 11.52it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:10<00:01, 11.52it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:10<00:01, 11.52it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:10<00:01, 11.52it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:10<00:01, 11.52it/s]

    Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:10<00:01, 11.52it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:10<00:01, 11.52it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:10<00:01, 11.52it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:10<00:00, 15.21it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:10<00:00, 15.21it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:10<00:00, 15.21it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:10<00:00, 15.21it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:10<00:00, 15.21it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:10<00:00, 15.21it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:10<00:00, 15.21it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:10<00:00, 15.21it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:10<00:00, 19.41it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:10<00:00, 19.41it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:10<00:00, 19.41it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:10<00:00, 19.41it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:10<00:00, 19.41it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:10<00:00, 19.41it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:10<00:00, 19.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.70 GB):   2%|▏         | 1/58 [00:00<00:07,  7.35it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.67 GB):   2%|▏         | 1/58 [00:00<00:07,  7.35it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.67 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.67 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.67 GB):   5%|▌         | 3/58 [00:00<00:07,  7.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.66 GB):   5%|▌         | 3/58 [00:00<00:07,  7.51it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.66 GB):   7%|▋         | 4/58 [00:00<00:07,  7.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.66 GB):   7%|▋         | 4/58 [00:00<00:07,  7.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.66 GB):   9%|▊         | 5/58 [00:00<00:06,  7.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.66 GB):   9%|▊         | 5/58 [00:00<00:06,  7.92it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.66 GB):  10%|█         | 6/58 [00:00<00:06,  8.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.65 GB):  10%|█         | 6/58 [00:00<00:06,  8.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.65 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.65 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.60it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.65 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.65 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.64 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.64 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.36it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.64 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.63 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.63 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.63 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.63 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.52it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=42.62 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.62 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.15 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=40.15 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.72it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=40.15 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.15 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=37.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=37.36 GB):  33%|███▎      | 19/58 [00:01<00:03, 12.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=37.36 GB):  33%|███▎      | 19/58 [00:01<00:03, 12.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=37.34 GB):  33%|███▎      | 19/58 [00:01<00:03, 12.07it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=37.34 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.48it/s]Capturing num tokens (num_tokens=960 avail_mem=37.35 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.48it/s] Capturing num tokens (num_tokens=896 avail_mem=37.35 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.48it/s]Capturing num tokens (num_tokens=896 avail_mem=37.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.71it/s]Capturing num tokens (num_tokens=832 avail_mem=37.34 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.71it/s]Capturing num tokens (num_tokens=768 avail_mem=37.34 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.71it/s]

    Capturing num tokens (num_tokens=768 avail_mem=37.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.81it/s]Capturing num tokens (num_tokens=704 avail_mem=37.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.81it/s]Capturing num tokens (num_tokens=640 avail_mem=37.33 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.81it/s]Capturing num tokens (num_tokens=576 avail_mem=37.33 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.81it/s]Capturing num tokens (num_tokens=576 avail_mem=37.33 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.33it/s]Capturing num tokens (num_tokens=512 avail_mem=37.32 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.33it/s]Capturing num tokens (num_tokens=480 avail_mem=37.33 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.33it/s]

    Capturing num tokens (num_tokens=448 avail_mem=37.33 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.33it/s]Capturing num tokens (num_tokens=448 avail_mem=37.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=416 avail_mem=37.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=384 avail_mem=37.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=352 avail_mem=37.32 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=352 avail_mem=37.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.39it/s]Capturing num tokens (num_tokens=320 avail_mem=37.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.39it/s]

    Capturing num tokens (num_tokens=288 avail_mem=37.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.39it/s]Capturing num tokens (num_tokens=256 avail_mem=37.31 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.39it/s]Capturing num tokens (num_tokens=256 avail_mem=37.31 GB):  64%|██████▍   | 37/58 [00:02<00:01, 20.10it/s]Capturing num tokens (num_tokens=240 avail_mem=37.31 GB):  64%|██████▍   | 37/58 [00:02<00:01, 20.10it/s]Capturing num tokens (num_tokens=224 avail_mem=37.31 GB):  64%|██████▍   | 37/58 [00:02<00:01, 20.10it/s]Capturing num tokens (num_tokens=208 avail_mem=37.30 GB):  64%|██████▍   | 37/58 [00:02<00:01, 20.10it/s]

    Capturing num tokens (num_tokens=208 avail_mem=37.30 GB):  69%|██████▉   | 40/58 [00:02<00:00, 20.64it/s]Capturing num tokens (num_tokens=192 avail_mem=37.30 GB):  69%|██████▉   | 40/58 [00:02<00:00, 20.64it/s]Capturing num tokens (num_tokens=176 avail_mem=37.30 GB):  69%|██████▉   | 40/58 [00:02<00:00, 20.64it/s]Capturing num tokens (num_tokens=160 avail_mem=37.30 GB):  69%|██████▉   | 40/58 [00:03<00:00, 20.64it/s]Capturing num tokens (num_tokens=160 avail_mem=37.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.42it/s]Capturing num tokens (num_tokens=144 avail_mem=37.29 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.42it/s]Capturing num tokens (num_tokens=128 avail_mem=37.29 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.42it/s]

    Capturing num tokens (num_tokens=112 avail_mem=37.29 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.42it/s]Capturing num tokens (num_tokens=112 avail_mem=37.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 22.13it/s]Capturing num tokens (num_tokens=96 avail_mem=37.28 GB):  79%|███████▉  | 46/58 [00:03<00:00, 22.13it/s] Capturing num tokens (num_tokens=80 avail_mem=37.28 GB):  79%|███████▉  | 46/58 [00:03<00:00, 22.13it/s]Capturing num tokens (num_tokens=64 avail_mem=37.28 GB):  79%|███████▉  | 46/58 [00:03<00:00, 22.13it/s]Capturing num tokens (num_tokens=64 avail_mem=37.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 22.09it/s]Capturing num tokens (num_tokens=48 avail_mem=37.27 GB):  84%|████████▍ | 49/58 [00:03<00:00, 22.09it/s]

    Capturing num tokens (num_tokens=32 avail_mem=37.27 GB):  84%|████████▍ | 49/58 [00:03<00:00, 22.09it/s]Capturing num tokens (num_tokens=28 avail_mem=37.26 GB):  84%|████████▍ | 49/58 [00:03<00:00, 22.09it/s]Capturing num tokens (num_tokens=28 avail_mem=37.26 GB):  90%|████████▉ | 52/58 [00:03<00:00, 22.65it/s]Capturing num tokens (num_tokens=24 avail_mem=37.26 GB):  90%|████████▉ | 52/58 [00:03<00:00, 22.65it/s]Capturing num tokens (num_tokens=20 avail_mem=37.26 GB):  90%|████████▉ | 52/58 [00:03<00:00, 22.65it/s]Capturing num tokens (num_tokens=16 avail_mem=37.26 GB):  90%|████████▉ | 52/58 [00:03<00:00, 22.65it/s]

    Capturing num tokens (num_tokens=16 avail_mem=37.26 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.10it/s]Capturing num tokens (num_tokens=12 avail_mem=37.25 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.10it/s]Capturing num tokens (num_tokens=8 avail_mem=37.25 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.10it/s] Capturing num tokens (num_tokens=4 avail_mem=37.25 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.10it/s]Capturing num tokens (num_tokens=4 avail_mem=37.25 GB): 100%|██████████| 58/58 [00:03<00:00, 23.17it/s]Capturing num tokens (num_tokens=4 avail_mem=37.25 GB): 100%|██████████| 58/58 [00:03<00:00, 15.72it/s]


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
    Generated text:  Alice, 15 years old, and I'm a bright, curious, and empathetic person. I am interested in learning about the world, people, and science. I've read many books, watched lots of TV shows, and taken many classes online. I have a tendency to ask questions that others might have to answer in order to understand something, and I'm interested in learning more about different subjects from a broad perspective.
    
    I'm looking for a hobby or activity that I can do on my own that will help me develop critical thinking skills and broaden my knowledge in various fields, such as history, science, literature, and technology
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by a vice president. If there are 4 vice presidents and the president is considered a "majority party member," what is the probability that a randomly chosen person from the vice presidential nominations will be a "majority party member"? To determine the probability that a randomly chosen person from the vice presidential nominations will be a "majority party member," we need to follow these steps:
    
    1. **Identify the total number of vice presidents**: The problem states that there are 4 vice presidents.
    2. **Identify the number of "majority party members":** The problem explicitly states that the president is a "majority
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. Paris
    
    巴黎是法国的首都，位于法国南部，巴黎是法国的首都，位于法国南部，巴黎是法国的首都。选择B。 解析：句中形容词性短语“首都”和“南部”是一对并列短语，因此，答案为B。
    
    【判断题】简要说明领导决策过程的四个阶段。
    
    正确答案： （1）分析问题和选择目标。 （2）收集信息，分析问题。 （3）评估选项和确定方案。 （4）提出建议。
    
    636.中共十八届四中全会强调，行政机关必须坚持
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of individuals. It can be used for good or for bad. If you want to know whether you can use AI in your work, then you must ask yourself some questions. Do you want to create or use AI? Do you want to use AI for good or for bad? Are you interested in the specific applications of AI? Do you want to know about the ethical considerations? Are you concerned with the future of AI?
    For example, when deciding if you want to use AI in your work, you may want to consider whether it can help improve efficiency, reduce costs, or enhance productivity. If you want to use AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about yourself]. I'm always looking for new opportunities and I'm always eager to learn new things. What do you do for a living? I'm a [insert a short, positive, enthusiastic statement about your job]. I'm always looking for new opportunities and I'm always eager to learn new things. What do you enjoy doing? I enjoy [insert a short, positive, enthusiastic statement about what
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is the capital of France and the largest city in the European Union. The city is known for its diverse population, rich history, and vibrant culture. It is a major economic and political center of France and plays a significant role in the country's economy and politics. Paris is also home to many international organizations and institutions, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI systems become more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, but there is a growing trend towards using AI to assist in diagnosis, treatment, and patient care.
    
    4. Greater use of AI in
    


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
    Generated text:  [Name] and I am a [career field] professional with a passion for [specific field]. I have a keen eye for detail, a passion for learning, and a natural ability to collaborate with others. I have a strong work ethic and am always looking for new ways to improve my skills. I enjoy staying organized and staying on top of my game. I am a quick learner and enjoy exploring new ideas and technologies. I am available for any type of consulting or research work. Thank you for asking. Is there anything specific you would like to share about yourself? I am passionate about [specific field] and am always looking for new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To elaborate, Paris is the largest city in France and the third-largest city in Europe. It is situated on the right bank of the Seine River and is the capital of the Île de France and Paris métropole. Paris is known for its rich history, architecture, and culture, and is a major center for art, literature, and business. It is also a significant economic and political center in France, hosting major institutions such as the French Academy of Sciences and the French National Library. Paris is a UNESCO City of Exceptional Cultural and Political Importance and a major financial and economic hub for France. According to the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  constantly evolving and unpredictable, but here are some possible trends that may be seen:
    
    1. Improved AI ethics: As AI becomes more advanced, it will become more important to consider ethical considerations. Governments and organizations will need to develop frameworks and regulations to guide the development of AI systems that are beneficial for society.
    
    2. AI will be integrated into human life: AI is already playing an increasingly important role in our daily lives, from self-driving cars to chatbots for customer service. It is likely that we will see a continued integration of AI into human life, from personal assistants to virtual assistants.
    
    3. AI will be more data-driven: With


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

    character

    ]

     at

     heart

    .

     I

    'm

     a

     [

    character

    ]

     who

    's

     always

     been

     drawn

     to

     the

     world

     of

     [

    field

    ,

     interest

    ].

     I

    'm

     enthusiastic

     and

     passionate

     about

     [

    mention

     a

     hobby

     or

     activity

    ].

     I

     thrive

     on

     [

    mention

     a

     hobby

     or

     activity

    ]

     and

     I

    'm

     always

     looking

     for

     new

     adventures

     and

     challenges

    .

     I

    'm

     very

     [

    positive

    ,

     energetic

    ,

     or

     outgoing

    ]

     and

     always

     have

     a

     smile

     on

     my

     face

    .

     What

     kind

     of

     person

     am

     I

    ?

     Can

     you

     describe

     me

     in

     more

     detail

    ?

     What

     kind

     of

     world

     are

     you

     drawn

     to

    ?

     What

     kind

     of

     adventure

     do

     you

     always

     look

     for

    ?

     What

     kind

     of

     challenges

     are

     you

     looking

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     bustling

     met

    ropolis

     renowned

     for

     its

     iconic

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

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

     also

     known

     as

     "

    The

     City

     of

     Light

    ,"

     symbol

    izing

     the

     city

    's

     light

     and

     creativity

    ,

     and

     is

     home

     to

     over

     

    1

    0

     million

     people

    .

     The

     city

     is

     a

     major

     center

     for

     the

     arts

    ,

     cuisine

    ,

     fashion

    ,

     and

     many

     other

     cultural

     and

     social

     activities

    .

     Its

     annual

     "

    F

    ête

     de

     la

     Saint

    -E

    st

    è

    me

    "

     is

     a

     major

     celebration

     of

     French

     culture

    ,

     featuring

     par

    ades

    ,

     concerts

    ,

     and

     fireworks

    .

     The

     French

     language

     and

     cuisine

     are

     also

     highly

     influential

     in

     the

     city

    ,

     with

     Paris

    ian

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     and

     we

     are

     seeing

     significant

     advances

     in

     several

     key

     areas

    .

     Here

     are

     some

     of

     the

     most

     promising

     trends

     in

     AI

     that

     are

     likely

     to

     shape

     the

     technology

     in

     the

     coming

     years

    :
    


    1

    .

     Artificial

     Intelligence

     in

     Healthcare

    :

     AI

     is

     already

     transforming

     healthcare

     by

     improving

     patient

     care

     and

     diagnostics

    .

     AI

     algorithms

     can

     analyze

     medical

     records

    ,

     identify

     patterns

    ,

     and

     predict

     disease

     progression

    ,

     making

     it

     easier

     for

     doctors

     to

     make

     informed

     decisions

    .

     AI

     is

     also

     being

     used

     to

     develop

     new

     treatments

     for

     diseases

     such

     as

     cancer

    ,

     neuro

    de

    gener

    ative

     disorders

    ,

     and

     traumatic

     injuries

    .
    


    2

    .

     Self

    -driving

     cars

    :

     Self

    -driving

     cars

     are

     on

     the

     horizon

    ,

     and

     their

     development

     is

     driven

     by

     advancements

     in

     AI

    .

    



```python
llm.shutdown()
```
