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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.75it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.75it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.75it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:12,  4.06it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:12,  4.06it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:12,  4.06it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:12,  4.06it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.47it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.47it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  6.47it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  6.47it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.20it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.20it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  9.20it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  9.20it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  9.20it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:02, 13.33it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:02, 13.33it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:02, 13.33it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:02, 13.33it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:02, 13.33it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 21.27it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 24.75it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 24.75it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 24.75it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 24.75it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 24.75it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 24.75it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 29.41it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 32.69it/s]

    Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 35.27it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 36.70it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 36.70it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 36.70it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 36.70it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 36.70it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 36.70it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 39.52it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 39.52it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 39.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.82 GB):   2%|▏         | 1/58 [00:00<00:10,  5.63it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.81 GB):   2%|▏         | 1/58 [00:00<00:10,  5.63it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.81 GB):   3%|▎         | 2/58 [00:00<00:09,  6.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.02 GB):   3%|▎         | 2/58 [00:00<00:09,  6.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.02 GB):   5%|▌         | 3/58 [00:00<00:08,  6.49it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.01 GB):   5%|▌         | 3/58 [00:00<00:08,  6.49it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.01 GB):   7%|▋         | 4/58 [00:00<00:07,  7.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.83 GB):   7%|▋         | 4/58 [00:00<00:07,  7.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.83 GB):   9%|▊         | 5/58 [00:00<00:08,  6.33it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.84 GB):   9%|▊         | 5/58 [00:00<00:08,  6.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.84 GB):  10%|█         | 6/58 [00:00<00:09,  5.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.85 GB):  10%|█         | 6/58 [00:00<00:09,  5.65it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=55.85 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.85 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.85 GB):  14%|█▍        | 8/58 [00:01<00:07,  7.04it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.97 GB):  14%|█▍        | 8/58 [00:01<00:07,  7.04it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.96 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.96 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.95 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.74it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=55.94 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.94 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.94 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.93 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.41it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=55.93 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.89 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.88 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.88 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.87 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.62it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=55.87 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.87 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.89 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.89it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=55.87 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.89it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=55.87 GB):  36%|███▌      | 21/58 [00:02<00:04,  7.61it/s]Capturing num tokens (num_tokens=960 avail_mem=55.89 GB):  36%|███▌      | 21/58 [00:02<00:04,  7.61it/s] Capturing num tokens (num_tokens=896 avail_mem=55.89 GB):  36%|███▌      | 21/58 [00:02<00:04,  7.61it/s]Capturing num tokens (num_tokens=896 avail_mem=55.89 GB):  40%|███▉      | 23/58 [00:02<00:03,  8.75it/s]Capturing num tokens (num_tokens=832 avail_mem=55.88 GB):  40%|███▉      | 23/58 [00:02<00:03,  8.75it/s]

    Capturing num tokens (num_tokens=768 avail_mem=55.87 GB):  40%|███▉      | 23/58 [00:02<00:03,  8.75it/s]Capturing num tokens (num_tokens=768 avail_mem=55.87 GB):  43%|████▎     | 25/58 [00:02<00:03,  9.78it/s]Capturing num tokens (num_tokens=704 avail_mem=55.87 GB):  43%|████▎     | 25/58 [00:02<00:03,  9.78it/s]Capturing num tokens (num_tokens=640 avail_mem=55.85 GB):  43%|████▎     | 25/58 [00:03<00:03,  9.78it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.85 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.62it/s]Capturing num tokens (num_tokens=576 avail_mem=55.87 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.62it/s]Capturing num tokens (num_tokens=512 avail_mem=55.85 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.62it/s]Capturing num tokens (num_tokens=512 avail_mem=55.85 GB):  50%|█████     | 29/58 [00:03<00:02, 11.31it/s]Capturing num tokens (num_tokens=480 avail_mem=55.86 GB):  50%|█████     | 29/58 [00:03<00:02, 11.31it/s]

    Capturing num tokens (num_tokens=448 avail_mem=55.86 GB):  50%|█████     | 29/58 [00:03<00:02, 11.31it/s]Capturing num tokens (num_tokens=448 avail_mem=55.86 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.84it/s]Capturing num tokens (num_tokens=416 avail_mem=55.85 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.84it/s]Capturing num tokens (num_tokens=384 avail_mem=55.85 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.84it/s]

    Capturing num tokens (num_tokens=384 avail_mem=55.85 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.31it/s]Capturing num tokens (num_tokens=352 avail_mem=55.84 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.31it/s]Capturing num tokens (num_tokens=320 avail_mem=55.83 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.31it/s]Capturing num tokens (num_tokens=320 avail_mem=55.83 GB):  60%|██████    | 35/58 [00:03<00:01, 12.49it/s]Capturing num tokens (num_tokens=288 avail_mem=55.82 GB):  60%|██████    | 35/58 [00:03<00:01, 12.49it/s]

    Capturing num tokens (num_tokens=256 avail_mem=55.82 GB):  60%|██████    | 35/58 [00:03<00:01, 12.49it/s]Capturing num tokens (num_tokens=256 avail_mem=55.82 GB):  64%|██████▍   | 37/58 [00:03<00:01, 12.83it/s]Capturing num tokens (num_tokens=240 avail_mem=55.81 GB):  64%|██████▍   | 37/58 [00:03<00:01, 12.83it/s]Capturing num tokens (num_tokens=224 avail_mem=55.81 GB):  64%|██████▍   | 37/58 [00:03<00:01, 12.83it/s]

    Capturing num tokens (num_tokens=224 avail_mem=55.81 GB):  67%|██████▋   | 39/58 [00:04<00:01, 12.97it/s]Capturing num tokens (num_tokens=208 avail_mem=55.80 GB):  67%|██████▋   | 39/58 [00:04<00:01, 12.97it/s]Capturing num tokens (num_tokens=192 avail_mem=55.80 GB):  67%|██████▋   | 39/58 [00:04<00:01, 12.97it/s]Capturing num tokens (num_tokens=192 avail_mem=55.80 GB):  71%|███████   | 41/58 [00:04<00:01, 13.07it/s]Capturing num tokens (num_tokens=176 avail_mem=55.79 GB):  71%|███████   | 41/58 [00:04<00:01, 13.07it/s]

    Capturing num tokens (num_tokens=160 avail_mem=55.79 GB):  71%|███████   | 41/58 [00:04<00:01, 13.07it/s]Capturing num tokens (num_tokens=160 avail_mem=55.79 GB):  74%|███████▍  | 43/58 [00:04<00:01, 13.19it/s]Capturing num tokens (num_tokens=144 avail_mem=55.78 GB):  74%|███████▍  | 43/58 [00:04<00:01, 13.19it/s]Capturing num tokens (num_tokens=128 avail_mem=55.77 GB):  74%|███████▍  | 43/58 [00:04<00:01, 13.19it/s]

    Capturing num tokens (num_tokens=128 avail_mem=55.77 GB):  78%|███████▊  | 45/58 [00:04<00:00, 13.22it/s]Capturing num tokens (num_tokens=112 avail_mem=55.77 GB):  78%|███████▊  | 45/58 [00:04<00:00, 13.22it/s]Capturing num tokens (num_tokens=96 avail_mem=55.76 GB):  78%|███████▊  | 45/58 [00:04<00:00, 13.22it/s] Capturing num tokens (num_tokens=96 avail_mem=55.76 GB):  81%|████████  | 47/58 [00:04<00:00, 13.26it/s]Capturing num tokens (num_tokens=80 avail_mem=55.75 GB):  81%|████████  | 47/58 [00:04<00:00, 13.26it/s]

    Capturing num tokens (num_tokens=64 avail_mem=55.74 GB):  81%|████████  | 47/58 [00:04<00:00, 13.26it/s]Capturing num tokens (num_tokens=64 avail_mem=55.74 GB):  84%|████████▍ | 49/58 [00:04<00:00, 13.22it/s]Capturing num tokens (num_tokens=48 avail_mem=55.74 GB):  84%|████████▍ | 49/58 [00:04<00:00, 13.22it/s]Capturing num tokens (num_tokens=32 avail_mem=55.73 GB):  84%|████████▍ | 49/58 [00:04<00:00, 13.22it/s]

    Capturing num tokens (num_tokens=32 avail_mem=55.73 GB):  88%|████████▊ | 51/58 [00:04<00:00, 13.07it/s]Capturing num tokens (num_tokens=28 avail_mem=55.73 GB):  88%|████████▊ | 51/58 [00:04<00:00, 13.07it/s]Capturing num tokens (num_tokens=24 avail_mem=55.72 GB):  88%|████████▊ | 51/58 [00:04<00:00, 13.07it/s]Capturing num tokens (num_tokens=24 avail_mem=55.72 GB):  91%|█████████▏| 53/58 [00:05<00:00, 12.96it/s]Capturing num tokens (num_tokens=20 avail_mem=55.71 GB):  91%|█████████▏| 53/58 [00:05<00:00, 12.96it/s]

    Capturing num tokens (num_tokens=16 avail_mem=55.71 GB):  91%|█████████▏| 53/58 [00:05<00:00, 12.96it/s]Capturing num tokens (num_tokens=16 avail_mem=55.71 GB):  95%|█████████▍| 55/58 [00:05<00:00, 13.38it/s]Capturing num tokens (num_tokens=12 avail_mem=55.70 GB):  95%|█████████▍| 55/58 [00:05<00:00, 13.38it/s]Capturing num tokens (num_tokens=8 avail_mem=55.69 GB):  95%|█████████▍| 55/58 [00:05<00:00, 13.38it/s] Capturing num tokens (num_tokens=8 avail_mem=55.69 GB):  98%|█████████▊| 57/58 [00:05<00:00, 14.00it/s]Capturing num tokens (num_tokens=4 avail_mem=55.69 GB):  98%|█████████▊| 57/58 [00:05<00:00, 14.00it/s]

    Capturing num tokens (num_tokens=4 avail_mem=55.69 GB): 100%|██████████| 58/58 [00:05<00:00, 10.74it/s]


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
    Generated text:  Ben and I am a registered sex offender.
    I am now pursuing a law degree from a prestigious university.
    I do not want anyone to be harmed.
    I do not want to commit any crimes.
    I will not be a victim.
    I will never be a victim of any crime.
    What should I do?
    What is the most important aspect of this question?
    What should I tell the people who know me?
    What are the potential consequences of being known to someone who knows me?
    What are the consequences of a sex offender being known to someone?
    What should I do if I suspect someone is following me? What should I do if someone stops calling
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is a head of state. His job is to make decisions for the country. 
    
    The president usually comes to work at the White House. It's a very big building. The president can have more than one meeting at a time. It's very important to have the president at the White House. 
    
    The president is not here to work. He comes to work in a special car. The car has a special window that he can open and close. It has a power switch. He can open and close the window himself. He doesn't need to go to work to open the window.
    
    The car doesn
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which region?
    A. South
    B. Central
    C. North
    D. East
    Answer:
    B
    
    Which of the following options refers to the space of various celestial objects in the universe?
    A. Space
    B. Earth
    C. Sky
    D. Sky
    Answer:
    A
    
    Please select the word that best fits in the blank: I _____ an apple. (A) ate (B) have (C) have had (D) have eaten.
    Answer:
    D
    
    Please select the most appropriate word to complete the sentence: The local government has taken __________ measures to improve the environment.
    A. an
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of a group of future leaders: the next generation of scientists, engineers and technology entrepreneurs. Here are some of their ideas for how to ensure that the next generation of scientists and engineers have access to the most advanced AI technology and infrastructure. We’ve tried to prioritize questions such as how best to create a diverse future of AI, how to build accountability and oversight, and how to ensure the contributions of women and people of color to AI research and development.
    Understanding the role of women and people of color in AI research is of critical importance, as research is a female-dominated field and it is often women and people of color who make


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm always looking for new challenges and opportunities to grow and learn. What are your hobbies or interests? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to grow and learn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its fashion, art, and cuisine, and is a popular destination for tourists and locals alike. It is also home to the French Parliament and the French National Library. Paris is a vibrant and dynamic city with a rich cultural heritage and is a major tourist destination. It is also known for its fashion, art, and cuisine. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to a more human-like experience for users.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, but it has the potential to revolutionize the field. AI-powered diagnostic tools could be used to identify diseases earlier, improve treatment outcomes, and reduce costs.
    
    3. Increased use of AI in transportation: AI is already being used in transportation to improve
    


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
    Generated text:  [Name], and I'm a [job title] at [company name]. I'm here today to contribute to [job title] at [company name]. What can you tell me about yourself? [Name] is a [job title] at [company name], specializing in [responsibilities]. I enjoy [reason for passion] and [other reasons for passion]. What do you enjoy doing outside of work? I love reading, hiking, and playing music. How do you stay motivated and motivated? I like to set achievable goals for myself and celebrate my successes along the way. And what motivates you to keep learning and growing?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, and its rich culinary tradition with dishes like foie gras, escargot, and ciabatta. Paris is a cultural and historical center that plays a crucial role in France's economy and politics. 
    
    Please choose the correct option to complete the sentence: "Paris is a city known for its iconic landmarks, what is the most famous of which?"
    
    A) Eiffel Tower
    B) Louvre Museum
    C) Notre-Dame Cathedral
    D) Montmartre
    
    C) Notre-Dame Cathedral is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting, with potential applications in areas like healthcare, finance, transportation, and security. Here are some potential future trends in AI:
    
    1. Increased Use of AI for Autonomous Vehicles: AI is increasingly being used in autonomous vehicles, allowing self-driving cars to navigate roads and avoid collisions. This will lead to safer and more efficient transportation systems.
    
    2. AI in Healthcare: AI can be used to analyze medical images, detect disease, and improve medical diagnoses. This will lead to more accurate and timely healthcare services.
    
    3. AI in Finance: AI can be used to analyze financial data, identify patterns, and make recommendations for investors. This will


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

     __

    ____________

     and

     I

     am

     a

    /an

     __

    ____________

    ___

     (

    cat

    ,

     dog

    ,

     fish

    ,

     etc

    .)

     who

     has

     the

     ability

     to

     __

    ____________

    _

    .
    


    Hi

     there

    !

     I

    'm

     __

    ____________

    ,

     a

    /an

     __

    ____________

    _

     (

    cat

    ,

     dog

    ,

     fish

    ,

     etc

    .).

     I

     have

     the

     ability

     to

     __

    ____________

    _

     and

     I

    'm

     always

     eager

     to

     learn

     about

     new

     things

     and

     meet

     new

     people

    .

     What

    's

     your

     name

    ,

     and

     what

     can

     you

     do

     that

     you

    're

     interested

     in

     learning

     more

     about

    ?

     I

     hope

     to

     connect

     with

     you

     soon

    !

     

    🐍

    ást

    ánt

    í

    !

     

    🌍

    ást

    ánt

    í

    !

     

    🐍

    ást

    ánt

    í

    !

     

    🌍

    ást

    ánt

    í

    !

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     the

     city

     of

     light

     and

     art

    ,

     is

     home

     to

     iconic

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

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     rich

     cultural

     heritage

     and

     diverse

     neighborhoods

    .

     The

     city

     is

     known

     for

     its

     bustling

     streets

    ,

     vibrant

     nightlife

    ,

     and

     numerous

     museums

     and

     attractions

    ,

     making

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     Despite

     its

     size

    ,

     Paris

     has

     a

     distinct

    ,

     charming

     character

    ,

     known

     for

     its

     romantic

    ism

     and

     grand

    eur

    ,

     and

     has

     become

     a

     major

     cultural

     hub

     in

     Europe

    .

     Paris

     is

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

     romantic

     atmosphere

    ,

     and

     has

     played

     an

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     there

     are

     several

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

     potential

     trends

     that

     experts

     predict

    :
    


    1

    .

     Advanced

     Natural

     Language

     Processing

     (

    N

    LP

    )

     -

     As

     AI

     becomes

     more

     capable

     of

     understanding

     and

     processing

     natural

     language

    ,

     it

     could

     lead

     to

     the

     development

     of

     advanced

     N

    LP

     technologies

     that

     can

     understand

     and

     generate

     human

    -like

     language

    .

     This

     could

     lead

     to

     more

     sophisticated

     chat

    bots

    ,

     language

     translation

     systems

    ,

     and

     natural

     language

     generation

     tools

    .
    


    2

    .

     Rob

    otic

     Intelligence

     -

     The

     integration

     of

     AI

     and

     robotics

     could

     lead

     to

     the

     development

     of

     machines

     that

     are

     more

     autonomous

     and

     can

     perform

     a

     wide

     range

     of

     tasks

     autonom

    ously

    .

     This

     could

     lead

     to

     increased

     productivity

     and

     efficiency

    ,

    



```python
llm.shutdown()
```
