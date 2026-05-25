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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.65it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:57,  4.16s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:57,  4.16s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:57,  4.16s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.35it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.35it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.35it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.35it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.35it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.23it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.23it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.23it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.23it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:03, 11.23it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 15.13it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 15.13it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:02, 15.13it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 15.13it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 15.13it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 15.13it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 20.68it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 25.74it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 25.74it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 25.74it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 25.74it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 25.74it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 25.74it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 25.74it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 31.68it/s]

    Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 37.36it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 37.36it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 37.36it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 37.36it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 37.36it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 37.36it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 37.36it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 42.38it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 42.38it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 42.38it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 42.38it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 42.38it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 42.38it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 42.38it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 46.67it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 46.67it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 46.67it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 46.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.82 GB):   2%|▏         | 1/58 [00:00<00:07,  7.23it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.78 GB):   2%|▏         | 1/58 [00:00<00:07,  7.23it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.78 GB):   3%|▎         | 2/58 [00:00<00:07,  7.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.78 GB):   3%|▎         | 2/58 [00:00<00:07,  7.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.78 GB):   5%|▌         | 3/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.78 GB):   5%|▌         | 3/58 [00:00<00:07,  7.47it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.78 GB):   7%|▋         | 4/58 [00:00<00:07,  7.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.78 GB):   7%|▋         | 4/58 [00:00<00:07,  7.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.78 GB):   9%|▊         | 5/58 [00:00<00:06,  7.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.77 GB):   9%|▊         | 5/58 [00:00<00:06,  7.95it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=38.77 GB):  10%|█         | 6/58 [00:00<00:06,  8.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.76 GB):  10%|█         | 6/58 [00:00<00:06,  8.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.76 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.76 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.67it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=38.76 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.76 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.76 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.72 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.07it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=38.72 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.71 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.71 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.71 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.71 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.59it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=38.71 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.71 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.70 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.70 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.70 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.41it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.70 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.70 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.69 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.69 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.69 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.69 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.79it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.67 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.67 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.70it/s]Capturing num tokens (num_tokens=960 avail_mem=38.68 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.70it/s] Capturing num tokens (num_tokens=896 avail_mem=38.68 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=896 avail_mem=38.68 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.84it/s]Capturing num tokens (num_tokens=832 avail_mem=38.68 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.84it/s]

    Capturing num tokens (num_tokens=768 avail_mem=38.67 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.84it/s]Capturing num tokens (num_tokens=768 avail_mem=38.67 GB):  43%|████▎     | 25/58 [00:02<00:01, 16.83it/s]Capturing num tokens (num_tokens=704 avail_mem=38.67 GB):  43%|████▎     | 25/58 [00:02<00:01, 16.83it/s]Capturing num tokens (num_tokens=640 avail_mem=38.67 GB):  43%|████▎     | 25/58 [00:02<00:01, 16.83it/s]Capturing num tokens (num_tokens=576 avail_mem=38.67 GB):  43%|████▎     | 25/58 [00:02<00:01, 16.83it/s]

    Capturing num tokens (num_tokens=576 avail_mem=38.67 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.94it/s]Capturing num tokens (num_tokens=512 avail_mem=38.65 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.94it/s]Capturing num tokens (num_tokens=480 avail_mem=38.67 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.94it/s]Capturing num tokens (num_tokens=480 avail_mem=38.67 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.41it/s]Capturing num tokens (num_tokens=448 avail_mem=38.66 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.41it/s]Capturing num tokens (num_tokens=416 avail_mem=38.66 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.41it/s]Capturing num tokens (num_tokens=384 avail_mem=38.66 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.41it/s]

    Capturing num tokens (num_tokens=384 avail_mem=38.66 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.29it/s]Capturing num tokens (num_tokens=352 avail_mem=38.65 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.29it/s]Capturing num tokens (num_tokens=320 avail_mem=38.65 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.29it/s]Capturing num tokens (num_tokens=288 avail_mem=38.65 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.29it/s]Capturing num tokens (num_tokens=288 avail_mem=38.65 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.56it/s]Capturing num tokens (num_tokens=256 avail_mem=38.65 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.56it/s]Capturing num tokens (num_tokens=240 avail_mem=38.64 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.56it/s]

    Capturing num tokens (num_tokens=224 avail_mem=38.64 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.56it/s]Capturing num tokens (num_tokens=224 avail_mem=38.64 GB):  67%|██████▋   | 39/58 [00:02<00:00, 19.94it/s]Capturing num tokens (num_tokens=208 avail_mem=38.63 GB):  67%|██████▋   | 39/58 [00:02<00:00, 19.94it/s]Capturing num tokens (num_tokens=192 avail_mem=38.63 GB):  67%|██████▋   | 39/58 [00:02<00:00, 19.94it/s]

    Capturing num tokens (num_tokens=192 avail_mem=38.63 GB):  71%|███████   | 41/58 [00:03<00:00, 18.79it/s]Capturing num tokens (num_tokens=176 avail_mem=38.63 GB):  71%|███████   | 41/58 [00:03<00:00, 18.79it/s]Capturing num tokens (num_tokens=160 avail_mem=38.63 GB):  71%|███████   | 41/58 [00:03<00:00, 18.79it/s]Capturing num tokens (num_tokens=160 avail_mem=38.63 GB):  74%|███████▍  | 43/58 [00:03<00:00, 17.53it/s]Capturing num tokens (num_tokens=144 avail_mem=38.62 GB):  74%|███████▍  | 43/58 [00:03<00:00, 17.53it/s]Capturing num tokens (num_tokens=128 avail_mem=38.62 GB):  74%|███████▍  | 43/58 [00:03<00:00, 17.53it/s]

    Capturing num tokens (num_tokens=128 avail_mem=38.62 GB):  78%|███████▊  | 45/58 [00:03<00:00, 16.78it/s]Capturing num tokens (num_tokens=112 avail_mem=38.62 GB):  78%|███████▊  | 45/58 [00:03<00:00, 16.78it/s]Capturing num tokens (num_tokens=96 avail_mem=38.62 GB):  78%|███████▊  | 45/58 [00:03<00:00, 16.78it/s] Capturing num tokens (num_tokens=96 avail_mem=38.62 GB):  81%|████████  | 47/58 [00:03<00:00, 16.34it/s]Capturing num tokens (num_tokens=80 avail_mem=38.61 GB):  81%|████████  | 47/58 [00:03<00:00, 16.34it/s]Capturing num tokens (num_tokens=64 avail_mem=38.61 GB):  81%|████████  | 47/58 [00:03<00:00, 16.34it/s]

    Capturing num tokens (num_tokens=64 avail_mem=38.61 GB):  84%|████████▍ | 49/58 [00:03<00:00, 16.09it/s]Capturing num tokens (num_tokens=48 avail_mem=38.60 GB):  84%|████████▍ | 49/58 [00:03<00:00, 16.09it/s]Capturing num tokens (num_tokens=32 avail_mem=38.60 GB):  84%|████████▍ | 49/58 [00:03<00:00, 16.09it/s]Capturing num tokens (num_tokens=32 avail_mem=38.60 GB):  88%|████████▊ | 51/58 [00:03<00:00, 15.84it/s]Capturing num tokens (num_tokens=28 avail_mem=38.60 GB):  88%|████████▊ | 51/58 [00:03<00:00, 15.84it/s]Capturing num tokens (num_tokens=24 avail_mem=38.59 GB):  88%|████████▊ | 51/58 [00:03<00:00, 15.84it/s]

    Capturing num tokens (num_tokens=24 avail_mem=38.59 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.67it/s]Capturing num tokens (num_tokens=20 avail_mem=38.59 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.67it/s]Capturing num tokens (num_tokens=16 avail_mem=38.59 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.67it/s]Capturing num tokens (num_tokens=12 avail_mem=38.58 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.67it/s]Capturing num tokens (num_tokens=12 avail_mem=38.58 GB):  97%|█████████▋| 56/58 [00:03<00:00, 17.71it/s]Capturing num tokens (num_tokens=8 avail_mem=38.58 GB):  97%|█████████▋| 56/58 [00:03<00:00, 17.71it/s] Capturing num tokens (num_tokens=4 avail_mem=38.58 GB):  97%|█████████▋| 56/58 [00:03<00:00, 17.71it/s]

    Capturing num tokens (num_tokens=4 avail_mem=38.58 GB): 100%|██████████| 58/58 [00:04<00:00, 14.43it/s]


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
    Generated text:  Emma, and I'm 13 years old. I have a huge imagination and my favorite hobby is collecting stamps. I love to travel and learn about different cultures. I have a crush on a boy and we often play pretend together. I also enjoy playing video games and reading books. What are your hobbies and interests? As an AI language model, I don't have hobbies or interests like humans do, but I can help you explore various topics and provide information on them. What would you like to know? I'm happy to help answer any questions you might have! Let's chat about anything you'd like! 😊
    
    I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a public office, it is a position that is incumbent upon the public, and it is an office that is for the public good. The president is the head of the executive branch of the federal government. It's the head of the federal government. The president is the commander-in-chief of the armed forces of the United States. It is a three-day event. It is a tremendous event. It takes place on the second Thursday of January. It is an important event. I hope we can have it on a Tuesday, because the New Year's celebrations are usually on New Year's Eve. There are a number of things that are different
    ===============================
    Prompt: The capital of France is
    Generated text:  in which direction? Paris is the capital city of France. France is a country in Europe, and its capital city is Paris. The exact location of Paris is in the northeast part of France, about 45 kilometers (28 miles) east of the Mediterranean Sea. If you were to travel from the north end of the Mediterranean to the south end, you would end up in Paris. The capital is usually located in the center of a country, so Paris is the most central city in France. The capital city is surrounded by many other cities and towns, which makes it a very important and bustling city in France. Paris has a
    ===============================
    Prompt: The future of AI is
    Generated text:  many things, but one of the most promising is the ability to make computer vision work like the eyes and ears of a human being.
    Human vision allows us to see things with our eyes, which gives us the ability to perceive light and color. The eyes are responsible for information acquisition and processing. They receive visual information and send it to the brain for processing, which helps us to perceive our environment in a three-dimensional manner.
    Computer vision, on the other hand, is the ability of a computer to perceive its surroundings by imaging or interpreting the visual signals. Most computer vision systems use cameras to receive visual information. Camera sensors are sensitive to light


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or achievement]. I'm [age] years old, and I'm [gender] and [race]. I'm [occupation] and I'm [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or achievement]. I'm [age] years old, and I'm [gender] and [race]. I'm [occupation] and I'm [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is located in the south of France and is the largest city in the country. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and art. Paris is a popular tourist destination and is home to many world-renowned museums, theaters, and landmarks. It is a cultural and intellectual center of France and a major economic hub. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater reliance on data: AI will become more data-driven, with more data being collected and analyzed to improve its performance. This could lead to more efficient and effective AI systems that can learn from data and make better predictions and decisions.
    
    3. Increased ethical considerations: As AI becomes more
    


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
    Generated text:  [Name] and I'm a [Age] year old [Occupation] or [Relationship Status]! I'm confident, organized, and skilled at [Skill or hobby], and always strive to be the best I can be. Whether I'm in the office or on the job, I'm always there to help you succeed. I'm someone who thrives on teamwork and always strive to make a positive impact on the world. Looking forward to seeing you! 🌟✨✨ #MeetMe #Friendly #OpenToTalk
    #HelloAndHello #FriendlyChat #FriendlyAndOpen #ProfessionalPerson #HelloWorld #FriendlyChat
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its history, culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. Paris is also known for its bustling urban environment, as well as its diverse population of more than 7 million people. Additionally, Paris is home to many notable and famous artists, including Picasso and Van Gogh, who are often celebrated for their work that emerged from the city. It's also famous for its fashion scene, including the iconic couture boutiques, and its annual Fête de la Musique, which attracts thousands of visitors each year to the city. Finally, the city is known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very bright and diverse. Here are some of the potential future trends:
    
    1. Autonomous vehicles: AI will continue to evolve and improve, making autonomous vehicles more practical and reliable. This could lead to a significant shift in the way we move around and interact with the world.
    
    2. Deep learning and machine learning: These are key areas of AI research, and will continue to develop rapidly. Deep learning will enable AI to recognize patterns and make more accurate predictions, while machine learning will enable AI to adapt and improve over time.
    
    3. Chatbots and virtual assistants: AI-powered chatbots and virtual assistants will become more sophisticated and widely available. These


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

    ],

     and

     I

     am

     a

     [

    insert

     profession

     or

     industry

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

    insert

     number

    ]

     years

    .

     Throughout

     that

     time

    ,

     I

     have

     hon

    ed

     my

     skills

     in

     [

    insert

     relevant

     skill

     or

     expertise

    ],

     and

     I

     am

     always

     striving

     to

     improve

     my

     abilities

    .

     I

     am

     a

     [

    insert

     personality

     trait

     or

     trait

    ],

     and

     I

     am

     always

     eager

     to

     learn

     and

     grow

    .

     What

    's

     your

     name

    ?

     What

    's

     your

     profession

    ?

     And

     what

    's

     your

     area

     of

     expertise

     or

     skill

    ?

     I

     look

     forward

     to

     our

     conversation

    !

     

    🤖

    ✨

     #

    self

    int

    roduction

     #

    inter

    personal

    communication

     #

    inter

    personal

    skills

     #

    personal

    development

     #

    professional

    development

     #

    career

    jour

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     “

    City

     of

     Light

    ,”

     which

     is

     renowned

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     Its

     annual

     Gross

     Domestic

     Product

     (

    G

    DP

    )

     is

     estimated

     to

     be

     around

     $

    2

    5

    0

     billion

    ,

     and

     it

     is

     a

     major

     financial

     and

     business

     center

    .

     Paris

     has

     been

     a

     cosm

    opolitan

     hub

     of

     the

     world

    ’s

     culture

    ,

     music

    ,

     and

     cuisine

     since

     its

     founding

     in

     the

     

    1

    2

    th

     century

    ,

     and

     it

     remains

     a

     cultural

     and

     political

     center

     in

     Europe

     and

     beyond

    .

     The

     city

     is

     a

     hub

     for

     arts

    ,

     literature

    ,

     and

     fashion

    ,

     and

     its

     iconic

     landmarks

    ,

     such

     as

     Notre

     Dame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     a

     range

     of

     advanced

     and

     exciting

     developments

    ,

     with

     many

     different

     areas

     of

     focus

    .

     Here

     are

     some

     of

     the

     most

     promising

     areas

    :
    


    1

    .

     Personal

    ized

     AI

    :

     One

     of

     the

     most

     exciting

     trends

     in

     AI

     is

     the

     development

     of

     AI

     that

     is

     able

     to

     tailor

     its

     responses

     and

     actions

     to

     the

     individual

     needs

     and

     preferences

     of

     each

     person

    .

     This

     could

     involve

     the

     use

     of

     machine

     learning

     to

     analyze

     data

     on

     individual

     behavior

     and

     preferences

    ,

     and

     then

     adjusting

     the

     AI

    's

     responses

     accordingly

    .

     This

     could

     lead

     to

     more

     effective

     and

     personalized

     services

     and

     products

    ,

     as

     well

     as

     potentially

     improved

     healthcare

     and

     education

     outcomes

    .
    


    2

    .

     Autonomous

     vehicles

    :

     As

     autonomous

     vehicles

     become

     more

     commonplace

    ,

     there

     will

     be

    



```python
llm.shutdown()
```
