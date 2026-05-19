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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:59,  1.09s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:59,  1.09s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:59,  1.09s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.80it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.80it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:29,  1.80it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:29,  1.80it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.46it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.46it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.46it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.46it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:14,  3.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.21it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.21it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  6.21it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  6.21it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:07,  6.21it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:04,  9.50it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:04,  9.50it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:04,  9.50it/s]

    Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:04,  9.50it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:04,  9.50it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 13.21it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 13.21it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 13.21it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 13.21it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:02, 13.21it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:02, 13.21it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 18.50it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 18.50it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 18.50it/s]

    Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 18.50it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:01, 18.50it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:01, 18.50it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 23.38it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 23.38it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 23.38it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 23.38it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.38it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 23.38it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 23.38it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]

    Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:00, 30.49it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 40.59it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 48.82it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 48.82it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 48.82it/s]

    Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 48.82it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 48.82it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 48.82it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 48.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.18 GB):   2%|▏         | 1/58 [00:00<00:08,  6.82it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.15 GB):   2%|▏         | 1/58 [00:00<00:08,  6.82it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.15 GB):   3%|▎         | 2/58 [00:00<00:08,  7.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.15 GB):   3%|▎         | 2/58 [00:00<00:08,  7.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.25 GB):   3%|▎         | 2/58 [00:00<00:08,  7.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.25 GB):   7%|▋         | 4/58 [00:00<00:05,  9.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.26 GB):   7%|▋         | 4/58 [00:00<00:05,  9.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.26 GB):   9%|▊         | 5/58 [00:00<00:06,  8.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.25 GB):   9%|▊         | 5/58 [00:00<00:06,  8.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.25 GB):  10%|█         | 6/58 [00:00<00:05,  8.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.24 GB):  10%|█         | 6/58 [00:00<00:05,  8.73it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.24 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.22 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.30 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.30 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.30 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.53it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.22 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.22 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.36 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.35 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.52it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.35 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.21 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.42 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.42 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.41 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.36it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.41 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.41 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.19 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.47 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.79it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.47 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.50 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.17 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.17 GB):  36%|███▌      | 21/58 [00:01<00:02, 12.92it/s]Capturing num tokens (num_tokens=960 avail_mem=58.52 GB):  36%|███▌      | 21/58 [00:01<00:02, 12.92it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=59.18 GB):  36%|███▌      | 21/58 [00:02<00:02, 12.92it/s]Capturing num tokens (num_tokens=896 avail_mem=59.18 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.68it/s]Capturing num tokens (num_tokens=832 avail_mem=59.18 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.68it/s]Capturing num tokens (num_tokens=768 avail_mem=58.58 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.68it/s]Capturing num tokens (num_tokens=768 avail_mem=58.58 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.90it/s]Capturing num tokens (num_tokens=704 avail_mem=59.17 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.90it/s]

    Capturing num tokens (num_tokens=640 avail_mem=59.08 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.90it/s]Capturing num tokens (num_tokens=640 avail_mem=59.08 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.71it/s]Capturing num tokens (num_tokens=576 avail_mem=58.64 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.71it/s]Capturing num tokens (num_tokens=512 avail_mem=59.15 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.71it/s]Capturing num tokens (num_tokens=512 avail_mem=59.15 GB):  50%|█████     | 29/58 [00:02<00:01, 15.02it/s]Capturing num tokens (num_tokens=480 avail_mem=58.67 GB):  50%|█████     | 29/58 [00:02<00:01, 15.02it/s]

    Capturing num tokens (num_tokens=448 avail_mem=59.16 GB):  50%|█████     | 29/58 [00:02<00:01, 15.02it/s]Capturing num tokens (num_tokens=448 avail_mem=59.16 GB):  53%|█████▎    | 31/58 [00:02<00:01, 15.60it/s]Capturing num tokens (num_tokens=416 avail_mem=58.70 GB):  53%|█████▎    | 31/58 [00:02<00:01, 15.60it/s]

    Capturing num tokens (num_tokens=384 avail_mem=58.70 GB):  53%|█████▎    | 31/58 [00:02<00:01, 15.60it/s]Capturing num tokens (num_tokens=384 avail_mem=58.70 GB):  57%|█████▋    | 33/58 [00:02<00:01, 12.98it/s]Capturing num tokens (num_tokens=352 avail_mem=59.15 GB):  57%|█████▋    | 33/58 [00:02<00:01, 12.98it/s]Capturing num tokens (num_tokens=320 avail_mem=58.71 GB):  57%|█████▋    | 33/58 [00:02<00:01, 12.98it/s]Capturing num tokens (num_tokens=320 avail_mem=58.71 GB):  60%|██████    | 35/58 [00:02<00:01, 13.83it/s]Capturing num tokens (num_tokens=288 avail_mem=59.14 GB):  60%|██████    | 35/58 [00:02<00:01, 13.83it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.74 GB):  60%|██████    | 35/58 [00:02<00:01, 13.83it/s]Capturing num tokens (num_tokens=256 avail_mem=58.74 GB):  64%|██████▍   | 37/58 [00:03<00:01, 14.56it/s]Capturing num tokens (num_tokens=240 avail_mem=59.13 GB):  64%|██████▍   | 37/58 [00:03<00:01, 14.56it/s]Capturing num tokens (num_tokens=224 avail_mem=58.77 GB):  64%|██████▍   | 37/58 [00:03<00:01, 14.56it/s]Capturing num tokens (num_tokens=224 avail_mem=58.77 GB):  67%|██████▋   | 39/58 [00:03<00:01, 15.59it/s]Capturing num tokens (num_tokens=208 avail_mem=59.11 GB):  67%|██████▋   | 39/58 [00:03<00:01, 15.59it/s]

    Capturing num tokens (num_tokens=192 avail_mem=59.13 GB):  67%|██████▋   | 39/58 [00:03<00:01, 15.59it/s]Capturing num tokens (num_tokens=192 avail_mem=59.13 GB):  71%|███████   | 41/58 [00:03<00:01, 16.40it/s]Capturing num tokens (num_tokens=176 avail_mem=58.82 GB):  71%|███████   | 41/58 [00:03<00:01, 16.40it/s]Capturing num tokens (num_tokens=160 avail_mem=59.12 GB):  71%|███████   | 41/58 [00:03<00:01, 16.40it/s]Capturing num tokens (num_tokens=160 avail_mem=59.12 GB):  74%|███████▍  | 43/58 [00:03<00:00, 16.85it/s]Capturing num tokens (num_tokens=144 avail_mem=58.85 GB):  74%|███████▍  | 43/58 [00:03<00:00, 16.85it/s]Capturing num tokens (num_tokens=128 avail_mem=59.11 GB):  74%|███████▍  | 43/58 [00:03<00:00, 16.85it/s]

    Capturing num tokens (num_tokens=112 avail_mem=59.10 GB):  74%|███████▍  | 43/58 [00:03<00:00, 16.85it/s]Capturing num tokens (num_tokens=112 avail_mem=59.10 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.24it/s]Capturing num tokens (num_tokens=96 avail_mem=58.89 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.24it/s] Capturing num tokens (num_tokens=80 avail_mem=59.05 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.24it/s]Capturing num tokens (num_tokens=64 avail_mem=59.08 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.24it/s]Capturing num tokens (num_tokens=64 avail_mem=59.08 GB):  84%|████████▍ | 49/58 [00:03<00:00, 19.45it/s]Capturing num tokens (num_tokens=48 avail_mem=59.07 GB):  84%|████████▍ | 49/58 [00:03<00:00, 19.45it/s]

    Capturing num tokens (num_tokens=32 avail_mem=59.07 GB):  84%|████████▍ | 49/58 [00:03<00:00, 19.45it/s]Capturing num tokens (num_tokens=28 avail_mem=58.94 GB):  84%|████████▍ | 49/58 [00:03<00:00, 19.45it/s]Capturing num tokens (num_tokens=28 avail_mem=58.94 GB):  90%|████████▉ | 52/58 [00:03<00:00, 21.33it/s]Capturing num tokens (num_tokens=24 avail_mem=58.94 GB):  90%|████████▉ | 52/58 [00:03<00:00, 21.33it/s]Capturing num tokens (num_tokens=20 avail_mem=59.04 GB):  90%|████████▉ | 52/58 [00:03<00:00, 21.33it/s]Capturing num tokens (num_tokens=16 avail_mem=59.04 GB):  90%|████████▉ | 52/58 [00:03<00:00, 21.33it/s]Capturing num tokens (num_tokens=16 avail_mem=59.04 GB):  95%|█████████▍| 55/58 [00:03<00:00, 22.56it/s]Capturing num tokens (num_tokens=12 avail_mem=59.03 GB):  95%|█████████▍| 55/58 [00:03<00:00, 22.56it/s]

    Capturing num tokens (num_tokens=8 avail_mem=59.03 GB):  95%|█████████▍| 55/58 [00:03<00:00, 22.56it/s] Capturing num tokens (num_tokens=4 avail_mem=58.96 GB):  95%|█████████▍| 55/58 [00:03<00:00, 22.56it/s]Capturing num tokens (num_tokens=4 avail_mem=58.96 GB): 100%|██████████| 58/58 [00:03<00:00, 14.61it/s]


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
    Generated text:  Shkline and I'm a full-time entrepreneur who's passionate about connecting individuals with the skills they need to achieve their goals. I'm a certified life coach, certified public speaker, and a certified personal and professional development coach with a focus on entrepreneurship and innovation. I'm also a certified facilitator of Agile and Scrum methodologies, and I help individuals improve their leadership and coaching skills.
    As an entrepreneur, I'm committed to helping individuals grow their businesses, from idea generation and execution to managing their teams and achieving their goals. I have a strong background in business and have been instrumental in launching and growing several successful small and medium-sized businesses
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a country with a population of 350 million. The president is interested in the number of smartphones in the country, and it is estimated that there are 200 million smartphones. The president decides to visit a supermarket that sells smartphones and he finds that each smartphone can be sold for $50. He realizes that 10% of the smartphones are defective and cannot be sold. If the president wants to maximize the number of phones he can sell for $50, how many smartphones should he buy and how much money should he spend?
    
    To determine how many smartphones the president should buy and how much money he should
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, famous for its 17th-century Notre-Dame Cathedral, Montmartre and the Palace of Versailles.
    Lyon is the capital of the French department of Languedoc-Roussillon, which includes the historic cities of Lyon, Serra and Léon.
    The city of Lyon is famous for its old town, Carrefour, and its Gothic church of St. Peter and Saint Paul.
    Lyon is famous for its rich and distinctive culture and its historical landmarks.
    Paris is the capital of France, with its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it’s in the hands of students. Read on to find out more about what AI is, the future of AI, and how students can engage with AI.
    What is AI?
    AI, short for artificial intelligence, is the simulation of human intelligence in computer systems. AI systems are designed to learn, develop, and perform tasks that would typically require human intelligence. Examples of AI include voice recognition, computer vision, natural language processing, and machine learning.
    The future of AI
    The future of AI is exciting and currently, there are many areas where AI is going to be important. AI is expected to transform many areas of society,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to meet you. I'm [age] years old, and I'm a [gender] person. I have a [job title] at [company name], and I'm always looking for ways to [job title] at [company name]. I'm a [job title] at [company name], and I'm always looking for ways to [job title] at [company name]. I'm a [job title] at [company name], and I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. The city is known for its fashion, art, and cuisine, and it is a major center for business and finance in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that is always on the move, with many new developments and events taking place throughout the year
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be increased scrutiny of how it is used and how it affects society. This will likely lead to more rigorous ethical standards and regulations.
    
    2. Integration with other technologies: AI is already being integrated into a wide range of technologies, including smart homes, self-driving cars, and virtual assistants. As these technologies continue to evolve, it is likely that AI will become more integrated with other technologies, such
    


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
    Generated text:  [Name], and I am a [job title] at [company name]. I have been with the company for [number of years] years and have grown from a [new employee label] to a [job title] with [company name]. In my first job at [company name], I was [mention the role at the company]. I am [mention any personal attributes or interests] and enjoy [mention any hobbies, interests, or passions]. I have always been [influential, independent, supportive, etc.] with a strong sense of [motivational value or value]. I am a [mention any technical skills or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historic city with a rich history dating back to the Middle Ages. It is a major transportation hub and a cultural center, and is home to many famous landmarks such as Notre-Dame Cathedral and the Louvre Museum. The city is also known for its cuisine, wine, and music. It is a city of contrasts, with its many beautiful architecture and diverse population. The city is also a city of history, with many historical sites and museums. The city is an important center for politics, science, and culture in the world. Paris has a rich history and culture, and it is a city that continues to thrive
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by a number of different trends and technologies, including:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient outcomes in hospitals, but the potential for AI in healthcare is expected to grow as more data is collected and analyzed.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, improve quality control, and reduce costs.
    
    3. AI in education: AI is expected to play an increasing role in education, with more schools implementing AI-powered learning tools and platforms.
    
    4. AI in transportation: AI is already being used in self-driving cars, and the potential


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

     am

     a

     [

    occupation

    ]

    !

     I

    'm

     confident

     and

     outgoing

     and

     love

     to

     share

     my

     experiences

     and

     learn

     new

     things

    .

     I

    'm

     really

     excited

     to

     be

     here

     and

     make

     some

     new

     friends

    .

     Let

    's

     build

     something

     together

    !

     [

    Name

    ]

     is

     a

     [

    occupation

    ].

     [

    Name

    ]

     is

     my

     best

     friend

    .

     They

    're

     always

     there

     for

     me

     when

     I

    'm

     feeling

     stuck

     and

     don

    't

     know

     what

     to

     do

    .

     [

    Name

    ]

     is

     a

     [

    occupation

    ].

     We

    're

     always

     together

     and

     we

    're

     really

     good

     friends

    .

     We

    're

     really

     good

     friends

    .

     Let

    's

     make

     some

     new

     friends

     and

     do

     something

     fun

     together

    !

     [

    Name

    ]

     is

     [

    occupation

    ].

     We

    're

     always

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     architecture

    ,

     museums

    ,

     and

     vibrant

     cultural

     scene

    .

     The

     city

     is

     also

     home

     to

     many

     important

     landmarks

     and

     festivals

     throughout

     the

     year

    .

     French

     cuisine

     is

     also

     a

     significant

     part

     of

     Paris

     culture

    ,

     with

     many

     renowned

     restaurants

     and

     food

     stalls

     available

    .

     It

     has

     a

     long

     and

     rich

     history

    ,

     dating

     back

     to

     the

     Roman

     Empire

    ,

     and

     is

     a

     vibrant

     and

     evolving

     city

     with

     a

     diverse

     range

     of

     cultures

     and

     influences

    .

     In

     summary

    ,

     Paris

     is

     a

     city

     with

     a

     rich

     and

     complex

     history

    ,

     known

     for

     its

     cultural

     attractions

    ,

     culinary

     traditions

    ,

     and

     ongoing

     evolution

    .

     It

     is

     a

     beautiful

    ,

     historical

    ,

     and

     fascinating

     city

     that

     is

     always

     on

     the

     move

    ,

     with

     many

     exciting

     events

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

    ,

     and

     it

     will

     continue

     to

     evolve

     and

     improve

     significantly

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     are

     currently

     being

     explored

     and

     researched

    :
    


    1

    .

     Increased

     transparency

    :

     AI

     systems

     will

     become

     more

     transparent

    ,

     making

     it

     easier

     for

     people

     to

     understand

     how

     they

     are

     being

     used

     and

     how

     the

     results

     are

     being

     generated

    .

     This

     will

     help

     to

     increase

     trust

     in

     AI

     and

     make

     it

     more

     accessible

     to

     everyone

    .
    


    2

    .

     Improved

     ethical

     AI

    :

     AI

     systems

     will

     become

     more

     ethical

     and

     honest

    ,

     with

     more

     emphasis

     on

     the

     benefits

     of

     using

     AI

     over

     the

     potential

     risks

    .

     This

     will

     lead

     to

     more

     positive

     outcomes

     for

     society

     as

     a

     whole

    .
    


    3

    .

     Development

     of

     AI

     agents

    :

     The

     development

     of

    



```python
llm.shutdown()
```
