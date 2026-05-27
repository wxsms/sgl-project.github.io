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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:32,  1.64s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:32,  1.64s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:04<01:32,  1.64s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:36,  1.48it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:36,  1.48it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:36,  1.48it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:20,  2.54it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:20,  2.54it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:20,  2.54it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:12,  3.85it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:12,  3.85it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:12,  3.85it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.06it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.06it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.06it/s]

    Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:05,  8.06it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:02, 13.82it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:02, 13.82it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:02, 13.82it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:02, 13.82it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 15.82it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 15.82it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 15.82it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 15.82it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 18.49it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 18.49it/s]

    Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 18.49it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 18.49it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 24.29it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 24.29it/s]

    Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 24.29it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 24.29it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 24.29it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 27.94it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 27.94it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 27.94it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 27.94it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 27.94it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 32.09it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 32.09it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 32.09it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 32.09it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 32.09it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 32.09it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 34.58it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 34.58it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 34.58it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 34.58it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 34.58it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 35.58it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 35.58it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 35.58it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 35.58it/s]

    Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 35.58it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 35.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=39.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=39.54 GB):   2%|▏         | 1/58 [00:00<00:06,  8.38it/s]Capturing num tokens (num_tokens=7680 avail_mem=39.08 GB):   2%|▏         | 1/58 [00:00<00:06,  8.38it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=39.08 GB):   3%|▎         | 2/58 [00:00<00:09,  5.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.40 GB):   3%|▎         | 2/58 [00:00<00:09,  5.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.40 GB):   5%|▌         | 3/58 [00:00<00:10,  5.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.78 GB):   5%|▌         | 3/58 [00:00<00:10,  5.46it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.78 GB):   7%|▋         | 4/58 [00:00<00:09,  5.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=39.52 GB):   7%|▋         | 4/58 [00:00<00:09,  5.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=39.52 GB):   9%|▊         | 5/58 [00:00<00:08,  6.08it/s]Capturing num tokens (num_tokens=5632 avail_mem=39.88 GB):   9%|▊         | 5/58 [00:00<00:08,  6.08it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=39.88 GB):  10%|█         | 6/58 [00:00<00:08,  6.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.55 GB):  10%|█         | 6/58 [00:00<00:08,  6.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.55 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.88 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.10it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=38.88 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=39.56 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=39.56 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.93 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.61it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=38.93 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.76it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.65 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.76it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.65 GB):  19%|█▉        | 11/58 [00:01<00:06,  6.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.92 GB):  19%|█▉        | 11/58 [00:01<00:06,  6.82it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=38.92 GB):  21%|██        | 12/58 [00:01<00:06,  7.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=39.59 GB):  21%|██        | 12/58 [00:01<00:06,  7.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=39.59 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.98 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.48it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=38.98 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.75 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.75 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=39.83 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.00it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=39.83 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.85 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.85 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=39.03 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.76it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=39.03 GB):  31%|███       | 18/58 [00:02<00:04,  8.45it/s]Capturing num tokens (num_tokens=1536 avail_mem=39.82 GB):  31%|███       | 18/58 [00:02<00:04,  8.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=39.67 GB):  31%|███       | 18/58 [00:02<00:04,  8.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=39.67 GB):  34%|███▍      | 20/58 [00:02<00:03,  9.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=39.06 GB):  34%|███▍      | 20/58 [00:02<00:03,  9.59it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=39.06 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.44it/s]Capturing num tokens (num_tokens=960 avail_mem=39.79 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.44it/s] Capturing num tokens (num_tokens=896 avail_mem=39.77 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.44it/s]Capturing num tokens (num_tokens=896 avail_mem=39.77 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.52it/s]Capturing num tokens (num_tokens=832 avail_mem=39.12 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.52it/s]

    Capturing num tokens (num_tokens=768 avail_mem=39.77 GB):  40%|███▉      | 23/58 [00:03<00:03, 10.52it/s]Capturing num tokens (num_tokens=768 avail_mem=39.77 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.21it/s]Capturing num tokens (num_tokens=704 avail_mem=39.77 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.21it/s]Capturing num tokens (num_tokens=640 avail_mem=39.17 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.21it/s]

    Capturing num tokens (num_tokens=640 avail_mem=39.17 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.90it/s]Capturing num tokens (num_tokens=576 avail_mem=39.68 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.90it/s]Capturing num tokens (num_tokens=512 avail_mem=39.21 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.90it/s]Capturing num tokens (num_tokens=512 avail_mem=39.21 GB):  50%|█████     | 29/58 [00:03<00:02, 11.51it/s]Capturing num tokens (num_tokens=480 avail_mem=39.23 GB):  50%|█████     | 29/58 [00:03<00:02, 11.51it/s]

    Capturing num tokens (num_tokens=448 avail_mem=39.75 GB):  50%|█████     | 29/58 [00:03<00:02, 11.51it/s]Capturing num tokens (num_tokens=448 avail_mem=39.75 GB):  53%|█████▎    | 31/58 [00:03<00:02, 12.31it/s]Capturing num tokens (num_tokens=416 avail_mem=39.24 GB):  53%|█████▎    | 31/58 [00:03<00:02, 12.31it/s]Capturing num tokens (num_tokens=384 avail_mem=39.73 GB):  53%|█████▎    | 31/58 [00:03<00:02, 12.31it/s]

    Capturing num tokens (num_tokens=384 avail_mem=39.73 GB):  57%|█████▋    | 33/58 [00:03<00:01, 12.90it/s]Capturing num tokens (num_tokens=352 avail_mem=39.26 GB):  57%|█████▋    | 33/58 [00:03<00:01, 12.90it/s]Capturing num tokens (num_tokens=320 avail_mem=39.21 GB):  57%|█████▋    | 33/58 [00:03<00:01, 12.90it/s]Capturing num tokens (num_tokens=320 avail_mem=39.21 GB):  60%|██████    | 35/58 [00:03<00:01, 13.01it/s]Capturing num tokens (num_tokens=288 avail_mem=39.71 GB):  60%|██████    | 35/58 [00:03<00:01, 13.01it/s]

    Capturing num tokens (num_tokens=256 avail_mem=39.28 GB):  60%|██████    | 35/58 [00:04<00:01, 13.01it/s]Capturing num tokens (num_tokens=256 avail_mem=39.28 GB):  64%|██████▍   | 37/58 [00:04<00:01, 13.04it/s]Capturing num tokens (num_tokens=240 avail_mem=39.70 GB):  64%|██████▍   | 37/58 [00:04<00:01, 13.04it/s]Capturing num tokens (num_tokens=224 avail_mem=39.26 GB):  64%|██████▍   | 37/58 [00:04<00:01, 13.04it/s]Capturing num tokens (num_tokens=224 avail_mem=39.26 GB):  67%|██████▋   | 39/58 [00:04<00:01, 13.49it/s]Capturing num tokens (num_tokens=208 avail_mem=39.68 GB):  67%|██████▋   | 39/58 [00:04<00:01, 13.49it/s]

    Capturing num tokens (num_tokens=192 avail_mem=39.32 GB):  67%|██████▋   | 39/58 [00:04<00:01, 13.49it/s]Capturing num tokens (num_tokens=192 avail_mem=39.32 GB):  71%|███████   | 41/58 [00:04<00:01, 13.95it/s]Capturing num tokens (num_tokens=176 avail_mem=39.68 GB):  71%|███████   | 41/58 [00:04<00:01, 13.95it/s]Capturing num tokens (num_tokens=160 avail_mem=39.33 GB):  71%|███████   | 41/58 [00:04<00:01, 13.95it/s]Capturing num tokens (num_tokens=160 avail_mem=39.33 GB):  74%|███████▍  | 43/58 [00:04<00:01, 14.75it/s]Capturing num tokens (num_tokens=144 avail_mem=39.66 GB):  74%|███████▍  | 43/58 [00:04<00:01, 14.75it/s]

    Capturing num tokens (num_tokens=128 avail_mem=39.35 GB):  74%|███████▍  | 43/58 [00:04<00:01, 14.75it/s]Capturing num tokens (num_tokens=128 avail_mem=39.35 GB):  78%|███████▊  | 45/58 [00:04<00:00, 15.29it/s]Capturing num tokens (num_tokens=112 avail_mem=39.64 GB):  78%|███████▊  | 45/58 [00:04<00:00, 15.29it/s]Capturing num tokens (num_tokens=96 avail_mem=39.63 GB):  78%|███████▊  | 45/58 [00:04<00:00, 15.29it/s] Capturing num tokens (num_tokens=96 avail_mem=39.63 GB):  81%|████████  | 47/58 [00:04<00:00, 16.05it/s]Capturing num tokens (num_tokens=80 avail_mem=39.40 GB):  81%|████████  | 47/58 [00:04<00:00, 16.05it/s]

    Capturing num tokens (num_tokens=64 avail_mem=39.62 GB):  81%|████████  | 47/58 [00:04<00:00, 16.05it/s]Capturing num tokens (num_tokens=64 avail_mem=39.62 GB):  84%|████████▍ | 49/58 [00:04<00:00, 16.10it/s]Capturing num tokens (num_tokens=48 avail_mem=39.61 GB):  84%|████████▍ | 49/58 [00:04<00:00, 16.10it/s]Capturing num tokens (num_tokens=32 avail_mem=39.42 GB):  84%|████████▍ | 49/58 [00:04<00:00, 16.10it/s]Capturing num tokens (num_tokens=28 avail_mem=39.45 GB):  84%|████████▍ | 49/58 [00:04<00:00, 16.10it/s]

    Capturing num tokens (num_tokens=28 avail_mem=39.45 GB):  90%|████████▉ | 52/58 [00:04<00:00, 17.91it/s]Capturing num tokens (num_tokens=24 avail_mem=39.57 GB):  90%|████████▉ | 52/58 [00:04<00:00, 17.91it/s]Capturing num tokens (num_tokens=20 avail_mem=39.56 GB):  90%|████████▉ | 52/58 [00:05<00:00, 17.91it/s]Capturing num tokens (num_tokens=20 avail_mem=39.56 GB):  93%|█████████▎| 54/58 [00:05<00:00, 18.15it/s]Capturing num tokens (num_tokens=16 avail_mem=39.57 GB):  93%|█████████▎| 54/58 [00:05<00:00, 18.15it/s]Capturing num tokens (num_tokens=12 avail_mem=39.44 GB):  93%|█████████▎| 54/58 [00:05<00:00, 18.15it/s]

    Capturing num tokens (num_tokens=12 avail_mem=39.44 GB):  97%|█████████▋| 56/58 [00:05<00:00, 18.62it/s]Capturing num tokens (num_tokens=8 avail_mem=39.52 GB):  97%|█████████▋| 56/58 [00:05<00:00, 18.62it/s] Capturing num tokens (num_tokens=4 avail_mem=39.48 GB):  97%|█████████▋| 56/58 [00:05<00:00, 18.62it/s]Capturing num tokens (num_tokens=4 avail_mem=39.48 GB): 100%|██████████| 58/58 [00:05<00:00, 11.07it/s]


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
    Generated text:  George and I'm a third year medical student at Harvard University. I have a lot of experience of practical operations and am aware of the importance of doing patient care in a compassionate and professional manner. I have a great deal of knowledge about the human body, physiology and the medical field in general. I have completed a Bachelor's of Science degree in Biology, with a minor in chemistry, and a Master's of Science degree in Medicine, with a specialization in clinical anesthesia.
    I have been in the medical field for almost 5 years and have worked on a variety of projects that involved patient care and treatment. I have also participated in various training
    ===============================
    Prompt: The president of the United States is
    Generated text:  in the executive branch of the government. Which of the following is NOT a branch of the government? The answer is
    
    The answer is **the legislative branch**. The legislative branch of the United States government consists of the Congress, which includes the Senate and the House of Representatives. The executive branch (the president) is responsible for implementing the laws passed by Congress. Therefore, the answer that is not a branch of the government is **the executive branch**.
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    
    The capital of France is Paris. The correct answer is: Paris
    
    To explain this in more detail:
    
    1. **Location**: Paris is located on the Île de la Cité, a large island in the Bataclais and Moulay Ismail botezoum canals, situated in the North Sea and the Atlantic Ocean, and some of the most important cities in France.
    
    2. **Government**: Paris is the capital city of France and the country's largest metropolitan area. It is the third largest city in Europe by population after London and Paris.
    
    3. **Culture and History**: Paris is known for
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain. The adoption of AI is closely linked with the socio-economic development, which in turn depends on the socio-economic development, which in turn depends on the socio-economic development, and so on. It is therefore highly plausible that AI will not change the world as we know it, but might even enable us to solve some of the biggest problems in the world.
    
    If AI technology is developed in the future, one of its important and major tasks will be to transform the way we handle and learn from the data. The first step is to ensure that data is not misused or improperly stored. This will require the implementation of more strict data


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your favorite hobby or activity? I'm a [job title] at [company name], and I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as numerous museums, theaters, and restaurants. Paris is also known for its fashion industry, with many famous designers and fashion houses operating in the city. The city is a major center for business, education, and entertainment, and is a popular tourist destination. It is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, transportation, healthcare, and finance. Automation will likely lead to increased efficiency and productivity, but it will also lead to the loss of jobs for humans.
    
    2. AI ethics and privacy concerns: As AI becomes more advanced, there will be increasing concerns about its ethical implications and potential privacy violations. There will likely be a need for regulations and guidelines
    


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
    Generated text:  [Name] and I'm a [Title/Job] with [Company Name]. I enjoy [Personal Trait/Interest/ passion]. My hobbies include [List of hobbies]. What is your background and how did you get to this position? I am [Age/Experience], and my education includes [Degree or College]. How do you stay up-to-date with the latest developments in your field? I love to read, listen to music, and stay in touch with my family and friends. What is your favorite hobby or activity? I have a love for [Favorite Activity/Religious/Artistic Expression/ etc.] and enjoy making new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest and most populous city in the country and is known for its rich history, art, and architecture. Paris is also one of the world's most tourist destinations, famous for its landmarks like Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower. It has a strong French influence on its culture and cuisine, and is known for its annual Mardi Gras celebrations and its role as a hub for the world of fashion. The city is also home to many prestigious universities, and is a major economic and financial center. Paris has a diverse population and is home to many different cultures and ethnic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bound to be both fascinating and unpredictable. In the near future, we may see the following trends:
    
    1. Increased intelligence and understanding of human emotions: As AI becomes more capable of interpreting and understanding human emotions, it could be used to create more intelligent and empathetic machines. This could lead to new forms of AI, such as "intelligent machines with a heart," that could understand and respond to the emotions of humans.
    
    2. AI-powered healthcare advancements: AI has the potential to revolutionize the way we approach healthcare, from predicting patient outcomes to developing personalized treatment plans. AI could be used to analyze medical images and identify diseases early,


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

     ____

    _.

     I

    'm

     a

    /an

     [

    insert

     occupation

    ]

     with

     a

    /an

     [

    insert

     relevant

     experience

     or

     skill

     set

    ]

     that

     has

     experience

     in

     [

    insert

     specific

     task

     or

     project

    ].

     I

     enjoy

     [

    insert

     personal

     interest

    ].

     I

     believe

     that

     _____

     and

     strive

     to

     [

    insert

     personal

     goal

     or

     aspiration

    ].

     I

    'm

     eager

     to

     [

    insert

     personal

     desire

     or

     curiosity

    ].

     My

     main

     goal

     is

     to

     [

    insert

     main

     goal

     or

     aspiration

    ].

     Thank

     you

     for

     having

     me

    .

     Please

     let

     me

     know

     if

     you

     would

     like

     to

     proceed

     with

     the

     introduction

     or

     if

     there

     are

     any

     specific

     topics

     you

    'd

     like

     to

     cover

    .

     (

    If

     you

     have

     any

     questions

     or

     need

     further

     information

    ,

     please

     don

    't

     hesitate

     to

     ask

    ).

     Let

     me

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     diverse

     cultural

     scene

    .

     It

     is

     a

     bustling

     and

     exciting

     city

     with

     a

     long

     history

     of

     ancient

     civilizations

     and

     a

     modern

     cultural

     center

    .

     Paris

     is

     also

     home

     to

     many

     famous

     landmarks

     and

     museums

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

    's

     location

     on

     the

     coast

     and

     its

     proximity

     to

     the

     Mediterranean

     Sea

     make

     it

     a

     popular

     tourist

     destination

    .

     Paris

     is

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     history

    ,

     art

    ,

     culture

    ,

     and

     nature

    .

     It

     has

     become

     a

     global

     hub

     for

     culture

    ,

     fashion

    ,

     and

     cuisine

    ,

     and

     continues

     to

     attract

     visitors

     from

     around

     the

     world

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

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

     could

     shape

     the

     industry

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     AI

     will

     become

     more

     integrated

     with

     human

     consciousness

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

     possibility

     of

     creating

     AI

     that

     can

     understand

     and

     even

     experience

     consciousness

    .

     This

     could

     lead

     to

     breakthrough

    s

     in

     areas

     such

     as

     artificial

     general

     intelligence

    ,

     where

     AI

     systems

     are

     capable

     of

     performing

     tasks

     without

     human

     intervention

    .

     It

     may

     also

     lead

     to

     the

     development

     of

     advanced

     artificial

     consciousness

    ,

     which

     could

     redefine

     human

     consciousness

     and

     consciousness

     itself

    .
    


    2

    .

     AI

     will

     become

     more

     human

    -like

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

     human

    -like

    .

     This

     could

    



```python
llm.shutdown()
```
