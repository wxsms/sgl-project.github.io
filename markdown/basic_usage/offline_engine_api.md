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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.07it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.07it/s]


    2026-04-06 01:54:46,942 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 01:54:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]

    Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:12,  3.95it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:12,  3.95it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:12,  3.95it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:12,  3.95it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:02<00:12,  3.95it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:02<00:12,  3.95it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:02<00:12,  3.95it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:02<00:05,  8.72it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:02<00:05,  8.72it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:05,  8.72it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:05,  8.72it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:05,  8.72it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:05,  8.72it/s]

    Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:03<00:05,  8.72it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:03<00:05,  8.72it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 15.34it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 15.34it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 15.34it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 15.34it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 15.34it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 15.34it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 15.34it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 15.34it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 22.55it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 22.55it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 22.55it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 22.55it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 22.55it/s]

    Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 22.55it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 22.55it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 22.55it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 34.42it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 34.42it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 34.42it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 34.42it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 34.42it/s]

    Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 34.42it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 34.42it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 34.42it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 41.15it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 48.27it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 48.27it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 48.27it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 48.27it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.22 GB):   2%|▏         | 1/58 [00:00<00:06,  8.87it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.19 GB):   2%|▏         | 1/58 [00:00<00:06,  8.87it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=71.19 GB):   3%|▎         | 2/58 [00:00<00:06,  9.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.19 GB):   3%|▎         | 2/58 [00:00<00:06,  9.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.84 GB):   3%|▎         | 2/58 [00:00<00:06,  9.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.84 GB):   7%|▋         | 4/58 [00:00<00:04, 11.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.18 GB):   7%|▋         | 4/58 [00:00<00:04, 11.11it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=71.18 GB):   7%|▋         | 4/58 [00:00<00:04, 11.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.18 GB):  10%|█         | 6/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.18 GB):  10%|█         | 6/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.93 GB):  10%|█         | 6/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.93 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.17 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.00it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.17 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.17 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.16 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.95 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.15 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.15 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.15 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.15it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=71.14 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.13 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.12 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.11 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.11 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.11 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.65it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=70.98 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.65it/s]Capturing num tokens (num_tokens=960 avail_mem=71.09 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.65it/s] Capturing num tokens (num_tokens=896 avail_mem=71.09 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.65it/s]Capturing num tokens (num_tokens=896 avail_mem=71.09 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.10it/s]Capturing num tokens (num_tokens=832 avail_mem=71.06 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.10it/s]Capturing num tokens (num_tokens=768 avail_mem=71.05 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.10it/s]Capturing num tokens (num_tokens=704 avail_mem=71.05 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.10it/s]Capturing num tokens (num_tokens=640 avail_mem=71.06 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.10it/s]

    Capturing num tokens (num_tokens=640 avail_mem=71.06 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=576 avail_mem=71.06 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=512 avail_mem=71.04 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=480 avail_mem=71.06 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=448 avail_mem=71.05 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=448 avail_mem=71.05 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.57it/s]Capturing num tokens (num_tokens=416 avail_mem=71.04 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.57it/s]Capturing num tokens (num_tokens=384 avail_mem=71.04 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.57it/s]Capturing num tokens (num_tokens=352 avail_mem=71.03 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.57it/s]Capturing num tokens (num_tokens=320 avail_mem=71.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.57it/s]

    Capturing num tokens (num_tokens=320 avail_mem=71.02 GB):  60%|██████    | 35/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  60%|██████    | 35/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=256 avail_mem=71.01 GB):  60%|██████    | 35/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=240 avail_mem=71.02 GB):  60%|██████    | 35/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=224 avail_mem=71.01 GB):  60%|██████    | 35/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=208 avail_mem=70.98 GB):  60%|██████    | 35/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=208 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=192 avail_mem=71.00 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=160 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.52it/s]

    Capturing num tokens (num_tokens=128 avail_mem=70.97 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=128 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=112 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=96 avail_mem=70.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.52it/s] Capturing num tokens (num_tokens=80 avail_mem=70.93 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=64 avail_mem=70.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=64 avail_mem=70.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.60it/s]Capturing num tokens (num_tokens=48 avail_mem=70.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.60it/s]

    Capturing num tokens (num_tokens=32 avail_mem=70.92 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.60it/s]Capturing num tokens (num_tokens=28 avail_mem=70.91 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.60it/s]Capturing num tokens (num_tokens=24 avail_mem=70.90 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.60it/s]Capturing num tokens (num_tokens=24 avail_mem=70.90 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.24it/s]Capturing num tokens (num_tokens=20 avail_mem=70.91 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.24it/s]Capturing num tokens (num_tokens=16 avail_mem=70.89 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.24it/s]Capturing num tokens (num_tokens=12 avail_mem=70.90 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.24it/s]Capturing num tokens (num_tokens=8 avail_mem=70.90 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.24it/s] Capturing num tokens (num_tokens=4 avail_mem=70.89 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.24it/s]

    Capturing num tokens (num_tokens=4 avail_mem=70.89 GB): 100%|██████████| 58/58 [00:02<00:00, 33.06it/s]Capturing num tokens (num_tokens=4 avail_mem=70.89 GB): 100%|██████████| 58/58 [00:02<00:00, 26.09it/s]


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
    Generated text:  Fred and I'm a teacher. In this world, everyone has different personalities and opinions. So I try to keep all the differences within the class. I always remind that the things you think are like mine. Sometimes we disagree, but we shouldn't worry. Because even the most opposite people can still be good friends. And I will make your life better with you. Question: According to the text, how can you become a good friend?  A. By studying everything you know  B. By remembering everything  C. By listening to everyone  D. By having a good attitude
    Answer:
    
    D. By having a good attitude
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is like a leader of the country. He is the leader of the country, and he works for the country. He helps the country have good things happen. He helps the country have not so good things happen. He helps the country to make the country better. That's his job. Some people don't agree with the president. They think he makes some bad things happen. He's like a monster. He makes everyone unhappy. He's mean and he's greedy. He's got a bad habit, like many other politicians. He spends too much money. He's greedy and he thinks he knows the
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Paris
    C. Paris
    D. Paris
    
    The capital of France is Paris. Therefore, the correct answer is:
    
    A. Paris
    
    In the context of the question, the capital of France is Paris, so the completed sentence would read:
    
    "The capital of France is Paris." (Note: The capital of France is Paris, not Paris. The original prompt asked about the capital of France, not Paris. Therefore, the correct answer is Paris.) However, since the original prompt didn't mention Paris, the sentence should be rephrased to match the capital of France, which is Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the human brain. It’s one of the most intriguing and exciting areas of research, and it will probably never end. With the current shortage of workers and the pace of technological advancement, there is a lot to be said for the idea of retraining the workforce with the right skills to meet the demands of the future.
    
    AI and the Human Brain
    
    The world of AI is far from static, and it is a constantly evolving field. This is where the human brain comes in. The brain is one of the most complex and unique systems in the world, and it is the most advanced AI that is currently available.
    
    As


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is known for its beautiful architecture, vibrant culture, and annual festivals. Paris is also a major financial center and a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to many cultural institutions and museums, including the Louvre Museum and the Musée d'Orsay. Paris is a city that is constantly evolving and is known for its creativity and innovation. It is a city that is a symbol of France and a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in
    


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
    Generated text:  [insert character's name], and I'm [insert character's age and gender]. I'm a self-described "charming" introverted person who loves to explore new experiences and learn new things. I've always been fascinated by the world around me, and I'm always eager to discover new things. I have a genuine interest in psychology, and I enjoy reading, traveling, and trying new foods. I'm a bit of a flaky liar, but I don't mind because I like to make a good first impression. I'm a bit of a mystery to some people, but I don't mind because I enjoy exploring the unknown
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France, the second largest in the European Union and the ninth largest in the world. Paris is known for its rich history, elegant architecture, world-famous museums, vibrant culture, and iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, Louvre Museum, and the Arc de Triomphe. It is also one of the world’s most populous cities with a population of over 23 million people. The city is home to many world-renowned cultural institutions and is an important economic and political center in France. As of 2021, Paris was ranked as the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  shaped by a multitude of factors, including advances in computing power, growing data sets, and evolving ethical considerations. Here are some potential trends in AI that may impact how it is used and developed:
    
    1. Increased integration with other technologies: AI is increasingly being integrated with other technologies such as sensors, machine learning, and blockchain, which may lead to new applications and opportunities for AI to be used in new ways.
    
    2. Greater focus on ethical and social implications: As AI becomes more pervasive, there will be increased scrutiny of how it is used and how it affects society. There will be a growing emphasis on ethical considerations and social implications of AI


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

    'm

     here

     to

     work

     with

     you

    .

     I

     specialize

     in

     [

    specific

     skill

     or

     expertise

    ].

     I

    'm

     excited

     to

     hear

     about

     your

     business

     goals

     and

     how

     we

     can

     work

     together

     to

     achieve

     them

    .

     If

     you

     need

     any

     advice

     or

     guidance

    ,

     feel

     free

     to

     reach

     out

     to

     me

    .

     I

     look

     forward

     to

     the

     opportunity

     to

     collaborate

     with

     you

    .

     [

    Name

    ]

     [

    Company

     Name

    ]

     [

    Company

     URL

    ]

     [

    Phone

     Number

    ]

     [

    Email

     Address

    ]

     [

    LinkedIn

     Profile

     (

    optional

    )]

     [

    Personal

     Website

     (

    optional

    )]

     [

    Social

     Media

     Handles

     (

    optional

    )]

     [

    Other

     Contact

     Information

     (

    optional

    )]

     [

    Professional

     Certificate

     (

    if

     applicable

    )]

     [

    Additional

     Information

     (

    if

     relevant

    )]

     [

    Professional

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

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

     many

     famous

     landmarks

     and

     historical

     sites

    ,

     including

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

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     The

     city

     is

     also

     home

     to

     important

     historical

     and

     cultural

     institutions

     such

     as

     the

     Lou

    vre

     Museum

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Muse

    e

     des

     Be

    aux

    -A

    rts

    ,

     among

     others

    .

     Paris

     is

     a

     vibrant

     and

     diverse

     city

     with

     a

     rich

     cultural

     heritage

    ,

     and

     it

     has

     a

     reputation

     as

     one

     of

     the

     most

     desirable

     places

     to

     live

     in

     the

     world

    .

     The

     French

     capital

     is

     known

     for

     its

     luxurious

     hotels

     and

     restaurants

    ,

     including

     the

     E

    iff

    el

     Tower

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     shape

     it

    :
    


    1

    .

     Increased

     usage

     of

     AI

     in

     healthcare

    :

     AI

     is

     becoming

     increasingly

     important

     in

     the

     healthcare

     industry

    ,

     as

     it

     can

     help

     diagnose

     diseases

    ,

     track

     patient

     data

    ,

     and

     provide

     personalized

     treatment

     plans

    .

     This

     trend

     is

     expected

     to

     grow

     as

     more

     and

     more

     healthcare

     systems

     adopt

     AI

     to

     improve

     patient

     outcomes

     and

     efficiency

    .
    


    2

    .

     Emer

    gence

     of

     more

     advanced

     AI

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     are

     likely

     to

     see

     more

     powerful

     and

     intelligent

     AI

     systems

     emerge

    .

     This

     could

     lead

     to

     breakthrough

    s

     in

     fields

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

    .
    


    3

    .

     Integration

     of

     AI

    



```python
llm.shutdown()
```
