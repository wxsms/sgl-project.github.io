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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.47it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.46it/s]


    2026-04-09 07:20:01,802 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 07:20:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.02it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.02it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.02it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.02it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:09,  5.02it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:05,  7.87it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:05,  7.87it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:05,  7.87it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:05,  7.87it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:05,  7.87it/s]

    Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:05,  7.87it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.24it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:02, 16.81it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:02, 16.81it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:02, 16.81it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:02, 16.81it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:02, 16.81it/s]

    Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:02, 16.81it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:02, 16.81it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 23.10it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 23.10it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 23.10it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 23.10it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 23.10it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 23.10it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 23.10it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 23.10it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 30.89it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 30.89it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 30.89it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 30.89it/s]

    Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 30.89it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 30.89it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 30.89it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 30.89it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 37.73it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]

    Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 42.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 53.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.39 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.38 GB):   7%|▋         | 4/58 [00:00<00:03, 15.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.36 GB):   7%|▋         | 4/58 [00:00<00:03, 15.08it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.13 GB):   7%|▋         | 4/58 [00:00<00:03, 15.08it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.13 GB):  10%|█         | 6/58 [00:00<00:05,  9.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.33 GB):  10%|█         | 6/58 [00:00<00:05,  9.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.31 GB):  10%|█         | 6/58 [00:00<00:05,  9.12it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.31 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.32 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.82 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.82 GB):  17%|█▋        | 10/58 [00:00<00:04, 10.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.66 GB):  17%|█▋        | 10/58 [00:00<00:04, 10.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.48 GB):  17%|█▋        | 10/58 [00:01<00:04, 10.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.49 GB):  17%|█▋        | 10/58 [00:01<00:04, 10.32it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=73.49 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.61 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.51 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.63 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.63 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.61 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.20it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=73.61 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.61 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.58 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.37it/s]Capturing num tokens (num_tokens=960 avail_mem=73.59 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.37it/s] Capturing num tokens (num_tokens=896 avail_mem=73.59 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.37it/s]Capturing num tokens (num_tokens=832 avail_mem=73.58 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.37it/s]Capturing num tokens (num_tokens=832 avail_mem=73.58 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.20it/s]Capturing num tokens (num_tokens=768 avail_mem=73.57 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.20it/s]Capturing num tokens (num_tokens=704 avail_mem=73.55 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.20it/s]Capturing num tokens (num_tokens=640 avail_mem=73.56 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.20it/s]

    Capturing num tokens (num_tokens=576 avail_mem=73.56 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.20it/s]Capturing num tokens (num_tokens=576 avail_mem=73.56 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.37it/s]Capturing num tokens (num_tokens=512 avail_mem=73.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.37it/s]Capturing num tokens (num_tokens=480 avail_mem=73.55 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.37it/s]Capturing num tokens (num_tokens=448 avail_mem=73.55 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.37it/s]

    Capturing num tokens (num_tokens=448 avail_mem=73.55 GB):  53%|█████▎    | 31/58 [00:01<00:01, 23.93it/s]Capturing num tokens (num_tokens=416 avail_mem=73.54 GB):  53%|█████▎    | 31/58 [00:01<00:01, 23.93it/s]Capturing num tokens (num_tokens=384 avail_mem=73.54 GB):  53%|█████▎    | 31/58 [00:01<00:01, 23.93it/s]Capturing num tokens (num_tokens=352 avail_mem=73.52 GB):  53%|█████▎    | 31/58 [00:01<00:01, 23.93it/s]Capturing num tokens (num_tokens=352 avail_mem=73.52 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.13it/s]Capturing num tokens (num_tokens=320 avail_mem=73.53 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.13it/s]Capturing num tokens (num_tokens=288 avail_mem=73.53 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.13it/s]Capturing num tokens (num_tokens=256 avail_mem=73.50 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.13it/s]Capturing num tokens (num_tokens=240 avail_mem=73.50 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.13it/s]

    Capturing num tokens (num_tokens=224 avail_mem=73.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.13it/s]Capturing num tokens (num_tokens=224 avail_mem=73.49 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=208 avail_mem=73.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=192 avail_mem=73.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=176 avail_mem=73.47 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=160 avail_mem=73.47 GB):  67%|██████▋   | 39/58 [00:02<00:00, 29.79it/s]Capturing num tokens (num_tokens=144 avail_mem=73.46 GB):  67%|██████▋   | 39/58 [00:02<00:00, 29.79it/s]Capturing num tokens (num_tokens=144 avail_mem=73.46 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.90it/s]Capturing num tokens (num_tokens=128 avail_mem=73.45 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.90it/s]Capturing num tokens (num_tokens=112 avail_mem=73.45 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.90it/s]Capturing num tokens (num_tokens=96 avail_mem=73.44 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.90it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=73.43 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.90it/s]Capturing num tokens (num_tokens=80 avail_mem=73.43 GB):  83%|████████▎ | 48/58 [00:02<00:00, 35.49it/s]Capturing num tokens (num_tokens=64 avail_mem=73.43 GB):  83%|████████▎ | 48/58 [00:02<00:00, 35.49it/s]Capturing num tokens (num_tokens=48 avail_mem=73.42 GB):  83%|████████▎ | 48/58 [00:02<00:00, 35.49it/s]Capturing num tokens (num_tokens=32 avail_mem=73.44 GB):  83%|████████▎ | 48/58 [00:02<00:00, 35.49it/s]

    Capturing num tokens (num_tokens=28 avail_mem=73.43 GB):  83%|████████▎ | 48/58 [00:02<00:00, 35.49it/s]Capturing num tokens (num_tokens=28 avail_mem=73.43 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.21it/s]Capturing num tokens (num_tokens=24 avail_mem=73.40 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.21it/s]Capturing num tokens (num_tokens=20 avail_mem=73.41 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.21it/s]Capturing num tokens (num_tokens=16 avail_mem=73.41 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.21it/s]Capturing num tokens (num_tokens=12 avail_mem=73.40 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.21it/s]Capturing num tokens (num_tokens=12 avail_mem=73.40 GB):  97%|█████████▋| 56/58 [00:02<00:00, 30.71it/s]Capturing num tokens (num_tokens=8 avail_mem=73.40 GB):  97%|█████████▋| 56/58 [00:02<00:00, 30.71it/s] Capturing num tokens (num_tokens=4 avail_mem=73.39 GB):  97%|█████████▋| 56/58 [00:02<00:00, 30.71it/s]Capturing num tokens (num_tokens=4 avail_mem=73.39 GB): 100%|██████████| 58/58 [00:02<00:00, 23.12it/s]


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
    Generated text:  Leo. I'm a part-time student. I have a special way of learning. I can learn anything I want to know. I can teach people how to learn. I can learn things I don't know. I do this by reading books, watching videos, and playing games. What do you think of me? Well, I like playing computer games. I like to watch videos. I like to read books. I like to play games. I think that's a good way to learn. I like to learn at my own pace. I can learn anything. What do you think? I think you are a good student. You
    ===============================
    Prompt: The president of the United States is
    Generated text:  a famous person in the world. Some people think the president does not need to do anything. But in fact, the president has a lot of responsibilities. Here are some ways to help the president do his work well. 1. Read newspapers and listen to the radio carefully. 2. Keep a diary to keep track of important news. 3. Think about the president's work. When he decides to make decisions, he will have a lot of ideas. 4. Think about how to deal with people. The president is the most important in the country, and he has to make important decisions. To do this, he
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It's a beautiful city in the south of France. It's far from the sea, so it has many hills. There are many gardens around the city, and many places to eat there. There are also lots of cafes and restaurants. Paris is the biggest city in France. It's the capital of the country. It's a very busy city. It's very big. It's also very beautiful. It has many tall buildings. The city is always changing. The new buildings are nice to look at. Some of the tall buildings are very old. Some of the old buildings are very new. It's a very interesting
    ===============================
    Prompt: The future of AI is
    Generated text:  mobile and connected. It is there, in the hands of its users and the devices they use. It is the next big thing and may soon be the next next big thing. What do you think?
    
    A. Yes, it is a real possibility in the near future.
    B. No, it is not a real possibility.
    C. I don’t know.
    D. I don’t think so.
    
    Based on the following options, what is the answer? To answer this question, you should consider the topic, context, and available information. Avoid answering and provide your answer before the final mark has been displayed. To find the correct answer


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the field]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [reason for interest in the field] and I'm always eager to learn and improve. I'm a [reason for interest in the field] and I'm always eager to learn and improve. I'm a [reason for interest in the field] and I'm always eager to learn and improve. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union. Paris is a historic center of France and a major cultural, economic, and political center. It is home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its cuisine, fashion, and art scene. It is a popular tourist destination and a major center for business and finance in Europe. The city is also home to many international organizations and institutions, including the European Parliament and the European Central Bank. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more and more AI systems are being developed, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential consequences of their creations and work to ensure that they are developed in a way that is fair and responsible.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including
    


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
    Generated text:  [Your Name]. I'm an engineer with a passion for designing and implementing cutting-edge technology solutions for organizations of all sizes. My expertise lies in project management, software development, and the creation of innovative solutions that enhance productivity and efficiency in various industries. I'm always seeking to learn and expand my knowledge in the field, and I'm excited to contribute my skills to help people succeed in their endeavors. Thank you for considering me for a job. Happy coding! 😊✨  
    Can you tell me more about your background and experience in project management? 🤔
    Absolutely! As an engineer with a strong background in project management, my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the "City of Light." The city is renowned for its stunning architecture, cultural richness, and rich history. It is home to the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral, among other notable landmarks. Paris is also known for its vibrant nightlife and delicious cuisine. The city is known for its cultural and artistic scene, and it is home to many prestigious institutions of higher education. Paris is one of the most visited cities in the world, and it is a popular tourist destination for many visitors. The city is also home to a diverse population, with residents coming from all over the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and bright, with potential to transform every aspect of human life. Some of the potential trends include:
    
    1. AI will continue to become more sophisticated and efficient, allowing for more complex and advanced applications. This could include healthcare, transportation, manufacturing, and even finance.
    
    2. AI will continue to become more autonomous, with the ability to learn and adapt in real-time. This will allow for more efficient and cost-effective operations.
    
    3. AI will continue to integrate into everyday life, with smart homes, self-driving cars, and the use of AI in natural language processing.
    
    4. AI will continue to be integrated into education, with more


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

    'm

     a

     [

    Your

     Profession

    ]

     who

     has

     been

     a

     [

    Your

     Career

     Goal

     or

     Experience

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     am

     passionate

     about

     [

    Your

     Profession

    ]

     and

     enjoy

     working

     with

     people

     to

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

     am

     also

     a

     [

    Your

     Personal

     Value

     or

     Character

     Trait

    ]

     who

     strive

     to

     be

     the

     best

     version

     of

     myself

     every

     day

    .

     I

     believe

     that

     with

     hard

     work

    ,

     dedication

    ,

     and

     a

     positive

     attitude

    ,

     anything

     is

     possible

    .

     I

     am

     excited

     to

     meet

     you

     and

     learn

     more

     about

     your

     story

    .

     What

    's

     your

     name

     and

     what

    's

     your

     profession

    ?

     That

    's

     all

     I

     need

     to

     know

    .

     Hi

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

     love

    ,

     is

     renowned

     for

     its

     rich

     history

    ,

     iconic

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     The

     iconic

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

     Notre

    -D

    ame

     Cathedral

     are

     among

     the

     most

     famous

     landmarks

     in

     the

     world

    .

     Paris

     is

     also

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     popular

     tourist

     destinations

    .

     The

     city

     is

     known

     for

     its

     coffee

     culture

     and

     the

     annual

     E

    iff

    el

     Tower

     and

     Carn

    ava

    let

     lights

     extrav

    agan

    za

    .

     It

    's

     a

     city

     that

     has

     a

     rich

     history

     and

     a

     unique

     blend

     of

     traditional

     and

     modern

     influences

    .

     Paris

     is

     a

     unique

     and

     unforgettable

     destination

    ,

     making

     it

     a

     must

    -

    visit

     city

     for

     anyone

     interested

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     incredibly

     diverse

     and

     rapidly

     evolving

    .

     Here

     are

     some

     possible

     trends

     in

     the

     AI

     landscape

     in

     the

     next

     decade

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     As

     more

     AI

     systems

     become

     integrated

     with

     human

     intelligence

    ,

     we

     can

     expect

     to

     see

     more

     complex

     and

     nuanced

     AI

     that

     can

     handle

     a

     wide

     range

     of

     tasks

    ,

     including

     human

    -like

     decision

    -making

    ,

     problem

    -solving

    ,

     and

     emotional

     intelligence

    .
    


    2

    .

     Greater

     focus

     on

     ethical

     AI

    :

     As

     society

     gr

    app

    les

     with

     issues

     of

     privacy

    ,

     bias

    ,

     and

     accountability

     in

     AI

    ,

     we

     can

     expect

     to

     see

     greater

     focus

     on

     ethical

     AI

     standards

     and

     regulations

    .

     This

     may

     lead

     to

     more

     transparent

     and

     accountable

     AI

     development

     and

     deployment

    ,

     as

     well

     as

    



```python
llm.shutdown()
```
