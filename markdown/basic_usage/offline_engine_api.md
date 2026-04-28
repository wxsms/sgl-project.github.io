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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.90it/s]


    2026-04-28 02:00:32,338 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 02:00:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=2816):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]

    Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.43 GB):   2%|▏         | 1/58 [00:00<00:07,  7.50it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   2%|▏         | 1/58 [00:00<00:07,  7.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.40 GB):   2%|▏         | 1/58 [00:00<00:07,  7.50it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=137.40 GB):   5%|▌         | 3/58 [00:00<00:04, 13.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.40 GB):   5%|▌         | 3/58 [00:00<00:04, 13.52it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   5%|▌         | 3/58 [00:00<00:04, 13.52it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.39 GB):   5%|▌         | 3/58 [00:00<00:04, 13.52it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.39 GB):  10%|█         | 6/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):  10%|█         | 6/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  10%|█         | 6/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  10%|█         | 6/58 [00:00<00:02, 18.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  10%|█         | 6/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.86it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.02it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=960 avail_mem=137.34 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.28it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=768 avail_mem=137.33 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=768 avail_mem=137.33 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.33it/s]

    Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 41.74it/s]

    Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  60%|██████    | 35/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  60%|██████    | 35/58 [00:01<00:00, 41.74it/s]

    Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.20it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.20it/s]Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.20it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.20it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.20it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 23.18it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 23.18it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 23.18it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 23.18it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 23.18it/s]

    Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 23.18it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.25it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.25it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.25it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.25it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.25it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.25it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.11it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 29.43it/s]


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
    Generated text:  Hana, and I have a photo album of 500 photos. I like to organize the photos into groups based on their number of likes. How many groups would I have if the 10 most liked photos had 100 likes, the 20 most liked photos had 10 likes, the 30 most liked photos had 1 likes, and the 40 most liked photos had 0 likes?
    
    To determine how many groups Hana has based on the number of likes each photo has, we need to count the number of photos for each category and then sum these counts.
    
    1. **
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 4 inches tall. The vice president is 5 feet 6 inches tall. How much shorter is the vice president than the president in inches?
    
    To determine how much shorter the vice president is than the president in inches, we need to follow these steps:
    
    1. Convert the heights of the president and the vice president from feet and inches to just inches.
    2. Subtract the height of the vice president from the height of the president.
    
    First, let's convert the heights of the president and the vice president from feet and inches to just inches. There are 12 inches in a foot, so we can convert the
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    
    To determine the capital of France, I'll follow these steps:
    
    1. **Identify the capital of France**: The capital of France is Paris.
    
    2. **Verification**: The capital of France is a well-known historical city in Europe. It is the largest city in France and serves as the seat of government, administration, and culture.
    
    Therefore, the capital of France is \boxed{Paris}.
    ===============================
    Prompt: The future of AI is
    Generated text:  in storage, and you need to know the next revolution. The future of AI is in storage, and you need to know the next revolution.
    The future of AI is in storage, and you need to know the next revolution. This is the theme of the #AIWeek 2023 live event, which took place on the 31st of October at the Alibaba Cloud headquarters in Hangzhou, Zhejiang, China. This year’s event was held to coincide with the first International Conference on Machine Learning (ICML) and to coincide with the 20th anniversary of the International Conference on Machine Learning (IC


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], and I have [Number of Years] years of experience in [Field/Industry]. I'm always looking for new opportunities to grow and learn, and I'm always eager to learn more about your company and its products or services. What can you tell me about your background and how it relates to your current role? I'm a [Age], [Gender], [Nationality
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is known for its cuisine, fashion, and art, and is a major economic and cultural center in Europe. It is the largest city in France and the second-largest city in the world by population. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to perform tasks that are currently only possible with human expertise. This could lead to a more human-like experience with AI, as well as a more efficient and effective use of resources.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues
    


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
    Generated text:  [name], and I am a [profession or title]. I come from [country], and I have always had a passion for [specific activity, hobby, or interest]. I am passionate about [why you love this activity, hobby, or interest]. This passion is what drives me to keep learning and improving, and I am always eager to expand my knowledge and skills in [area]. My goal is to [what you enjoy about your profession or title]. I am confident that my dedication and passion for [what you enjoy about your profession or title] will make a positive impact on [your area of interest or community]. I am excited
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly changing, and it is likely to continue to evolve in the following ways:
    
    1. Increased sophistication and automation: With advancements in machine learning algorithms and natural language processing, AI systems are likely to become more sophisticated and automated. This will enable machines to perform tasks that were previously considered too complex or expensive to automate.
    
    2. Enhanced privacy and security: AI systems will continue to become more sophisticated, but they will also become more complex and likely more vulnerable to cyber attacks. As such, it is likely that AI systems will be designed with greater security and privacy features.
    
    3. Integration with other technologies: AI systems will continue to be integrated with


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

    job

     title

     or

     profession

    ].

     I

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    industry

     or

     field

    ].

     I

     have

     always

     [

    mention

     a

     positive

     trait

     or

     accomplishment

     you

     have

     had

    ].

     My

     career

     path

     has

     been

     dedicated

     to

     [

    the

     focus

     of

     your

     current

     job

     or

     profession

    ].

     I

     am

     passionate

     about

     [

    your

     professional

     interest

     or

     area

     of

     expertise

    ],

     and

     I

     am

     always

     eager

     to

     learn

     new

     things

    .

     I

     am

     also

     a

     [

    mention

     a

     hobby

     or

     interest

     in

     your

     life

    ].

     I

     am

     a

     [

    mention

     any

     characteristics

     or

     qualities

     that

     make

     you

     unique

    ]

     and

     I

     believe

     in

     [

    mention

     a

     belief

     or

     value

     you

     hold

     dear

    ].

     I

     strive

     to

     be

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     often

     referred

     to

     as

     "

    The

     City

     of

     Light

    ."

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     charming

     architecture

    ,

     and

     vibrant

     culture

    ,

     making

     it

     the

     most

     popular

     city

     in

     France

     and

     a

     major

     tourist

     destination

    .

     It

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

     and

     many

     other

     iconic

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

     industry

     and

     its

     role

     in

     the

     world

     of

     fashion

     and

     design

    .

     Paris

     is

     a

     city

     that

     has

     been

     significantly

     influenced

     by

     its

     historical

     and

     cultural

     heritage

    ,

     and

     it

     continues

     to

     be

     a

     center

     of

     creativity

    ,

     innovation

    ,

     and

     cultural

     expression

    .

     Its

     status

     as

     the

     capital

     of

     France

     means

     that

     it

     remains

     a

     major

     cultural

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

     with

     many

     potential

     trends

     that

     could

     shape

     its

     direction

    .

     Here

     are

     some

     potential

     trends

     that

     could

     emerge

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     of

     AI

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     machine

     learning

    ,

     and

     deep

     learning

    ,

     we

     could

     see

     a

     greater

     adoption

     of

     AI

     in

     fields

     like

     healthcare

    ,

     transportation

    ,

     and

     education

    .
    


    2

    .

     Improved

     privacy

     and

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     inevitably

     interact

     with

     more

     of

     our

     lives

    .

     We

     need

     to

     ensure

     that

     AI

     is

     designed

     and

     implemented

     in

     a

     way

     that

     respects

     privacy

     and

     ethical

     considerations

    .
    


    3

    .

     Adv

    ancements

     in

     AI

     for

     healthcare

    



```python
llm.shutdown()
```
