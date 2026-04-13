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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.83it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.82it/s]


    2026-04-13 05:25:33,789 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 05:25:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.26it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.18it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.18it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.18it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.18it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.18it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.18it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.18it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.18it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 22.76it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 30.26it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 30.26it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 36.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.16 GB):   2%|▏         | 1/58 [00:00<00:06,  9.09it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.13 GB):   2%|▏         | 1/58 [00:00<00:06,  9.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.12 GB):   2%|▏         | 1/58 [00:00<00:06,  9.09it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=57.12 GB):   5%|▌         | 3/58 [00:00<00:04, 13.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.12 GB):   5%|▌         | 3/58 [00:00<00:04, 13.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.12 GB):   5%|▌         | 3/58 [00:00<00:04, 13.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.12 GB):   5%|▌         | 3/58 [00:00<00:04, 13.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.12 GB):  10%|█         | 6/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.12 GB):  10%|█         | 6/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.11 GB):  10%|█         | 6/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.11 GB):  10%|█         | 6/58 [00:00<00:02, 18.71it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=57.11 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.11 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.10 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.10 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.52it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.10 GB):  21%|██        | 12/58 [00:00<00:02, 17.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.10 GB):  21%|██        | 12/58 [00:00<00:02, 17.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.10 GB):  21%|██        | 12/58 [00:00<00:02, 17.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.10 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.09 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.09 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.08 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.08 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.96it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=57.08 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=960 avail_mem=57.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.38it/s] Capturing num tokens (num_tokens=896 avail_mem=57.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=832 avail_mem=57.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=832 avail_mem=57.03 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=768 avail_mem=57.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=704 avail_mem=57.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]

    Capturing num tokens (num_tokens=640 avail_mem=57.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=576 avail_mem=57.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=576 avail_mem=57.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.56it/s]Capturing num tokens (num_tokens=512 avail_mem=57.01 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.56it/s]Capturing num tokens (num_tokens=480 avail_mem=57.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.56it/s]Capturing num tokens (num_tokens=448 avail_mem=57.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.56it/s]Capturing num tokens (num_tokens=416 avail_mem=57.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.56it/s]Capturing num tokens (num_tokens=384 avail_mem=57.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.56it/s]Capturing num tokens (num_tokens=352 avail_mem=57.01 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.56it/s]Capturing num tokens (num_tokens=352 avail_mem=57.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=320 avail_mem=57.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=288 avail_mem=57.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=256 avail_mem=57.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]

    Capturing num tokens (num_tokens=240 avail_mem=57.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=224 avail_mem=56.99 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=208 avail_mem=56.99 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=208 avail_mem=56.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=192 avail_mem=56.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=176 avail_mem=56.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=160 avail_mem=56.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=144 avail_mem=56.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=128 avail_mem=56.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=128 avail_mem=56.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=112 avail_mem=56.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=96 avail_mem=56.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=56.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=64 avail_mem=56.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=48 avail_mem=56.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=48 avail_mem=56.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=32 avail_mem=56.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=28 avail_mem=56.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.87it/s]

    Capturing num tokens (num_tokens=24 avail_mem=56.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=20 avail_mem=56.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=16 avail_mem=56.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=16 avail_mem=56.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 27.27it/s]Capturing num tokens (num_tokens=12 avail_mem=56.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 27.27it/s]

    Capturing num tokens (num_tokens=8 avail_mem=56.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 27.27it/s] Capturing num tokens (num_tokens=4 avail_mem=56.93 GB):  95%|█████████▍| 55/58 [00:01<00:00, 27.27it/s]Capturing num tokens (num_tokens=4 avail_mem=56.93 GB): 100%|██████████| 58/58 [00:02<00:00, 28.74it/s]


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
    Generated text:  Yuki and I'm a vegetarian who enjoys eating lots of salads. I like to eat salads for breakfast, lunch, dinner, and snacks. Here are some of my favorite vegetarian salads:
    
    1. Tomato and mozzarella salad
    2. Caesar salad
    3. Avocado and goat cheese salad
    4. Hummus and pita bread salad
    5. Kalamata olives and feta cheese salad
    6. Spinach and feta cheese salad
    7. Pesto and cherry tomatoes salad
    
    Please share your favorite vegetarian salad that you enjoy and why you like it so much. How do you incorporate more seasonal vegetables and seasonal
    ===============================
    Prompt: The president of the United States is
    Generated text:  a relatively new office, created in 2004. In the past, it has been filled by the head of the executive branch for three-year terms. Here are some of the key points about the new office:  - It is the fourth most powerful office in the United States government. - The presidential candidate must choose a candidate before the election begins. - The next president of the United States will be chosen by a simple majority, and must be a natural born citizen, or a U.S. citizen who has been a U.S. citizen for at least 14 years. - The office is a remote one, and
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Lyon
    C. Lille
    D. Tours
    Answer:
    A
    
    The capital of France is ____
    A. Paris
    B. Lyon
    C. Lille
    D. Tours
    Answer:
    A
    
    The capital of France is ____
    A. Paris
    B. Lyon
    C. Lille
    D. Tours
    Answer:
    A
    
    In the following sentences, the one where the rhetorical device is different from the other three is ____
    A. The mountains are silent, the clouds are still, the river is still, the village is still, and the horses are still.
    B
    ===============================
    Prompt: The future of AI is
    Generated text:  here. With the development of AI, people can create new products and services. If we continue to use AI to solve problems, the new product and service we create may be more powerful than the products and services we currently have. In the future, the competition between different companies is more intense. They will be able to create products and services that no other company can create. As a result, companies will have to invest more in research and development. This will have a huge impact on the economy, as companies will be forced to invest more money in R&D to stay competitive. On the other hand, companies that have fallen behind in research


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your character or profession]. What do you like to do? I enjoy [insert a short description of your hobbies or interests]. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests]. What do you like to do when you're not working? I like to [insert a short description of your hobbies or interests]. What do you like to do when you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its cuisine, fashion, and art scene. It is also home to many international organizations and institutions. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. Its status as the capital of France has made it a major hub for international affairs and diplomacy. The city is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks, from simple tasks like language translation to complex tasks like autonomous driving and medical diagnosis. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption of AI in areas such as healthcare, finance, and transportation. Additionally, AI will continue to be used for research and development, with the goal of advancing our understanding of the world and developing new technologies that will have a significant impact on
    


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
    Generated text:  [Your Name]. I'm a [Your occupation] from [Your hometown/college]. I enjoy [Your hobbies, interests, or hobbies]. I'm [Your age] years old. And I'm [Your personality type]. To sum up, I am [Your character type]. What brings you to this place? I'm a/an [Your character type] who has come to [Your character type's character type] and [Your character type's character type's character type]. I have no romantic interest in you. I don't have plans or goals for this place. I'm here just to [Your character type's character type
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower, bustling street life, and rich culture.
    Paris, officially the City of Paris, is the capital and largest city of France. It is renowned for its iconic Eiffel Tower, bustling streets, and rich cultural scene. The city is home to many renowned institutions and landmarks, including the Louvre and Notre-Dame Cathedral, and is a significant cultural and economic hub in France. Its significance as the capital of France has made it a global tourist destination and a cultural melting pot, drawing people from all over the world to experience its unique blend of old-world charm and modernity. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field with many possibilities and possibilities. Here are some possible trends to look out for:
    
    1. Increased Integration of AI with Other Technologies: AI is already making an impact on other technologies like medical imaging, facial recognition, and language translation. With the integration of AI with more technologies, we can expect even more innovative applications to emerge.
    
    2. Enhanced Privacy and Security: As AI becomes more prevalent in our lives, it's important to ensure that data is protected and privacy is maintained. This includes measures such as data anonymization, encryption, and the use of AI to monitor and track individuals.
    
    3. AI in Healthcare: AI


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

     a

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

     am

     [

    Age

    ]

     years

     old

    ,

     and

     my

     love

     for

     learning

     and

     innovation

     always

     drives

     me

     to

     keep

     up

     with

     the

     latest

     trends

     and

     technologies

    .

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     up

     for

     new

     challenges

    .

     I

     enjoy

     sharing

     my

     knowledge

     and

     insights

    ,

     and

     I

    'm

     committed

     to

     using

     my

     skills

     to

     help

     people

     achieve

     their

     goals

    .

     In

     my

     spare

     time

    ,

     I

     enjoy

     reading

    ,

     traveling

    ,

     and

     spending

     time

     with

     my

     family

    .

     Thank

     you

     for

     taking

     the

     time

     to

     learn

     about

     me

    .

     Let

     me

     know

     if

     you

     have

     any

     questions

    !

     [

    Name

    ]

     [

    Date

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     known

     for

     its

     historic

     landmarks

    ,

     rich

     culture

    ,

     and

     vibrant

     nightlife

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     The

     city

     is

     also

     home

     to

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     which

     are

     some

     of

     France

    's

     most

     famous

     attractions

    .

     Paris

     is

     a

     city

     that

     is

     constantly

     evolving

     with

     new

     cultural

     and

     artistic

     developments

    ,

     as

     well

     as

     political

     changes

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     contrasts

     and

     breathtaking

     beauty

     that

     is

     highly

     regarded

     as

     a

     world

    -ren

    owned

     met

    ropolis

    .

     Based

     on

     the

     passage

     you

     just

     heard

    ,

     what

     was

     one

     fact

     that

     could

     be

     added

     to

     make

     the

     statement

     more

     specific

    ?

     Paris

     is

     a

     world

    -ren

    owned

     capital

     city

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

     and

     there

     are

     many

     potential

     trends

     that

     could

     shape

     the

     direction

     of

     the

     field

    .

     Some

     of

     the

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     increasingly

     being

     used

     in

     healthcare

     to

     help

     diagnose

     and

     treat

     diseases

    ,

     predict

     patient

     outcomes

    ,

     and

     improve

     patient

     care

    .

     The

     use

     of

     AI

     in

     healthcare

     will

     likely

     continue

     to

     grow

     as

     more

     data

     and

     patient

     data

     become

     available

    .
    


    2

    .

     Autonomous

     vehicles

    :

     With

     the

     increasing

     use

     of

     AI

     in

     transportation

    ,

     autonomous

     vehicles

     (

    AV

    s

    )

     are

     becoming

     more

     common

    .

     AI

    -powered

     cars

     can

     learn

     from

     traffic

     patterns

     and

     other

     driving

     data

    ,

     making

     them

     safer

     and

     more

     efficient

    .
    


    3

    .

     AI

     in

    



```python
llm.shutdown()
```
