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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.15it/s]


    2026-04-11 00:57:05,855 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 00:57:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:24,  2.53s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:24,  2.53s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:24,  2.53s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:24,  2.53s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.03it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.54it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 20.92it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 20.92it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 20.92it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 20.92it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.92it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.92it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.92it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.92it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.92it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.82it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.82it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.82it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.82it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.82it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.82it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.82it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.82it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 35.41it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 35.41it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 35.41it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 35.41it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 35.41it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 35.41it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 35.41it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 35.41it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 41.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.01 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.98 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.98 GB):   3%|▎         | 2/58 [00:00<00:03, 14.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.98 GB):   3%|▎         | 2/58 [00:00<00:03, 14.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.98 GB):   3%|▎         | 2/58 [00:00<00:03, 14.53it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.98 GB):   3%|▎         | 2/58 [00:00<00:03, 14.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.98 GB):   9%|▊         | 5/58 [00:00<00:03, 13.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 13.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 13.85it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 13.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 13.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.33it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.08it/s]Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.08it/s] Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.08it/s]Capturing num tokens (num_tokens=832 avail_mem=72.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.08it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.08it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.08it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=640 avail_mem=72.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.46it/s]

    Capturing num tokens (num_tokens=576 avail_mem=72.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.18it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.18it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.18it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.18it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.18it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  55%|█████▌    | 32/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=240 avail_mem=72.50 GB):  55%|█████▌    | 32/58 [00:01<00:00, 45.18it/s]

    Capturing num tokens (num_tokens=240 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.85it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.85it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.85it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.85it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.85it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.85it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.85it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.00it/s]Capturing num tokens (num_tokens=32 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.00it/s]Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.00it/s]Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.00it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.00it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.00it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.00it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  97%|█████████▋| 56/58 [00:01<00:00, 52.01it/s]Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  97%|█████████▋| 56/58 [00:01<00:00, 52.01it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  97%|█████████▋| 56/58 [00:01<00:00, 52.01it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 41.17it/s]


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
    Generated text:  Sophie and I am a 21-year-old college student who loves spending time with my friends. I love to cook and cooking is a hobby of mine. I am a vegetarian and I do not eat meat. I am a fan of flavorful and complex recipes. I love to make desserts such as fruit tart and chocolate cake. I have been trying to create a more sustainable and eco-friendly living, so I decided to start a baking blog called "Green Baking" where I share my delicious recipes. My first recipe is a banana and raspberry cake. I hope you enjoy my baking efforts and if you have any questions, please don
    ===============================
    Prompt: The president of the United States is
    Generated text:  in New York. He took a train to Washington D. C. and spent the first day in Washington D. C. The president got off the train at the Capitol and headed to the White House. On the second day, he got off the train at the White House and headed to the Oval Office. On the third day, he got off the train at the Oval Office and headed to the White House. 
    
    How many different routes can the president take from the Capitol to the White House? To determine the number of different routes the president can take from the Capitol to the White House, we need to consider the sequence of stops he
    ===============================
    Prompt: The capital of France is
    Generated text: 
    
    1. Paris
    2. Lyon
    3. Marseille
    4. Nice
    5. Toulouse
    
    1. Paris
    2. Lyon
    3. Marseille
    4. Nice
    5. Toulouse
    
    1. Paris
    2. Lyon
    3. Marseille
    4. Nice
    5. Toulouse
    
    1. Paris
    2. Lyon
    3. Marseille
    4. Nice
    5. Toulouse
    
    1. Paris
    2. Lyon
    3. Marseille
    4. Nice
    5. Toulouse
    
    1. Paris
    2. Lyon
    3. Marseille
    4. Nice
    5. Toulouse
    
    1
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be an AI universe
    
    Hannah Burnett
    Mar 28, 2019
    
    As the world's most valuable software company, Google is a giant on the horizon. In just a few years, Google will be a world leader, more than any other company.
    
    It's a dream come true, but it's also a daunting one. We see the future of Google as being very different from what it is today. And it's going to be different from what it was yesterday. What's happening to Google is happening to the rest of the world.
    
    From a cultural perspective, Google is far more than a search


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" or "La Ville de Paris, la capitale de l'Europe". It is the largest city in France and the second-largest city in the European Union, with a population of over 10 million people. Paris is known for its rich history, art, and culture, and is a major tourist destination. It is also home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city is also known for its fashion industry, with many famous designers and boutiques located in the city. Paris is a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and cost-effective solutions, but it could also lead to job displacement for some workers.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even
    


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
    Generated text:  [Name], and I am a [Age] year old [Occupation/Role]. I love to [Favorite Activity/Activity/Interest]. I also enjoy [Favorite Hobby], which helps me stay [Hungry]. I like [Favorite Book/Artwork/Theme], which inspires me to [Favorite Personality/Personality]. I have [Favorite Experiential Activity], which is [Favorite Hobby], which allows me to [Favorite Skill or Ability]. I'm excited to have the opportunity to meet you and learn more about you! [Character's Name] loves to [Favorite Activity/Activity/Interest].
    [Character's Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest and most populous city in France, located on the Seine River in the Loire Valley region. It was founded in the 6th century by the Roman Empire and has been the seat of government and culture for over 1,000 years. It is the heart of France, a UNESCO World Heritage site and a cultural and artistic hub. Paris is also the second largest city in the European Union and plays a significant role in the economy and tourism industry. As of 2021, the population of the city is approximately 2.7 million. It is also known as the "
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and growing rapidly, with many possible trends that could shape the landscape of the technology industry. Here are some of the most likely areas of development:
    
    1. Increased autonomy: As AI becomes more advanced, we may see more autonomous systems that can make decisions and take actions without human intervention.
    
    2. Improved natural language processing: With the continued development of AI, we may see even more sophisticated natural language processing algorithms that can understand and interpret language in new and complex ways.
    
    3. Greater reliance on AI in industries: As AI becomes more integrated into various industries, we may see more widespread adoption of AI technologies, with more companies and governments investing


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

     [

    Age

    ].

     I

    'm

     a

     [

    Occup

    ation

    ]

     who

     enjoy

     [

    Occup

    ation

    ].

     I

     enjoy

     [

    Occup

    ation

    ]

     as

     much

     as

     I

     love

     [

    Occup

    ation

    ].

     I

    'm

     looking

     forward

     to

     [

    Future

     Event

    ],

     [

    Future

     Event

    ]

     and

     [

    Future

     Event

    ],

     and

     I

    'm

     eagerly

     awaiting

     your

     introduction

     to

     me

    .

     What

     is

     your

     name

    ,

     age

    ,

     and

     what

     are

     you

     interested

     in

    ?

     [

    Name

    ]

     [

    Age

    ]

     [

    Occup

    ation

    ]

     Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     [

    Age

    ].

     I

    'm

     a

     [

    Occup

    ation

    ]

     who

     enjoy

     [

    Occup

    ation

    ].

     I

     enjoy

     [

    Occup

    ation

    ]

     as

     much

     as

     I

     love

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    To

     elaborate

    ,

     Paris

     is

     the

     largest

     city

     in

     France

    ,

     located

     on

     the

     western

     bank

     of

     the

     Se

    ine

     River

    ,

     at

     the

     mouth

     of

     the

     River

     Se

    ine

    .

     It

     is

     the

     seat

     of

     the

     Government

     and

     has

     a

     population

     of

     over

     

    1

    .

     

    5

     million

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     cultural

     scene

    ,

     and

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

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     one

     of

     the

     world

    's

     most

     important

     financial

     centers

     and

     a

     major

     tourist

     destination

    .

     Paris

     is

     considered

     a

     major

     cultural

     and

     political

     center

     in

     Europe

     and

     is

     known

     for

     its

     fashion

    ,

     art

    ,

     music

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     increasingly

     complex

    ,

     with

     many

     new

     trends

     and

     innovations

     emerging

     in

     the

     years

     ahead

    .

     Here

     are

     some

     of

     the

     key

     trends

     we

     can

     expect

     to

     see

     in

     the

     field

     of

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     use

     of

     machine

     learning

     and

     deep

     learning

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     and

     more

     advanced

     forms

     of

     machine

     learning

     and

     deep

     learning

    ,

     which

     can

     help

     us

     solve

     more

     complex

     problems

     and

     develop

     new

     technologies

    .
    


    2

    .

     Greater

     reliance

     on

     AI

     in

     healthcare

    :

     With

     the

     increasing

     use

     of

     AI

     in

     various

     fields

    ,

     including

     healthcare

    ,

     we

     can

     expect

     to

     see

     more

     and

     more

     AI

    -based

     applications

     that

     can

     help

     doctors

     make

     better

     diagnoses

    ,

     improve

    



```python
llm.shutdown()
```
