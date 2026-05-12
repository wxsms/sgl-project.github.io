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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.33it/s]


    2026-05-12 08:34:54,624 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 08:34:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.30it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.49it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 22.49it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 22.49it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 40.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 16.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 16.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 16.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 16.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 19.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 19.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 19.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 19.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.13it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.13it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.35it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.53it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.05it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.05it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.87it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.87it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.87it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.87it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 39.33it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.69it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=112 avail_mem=74.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=96 avail_mem=74.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.63it/s] Capturing num tokens (num_tokens=80 avail_mem=74.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=64 avail_mem=74.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=48 avail_mem=74.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.63it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=32 avail_mem=74.17 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=28 avail_mem=74.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=24 avail_mem=73.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=20 avail_mem=73.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=16 avail_mem=73.66 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=16 avail_mem=73.66 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.72it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.72it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.72it/s] Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.72it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:01<00:00, 34.27it/s]


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
    Generated text:  Sam. I'm 15 years old, and my hobby is playing video games. I also like listening to music. My best friend, Leo, is my best friend too. He's 14 years old, and he also likes playing video games. He also likes to listen to music. What is Sam and Leo's favorite hobby? Let's solve this logic problem:
    Based on the information given, Sam and Leo's favorite hobbies are playing video games and listening to music. Therefore, the answer is video games and music.
    ===============================
    Prompt: The president of the United States is
    Generated text:  34 years older than the president of Brazil, and the president of Brazil is 3 times older than the president of France. If the president of the United States is currently 67 years old, what will be the sum of their ages 50 years from now?
    To determine the sum of the ages of the president of the United States and Brazil 50 years from now, we need to follow these steps:
    
    1. Identify the current ages of the presidents of the United States and Brazil.
    2. Calculate the age of the president of Brazil 50 years from now.
    3. Calculate the age of the president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and this is where the Eiffel Tower is located. This tower was built in 1889, but it was built for the 1889 World’s Fair. The tower stands at 324 meters tall, and it’s the tallest tower in the world.
    The Eiffel Tower is a popular tourist attraction in Paris. Its colorful exterior and iconic silhouette make it a popular tourist destination. The tower is a symbol of Paris and a popular destination for tourists visiting the city.
    The Eiffel Tower is a popular tourist attraction in Paris. Its colorful exterior and iconic silhouette make it a popular tourist
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be different if you ask me. In my personal time, I do not have any specific interests or hobbies, so I can't say if I like or dislike certain fields or topics. However, based on my expertise in AI, I can say that the future of AI is likely to be heavily influenced by the developments in natural language processing (NLP), machine learning, and computer vision.
    
    NLP is a key area of AI that is expected to continue to grow in importance as more and more people interact with AI-driven technologies. NLP involves the ability of machines to understand and generate human language, and it has already been used


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always up for a good challenge and love to explore new experiences. What's your favorite book or movie? I love [insert a short description of your favorite book or movie]. I'm always looking for new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is home to many notable French artists, writers, and composers, and is known for its rich history and cultural heritage. Paris is a vibrant and dynamic city that continues to be a major center of French culture and politics. The city is also home to many international organizations and institutions, including the French Academy of Sciences and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud
    


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
    Generated text:  [Your Name] and I am a [Your Profession] with [Your Educational Background] and [Your Professional Experience]. I am passionate about [Your Interests or hobbies] and enjoy [Your Skills or Strengths]. I am always looking for opportunities to learn and grow, and I am confident in my ability to succeed in any position I choose to take on. Thank you for asking to meet me! Based on the passage above, Can you provide a summary of the character's background and interests? Certainly! Based on the information provided in the passage, the character's background includes:
    
    1. **Name**: [Your Name]
    2.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! The capital of France is Paris, which is the largest and most populous city in the country. It's located in the west of the country and is home to many of France's most famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city also has a rich history dating back thousands of years, with its ancient roots extending all the way back to prehistoric times. Today, Paris is a bustling metropolis with a vibrant culture and a rich mix of French, Italian, and other European influences. It's also a center of business, entertainment,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid technological advancements, integration with other sectors, and a growing emphasis on ethical considerations. Here are some possible trends:
    
    1. Autonomous vehicles: Autonomous vehicles will become increasingly common, with self-driving cars becoming more prevalent in the future. These vehicles will be able to drive on the roads without human intervention, which could lead to significant reductions in traffic accidents and pollution.
    
    2. Enhanced cognitive capabilities: AI will continue to learn and improve, leading to even greater cognitive abilities. This could include new forms of AI that are able to learn from feedback and improve over time, or AI systems that can adapt to new situations or tasks


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

    name

    ].

     I

    'm

     here

     to

     help

     you

     achieve

     your

     goals

     and

     dreams

    .

     What

     can

     I

     do

     for

     you

    ?

     As

     a

     professional

     writer

    ,

     I

     can

     help

     you

     with

     research

    ,

     writing

    ,

     editing

    ,

     proof

    reading

    ,

     and

     marketing

    .

     Whether

     you

    're

     writing

     a

     business

     proposal

    ,

     a

     novel

    ,

     a

     screenplay

    ,

     or

     a

     resume

    ,

     I

     can

     guide

     you

     through

     the

     process

     and

     give

     you

     valuable

     insights

     into

     the

     industry

    .

     Plus

    ,

     I

     offer

     a

     free

     consultation

     to

     help

     you

     identify

     your

     unique

     voice

     and

     style

    .

     So

    ,

     if

     you

    're

     looking

     to

     express

     yourself

     auth

    ent

    ically

     and

     effectively

    ,

     I

    'm

     here

     to

     help

     you

     succeed

    .

     Let

    's

     get

     started

    !

     

    📝

    💼

    💼

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

    -central

     region

     of

     the

     country

     and

     is

     the

     largest

     city

     in

     both

     France

     and

     Europe

    .


    The

     capital

     of

     France

     is

     Paris

    ,

     located

     in

     the

     south

    -central

     region

     of

     the

     country

    .

     It

     is

     the

     largest

     city

     in

     both

     France

     and

     Europe

     and

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

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

     Paris

     also

     plays

     a

     crucial

     role

     in

     French

     culture

    ,

     politics

    ,

     and

     economy

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     a

     hub

     for

     international

     business

     and

     diplomacy

    .

     The

     city

     has

     a

     diverse

     population

     and

     is

     home

     to

     many

     world

    -ren

    owned

     institutions

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     availability

     and

     accessibility

     of

     AI

     systems

    :

     As

     the

     technology

     and

     algorithms

     that

     power

     AI

     systems

     become

     more

     widespread

     and

     accessible

    ,

     we

     may

     see

     a

     more

     widespread

     adoption

     of

     AI

     in

     our

     daily

     lives

    .
    


    2

    .

     Democrat

    ization

     of

     AI

    :

     As

     AI

     systems

     become

     more

     affordable

     and

     accessible

    ,

     more

     people

     may

     become

     interested

     in

     using

     AI

     for

     a

     range

     of

     tasks

    ,

     from

     helping

     with

     tasks

     such

     as

     scheduling

     appointments

     to

     playing

     games

    .
    


    3

    .

     AI

    -driven

     innovation

    :

     AI

     is

     already

     transforming

     many

     industries

    ,

     from

     healthcare

     and

     finance

     to

     retail

     and

     transportation

    .

     As

     AI

     continues

     to

     evolve

    ,

     we

     may

     see

     even

     more

     innovation

     in

     areas

     such

    



```python
llm.shutdown()
```
