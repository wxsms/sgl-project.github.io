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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.11it/s]


    2026-05-14 04:22:55,098 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 04:22:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:51,  1.05it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:51,  1.05it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:51,  1.05it/s]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:05<00:51,  1.05it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:23,  2.17it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:23,  2.17it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:23,  2.17it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:23,  2.17it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:05<00:23,  2.17it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:05<00:23,  2.17it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:09,  4.64it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:09,  4.64it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:09,  4.64it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:09,  4.64it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:09,  4.64it/s]

    Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:09,  4.64it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:09,  4.64it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:04,  8.46it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:04,  8.46it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:04,  8.46it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:04,  8.46it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:04,  8.46it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:04,  8.46it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 12.18it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 12.18it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 12.18it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 12.18it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 12.18it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 12.18it/s]

    Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 12.18it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 17.49it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 17.49it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 17.49it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 17.49it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 17.49it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 17.49it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 17.49it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 23.37it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 31.07it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s] 

    Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 49.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.08 GB):   3%|▎         | 2/58 [00:00<00:03, 15.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.07 GB):   3%|▎         | 2/58 [00:00<00:03, 15.01it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.07 GB):   3%|▎         | 2/58 [00:00<00:03, 15.01it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.07 GB):   7%|▋         | 4/58 [00:00<00:03, 16.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.07 GB):   7%|▋         | 4/58 [00:00<00:03, 16.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.06 GB):   7%|▋         | 4/58 [00:00<00:03, 16.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.04 GB):   7%|▋         | 4/58 [00:00<00:03, 16.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.04 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.55it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.03 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.01 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.01 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.68it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  31%|███       | 18/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  31%|███       | 18/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.98 GB):  31%|███       | 18/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.96 GB):  31%|███       | 18/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=960 avail_mem=70.97 GB):  31%|███       | 18/58 [00:00<00:01, 30.78it/s] Capturing num tokens (num_tokens=896 avail_mem=70.97 GB):  31%|███       | 18/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=896 avail_mem=70.97 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.85it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.85it/s]Capturing num tokens (num_tokens=768 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.85it/s]

    Capturing num tokens (num_tokens=704 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.85it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.85it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.85it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=512 avail_mem=70.94 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 37.25it/s]Capturing num tokens (num_tokens=384 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 37.25it/s]Capturing num tokens (num_tokens=384 avail_mem=70.95 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.13it/s]

    Capturing num tokens (num_tokens=320 avail_mem=70.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=240 avail_mem=70.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=240 avail_mem=70.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.98it/s]

    Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.92it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.92it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.92it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.92it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.92it/s]

    Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.63it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=64 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=16 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.79it/s]

    Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.70it/s] Capturing num tokens (num_tokens=4 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=4 avail_mem=70.87 GB): 100%|██████████| 58/58 [00:01<00:00, 32.68it/s]


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
    Generated text:  Allen and I'm a 27 year old man with a strong interest in natural and organic products. I currently live in the Sydney area. I spend a lot of my free time exploring the outdoors, reading and trying new recipes. I'm currently working on my Master of Science in sustainable living at the University of Sydney.
    
    What are some of your favorite recipes or foods to try? As an AI language model, I don't have personal preferences or experiences, but I can suggest some popular and delicious recipes that are typically enjoyed by many people:
    
    1. Greek yogurt with honey and berries
    
    2. Cottage cheese with fruits and nuts
    
    3
    ===============================
    Prompt: The president of the United States is
    Generated text:  retiring and has 100 employees, including the president, who are all in the same department. Each employee has a different salary. The president's salary is $100,000. The remaining employees have salaries that are either $50,000 or $70,000. The president's salary must be the highest possible, and the salaries of the remaining employees must be in ascending order and cannot be consecutive. What is the maximum salary that the president can have?
    To determine the maximum possible salary for the president, we need to ensure that the salaries of the remaining employees are in ascending
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and the third largest city in the United Kingdom. It is located on the left bank of the Seine river, in the north-central part of the French department of Paris. Paris is the cultural capital of France. It is also the second most populous city in the European Union after London. Paris is a major international financial centre and a major hub of international trade. It is a cultural and historical centre, the birthplace of many historical figures such as Charles, Napoleon, Victor Hugo, and Alexandre Dumas. It also has a number of famous landmarks, including the Eiffel Tower
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it is about to change the world in a way that is going to profoundly impact us all. This guide will help you understand the impact of AI on education, healthcare, transportation, manufacturing, and the environment. We'll explore the challenges of incorporating AI into these areas and look ahead to the future of AI in education, healthcare, transportation, manufacturing, and the environment. We'll discuss the strengths and weaknesses of AI in education and healthcare, and examine the ethical, legal, and societal impacts of AI in education and healthcare. Finally, we'll discuss the ways in which AI is changing the workforce and how we can prepare for


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French Parliament House. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the Renaissance. It is a major transportation hub and a major tourist destination. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. Paris is also home to the French Academy of Sciences, the French National Library, and the Louvre Museum. It is a city that is steeped in history and culture, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we are likely to see more widespread adoption of AI technologies. This could include things like smart home devices, self-driving cars, and virtual assistants like Siri or Alexa.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This could include things like ensuring that AI
    


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
    Generated text:  [insert name], and I am a [insert profession or title] with a passion for [insert what you enjoy doing or the field you're most interested in]. I love to explore new places, learn new things, and make new friends. My [insert hobby or interest] is [insert what it is]. What makes you unique and what do you like to do? Please include a [insert activity or hobby] that you do on a regular basis and that reflects your personality. Good luck with your self-introduction!
    Hello, my name is [insert name], and I am a [insert profession or title] with a passion for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city in France, and also one of the most important cultural, economic, and political centers in the country. 
    
    Paris, known as "La Roche", is located on the Seine River and has been a major center of French culture for more than 1,000 years. Its historic center, called the Left Bank, was built in the 13th century and is home to the Louvre Museum, the most famous art museum in the world. The city also has a rich history in music, with several major orchestras and a famous opera house, the Théâtre
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by increasing sophistication, automation, and integration with other technologies. Here are some of the possible future trends in AI:
    
    1. Increased automation and AI-powered decision-making: As AI becomes more sophisticated, it is likely to become more effective at making decisions, providing better service, and improving efficiency in various industries. Automation will continue to be used for repetitive tasks, such as data entry, processing and reporting, and routine maintenance.
    
    2. AI will continue to power more complex applications: AI will continue to be used in more complex applications, such as medical diagnosis, fraud detection, and personalized recommendations. AI will also be used in


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

    ].

     I

     come

     from

     [

    Home

     Country

    ].

     I

     graduated

     from

     [

    University

    ].

     What

     kind

     of

     character

     are

     you

    ?
    


    Hello

    ,

     my

     name

     is

     [

    Name

    ].

     I

     come

     from

     [

    Home

     Country

    ].

     I

     graduated

     from

     [

    University

    ].

     What

     kind

     of

     character

     are

     you

    ?

     Well

    ,

     I

    'm

    ...

     well

    ,

     I

    'm

     just

     a

     regular

     person

    .

     Just

     someone

     who

     has

     been

     through

     life

     and

     has

     learned

     a

     few

     lessons

     along

     the

     way

    .

     I

    'm

     here

     to

     help

     people

    ,

     and

     I

    'm

     here

     to

     make

     a

     difference

     in

     their

     lives

    .

     How

     do

     you

     feel

     about

     that

    ?
    


    Hello

    ,

     my

     name

     is

     [

    Name

    ].

     I

     come

     from

     [

    Home

     Country

    ].

     I

     graduated

     from

     [

    University

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     most

     populous

     city

     in

     the

     European

     Union

     and

     the

     second

    -largest

     city

     in

     the

     world

     by

     population

    .

     Paris

     is

     located

     in

     the

     Mos

    elle

     and

     Se

    ine

     river

     valleys

     in

     southwestern

     France

    .

     It

     is

     home

     to

     the

     French

     Parliament

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     numerous

     other

     historic

     sites

     and

     landmarks

    .

     The

     city

     is

     known

     for

     its

     rich

     cultural

     heritage

    ,

     including

     its

     museums

    ,

     art

     galleries

    ,

     and

     the

     Op

    éra

     Garn

    ier

     opera

     house

    .

     Paris

     is

     also

     renowned

     for

     its

     fashion

     industry

    ,

     food

    ,

     and

     its

     distinctive

     French

     cuisine

    ,

     which

     is

     influenced

     by

     the

     French

     colonial

     experience

     in

     North

     America

     and

     Europe

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

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

     industry

     in

     the

     coming

     years

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     could

     be

     expected

     to

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

     with

     the

     physical

     world

    :

     With

     the

     rapid

     development

     of

     machine

     learning

     and

     artificial

     intelligence

    ,

     it

    's

     becoming

     increasingly

     common

     to

     see

     AI

     systems

     interacting

     directly

     with

     the

     physical

     world

    ,

     such

     as

     robots

     and

     drones

    .

     This

     could

     lead

     to

     a

     more

     integrated

     and

     pervasive

     nature

     of

     AI

     in

     the

     physical

     world

    ,

     with

     systems

     that

     can

     learn

     and

     adapt

     to

     their

     environment

    .
    


    2

    .

     Personal

    ized

     experiences

    :

     With

     the

     increasing

     availability

     of

     AI

    -powered

     assistants

     and

     chat

    bots

    ,

     it

    's

     likely

    



```python
llm.shutdown()
```
