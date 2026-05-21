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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]


    2026-05-21 00:32:01,684 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-21 00:32:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.77it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.75it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.87it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 29.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 41.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.57 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.75it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.71it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.71it/s]Capturing num tokens (num_tokens=960 avail_mem=72.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.71it/s] Capturing num tokens (num_tokens=960 avail_mem=72.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.41it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=640 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=640 avail_mem=72.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.73it/s]Capturing num tokens (num_tokens=576 avail_mem=72.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.73it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.73it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.73it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.73it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.73it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.86it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.22it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  71%|███████   | 41/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  71%|███████   | 41/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  71%|███████   | 41/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  71%|███████   | 41/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  71%|███████   | 41/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=112 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.59it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.59it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.42it/s] Capturing num tokens (num_tokens=4 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=4 avail_mem=72.46 GB): 100%|██████████| 58/58 [00:01<00:00, 35.10it/s]


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
    Generated text:  Angelic and I am a writer and creative director from Atlanta. My passion is my communities, and I want to inspire and empower people to grow in their own unique way, with the help of me, your story, and your own creativity and effort. I work hard to create a space where everyone feels welcome and safe. I am a team player and always ready to help others with their unique stories and dreams.
    I grew up in a middle-class family, and my father encouraged me to pursue a career in publishing. I went to college, and I majored in writing and creative writing. I worked in marketing and advertising, and I
    ===============================
    Prompt: The president of the United States is
    Generated text:  now considered to be a member of a special cabinet, which includes the Speaker of the House, the President of the Senate, and the Attorney General. In how many ways can the president and the Speaker of the House be swapped?
    
    To determine the number of ways the president of the United States can be swapped with the Speaker of the House, we need to consider that the president and the Speaker of the House are distinct individuals. Each can be swapped in two different ways (president to Speaker or Speaker to president).
    
    Here is the step-by-step reasoning:
    
    1. Identify the two individuals: The president and the Speaker of the House.
    2
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. London C. Tokyo D. New York D. New York
    
    The capital of France is Paris.
    
    Paris is the capital city of France, located in the Île de la Cité on the western bank of the Seine River. It is known for its stunning architecture, museums, and cultural institutions, as well as its rich history and vibrant nightlife.
    
    To sum up, Paris is the right answer for the capital of France, while the other options are not the capital of France. The correct answer is A. Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  changing how we interact with technology, and how we consume it. But, what does it mean to be an AI agent, or a model for that matter? Well, as we saw in the first part of this blog, it is important to understand that we are talking about a software package that takes in data, performs tasks, and generates outputs, but this isn’t the only way it can be used.
    In this part of the series, we will be discussing what an AI agent is. To fully understand the concept, it is important to first understand what an AI model is.
    In this article, we will be discussing what an AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I've been working here for [number] years. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm always eager to learn and grow, and I'm always looking for opportunities to contribute to the team. I'm a [job title] at [company name], and I'm excited to be here and help make [company name] a success. What's your name? What's your job title? What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and being home to many famous museums and art galleries. Paris is a popular tourist destination, known for its beautiful architecture, vibrant nightlife, and diverse food and drink scene. The city is also home to many international organizations and institutions, including the French Academy of Sciences and the French National Library. Paris is a city of contrasts, with its modern skyscrapers and historic neighborhoods, but it is also a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Increased use of AI for autonomous systems: Autonomous systems, such as drones and self-driving cars, are likely to become more prevalent in the future. AI will be used
    


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
    Generated text:  [insert name]. I'm a [insert profession or role], and I'm excited to be here today. I'm going to share my experiences with you and try to make you laugh. How can I assist you today? [insert some open-ended questions or activities to encourage conversation]. Let's get to know each other better and build a connection! [insert a brief conversation starter or question to start the interaction]. So, what brings you here today? [insert a brief personal anecdote or story that relates to your experience or interests]. This helps me to understand you better, and it makes us both feel welcome. I'm [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the heart of its economic, cultural, and political center. Its population is around 2.1 million people, and it is home to numerous museums, theaters, and other cultural institutions. Paris has a rich history, with many landmarks and monuments including Notre-Dame Cathedral and the Eiffel Tower. The city is also known for its distinctive French culture and cuisine. The French capital is a major hub for global business and is home to many multinational corporations and influential institutions. It is also a world-renowned center for art, music, and literature, with many prestigious museums, theaters,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see significant advancements in several areas, including:
    
    1. Increased automation: AI is increasingly being used to automate repetitive and mundane tasks, leading to increased efficiency and productivity.
    
    2. Personalized AI: AI will be able to learn from user behavior and preferences, providing more accurate and personalized recommendations.
    
    3. Autonomous vehicles: Self-driving cars are already a reality, and AI will continue to improve in this area, reducing the likelihood of human error and increasing safety.
    
    4. Cybersecurity: AI will be used to improve cybersecurity by analyzing large amounts of data to detect and respond to security threats.
    
    5. Smart cities: AI will be used


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

    ].

     I

     am

     an

     aspiring

     writer

     with

     a

     love

     for

     capturing

     the

     essence

     of

     humanity

    .

     I

     have

     always

     been

     fascinated

     by

     the

     human

     condition

     and

     how

     we

     navigate

     the

     world

    .

     I

     am

     passionate

     about

     creating

     compelling

     narratives

     that

     explore

     the

     complexities

     of

     the

     human

     experience

    .

     My

     writing

     has

     been

     recognized

     and

     praised

     for

     its

     depth

     and

     resonance

     with

     the

     human

     experience

    .

     I

     am

     a

     creative

     and

     highly

     intelligent

     individual

     who

     thr

    ives

     on

     the

     challenge

     of

     exploring

     new

     ideas

     and

     creating

     something

     truly

     unique

     and

     captivating

    .

     I

     am

     a

     writer

     who

     is

     always

     striving

     to

     push

     the

     boundaries

     of

     what

     I

     can

     do

     with

     words

     and

     I

     am

     excited

     to

     bring

     my

     stories

     to

     life

    .

     What

     are

     you

     up

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     the

     eighth

    -largest

     country

     in

     the

     world

    .

     The

     city

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

     and

     its

     colorful

     city

    scape

    ,

     which

     is

     famous

     for

     its

     Notre

    -D

    ame

     Cathedral

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     home

     to

     the

     Lou

    vre

     Museum

     and

     the

     French

     Academy

     of

     Sciences

    ,

     and

     is

     a

     major

     center

     for

     art

    ,

     culture

    ,

     and

     politics

     in

     France

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     and

     is

     considered

     a

     city

     of

     living

     history

    ,

     with

     many

     historic

     landmarks

     and

     sites

     to

     explore

    .

     It

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     modern

     architecture

     and

     vibrant

     streets

    ,

     as

     well

     as

     its

     old

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     not

     yet

     clear

    ,

     but

     it

     is

     likely

     to

     continue

     to

     evolve

     rapidly

    .

     Here

     are

     some

     possible

     trends

     that

     experts

     predict

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

    -powered

     diagnostic

     tools

     and

     predictive

     analytics

     can

     help

     healthcare

     providers

     make

     more

     accurate

     diagnoses

     and

     identify

     potential

     health

     issues

     early

    .

     This

     can

     lead

     to

     earlier

     treatment

     and

     better

     patient

     outcomes

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     transportation

    :

     AI

    -powered

     self

    -driving

     cars

     could

     reduce

     traffic

     congestion

     and

     improve

     safety

    .

     Autonomous

     vehicles

     could

     also

     reduce

     the

     environmental

     impact

     of

     transportation

     by

     reducing

     the

     need

     for

     fuel

     and

     emissions

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     finance

    :

     AI

    -powered

     fraud

     detection

     and

     risk

     management

     could

     help

     financial

     institutions

     prevent

     fraud

    



```python
llm.shutdown()
```
