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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.05it/s]


    2026-05-07 18:02:18,441 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 18:02:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.15it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:02, 14.33it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.33it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:05<00:02, 14.33it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 22.46it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 31.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.60it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.60it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.60it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.60it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.60it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.60it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.18it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  71%|███████   | 41/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 47.46it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.03it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.03it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.03it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.03it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.03it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.03it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.17it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.17it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.17it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.17it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.17it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.17it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.69it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.69it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.69it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 42.75it/s]


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
    Generated text:  Jake and I am a software engineer working on a project with my team. One of our new requirements is to have a multi-user feature within our application. We've been working on it for a few months now and we've been able to get some progress, but we're currently stuck on the next step. I am currently in the middle of writing a unit test for our new feature. However, I've been stuck on one of the unit tests for a while and I am having a hard time coming up with a solution. I have a variable that we will be testing, and I have a method that will be called with this variable
    ===============================
    Prompt: The president of the United States is
    Generated text:  now considered an international affairs chief and does not lead the military. In what year did the United States begin the practice of nominating a president to become the leader of the military?
    The answer to the trivia question "What does the President of the United States do now? " is military. The answer can be found by searching the web. The answer is Secretary of Defense.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. On the first day of January 2017, an economist wrote an article in the New York Times called, “The Perfect City, Paris.” The article covered a number of aspects of the city. The following are some of the things the article contained:
    
      1. Paris is the perfect city because it offers the highest quality of life in the world.
      2. To achieve a perfect city, the following must be done: increase the standard of living and the average income.
      3. A country with a high standard of living and average income is almost certainly a perfect city.
    
    Based on the above
    ===============================
    Prompt: The future of AI is
    Generated text:  highly promising, but it's also complex and challenging. To help you understand more about the topic, let's break down some of the key terms and concepts in the field of AI.
    1. Machine Learning: Machine Learning is a subset of AI that involves using algorithms to allow computers to learn and improve through experience. It uses statistical techniques to identify patterns and make predictions based on data.
    2. Deep Learning: Deep Learning is a subset of Machine Learning that involves using neural networks with multiple layers of computation to learn patterns and features. It is particularly well-suited for tasks that require high-level reasoning and image recognition.
    3. Neural Networks:


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [job title] at [company name], and I'm always looking for ways to [describe your job or passion]. I'm always eager to learn and grow, and I'm always looking for new experiences and opportunities to grow. I'm a [describe your personality or character traits] person. I'm always looking for ways to [describe your goals or aspirations].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and music. Paris is a cultural and historical center that attracts millions of visitors each year. It is a major transportation hub and a major financial center. The city is home to many world-renowned universities and research institutions. Paris is a city of contrasts, with its rich history and modernity. It is a city of art
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as the Internet of Things (IoT), blockchain, and machine learning. This integration could lead to new applications and services that are impossible to imagine today.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with other technologies, there will be a greater emphasis on ethical considerations. This could lead to new regulations and standards that are designed to ensure that AI is used in a responsible and ethical
    


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
    Generated text:  [Name] and I'm a/an [career] at [company] in [location]. I started [job title] as a [degree] in [area of study] at [school name] in [city, state, or country]. I am a lifelong learner and always aim to keep up with the latest [technology, trends, or skills]. I have a passion for [why you love your job], and I'm always striving to make the workplace a better place for everyone involved. I am a hardworking and responsible individual who takes pride in my work and always strives to excel. I am confident and open-minded, and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    (Credit: Wikipedia - Public domain)
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and it is difficult to predict with certainty what will happen. However, some possible trends that are currently being explored and discussed in the industry include:
    
    1. Self-awareness: AI is already capable of self-awareness, and it is likely that in the future, this capability will increase. This could lead to the creation of intelligent machines that are self-directed, self-regulated, and self-reflective.
    
    2. Ethics and accountability: As AI becomes more integrated into our lives, there will be increasing pressure to address the ethical implications of its use. This could lead to the development of new ethical standards for AI, such as


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

     at

     [

    Company

    ],

     and

     I

     specialize

     in

     [

    my

     area

     of

     expertise

    ].

     I

     am

     a

     friendly

    ,

     reliable

    ,

     and

     always

     willing

     to

     help

     my

     colleagues

    .

     I

     have

     a

     track

     record

     of

     being

     proactive

    ,

     detail

    -oriented

    ,

     and

     always

     ready

     to

     meet

     deadlines

    .

     I

     am

     easy

     to

     work

     with

    ,

     as

     I

     am

     always

     calm

     and

     composed

     even

     in

     difficult

     situations

    .

     I

     have

     a

     great

     sense

     of

     humor

     and

     am

     always

     able

     to

     lighten

     the

     mood

    .

     Overall

    ,

     I

     am

     a

     valuable

     member

     of

     the

     team

     who

     will

     help

     my

     colleagues

     and

     the

     company

     achieve

     their

     goals

    .

     
    


    I

     am

     [

    Occup

    ation

    ]

     at

     [

    Company

    ]

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

    ,

     officially

     known

     as

     the

     City

     of

     Light

    ,

     is

     the

     capital

     and

     largest

     city

     of

     France

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     home

     to

     the

     French

     Parliament

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     many

     world

    -ren

    owned

     cultural

     institutions

     and

     landmarks

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

     and

     continues

     to

     be

     an

     important

     cultural

     and

     commercial

     hub

     for

     France

    .

     Paris

     is

     known

     for

     its

     vibrant

     arts

     scene

    ,

     delicious

     cuisine

    ,

     and

     breathtaking

     architecture

    .

     It

     has

     been

     described

     as

     "

    the

     city

     of

     love

    "

     and

     "

    the

     city

     of

     dreams

    ."

     The

     city

     is

     also

     home

     to

     some

     of

     the

     world

    's

     most

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     evolve

     rapidly

    ,

     driven

     by

     new

     technologies

     and

     applications

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     AI

     landscape

    :
    


    1

    .

     Enhanced

     privacy

     and

     data

     security

    :

     As

     AI

     becomes

     more

     integrated

     into

     everyday

     life

    ,

     there

     is

     a

     growing

     need

     for

     greater

     protection

     of

     data

     privacy

     and

     security

    .

     This

     could

     involve

     the

     development

     of

     new

     technologies

     that

     can

     enhance

     privacy

    ,

     such

     as

     deep

     learning

     and

     natural

     language

     processing

    .

     It

     could

     also

     involve

     the

     development

     of

     new

     regulations

     to

     ensure

     that

     AI

     systems

     are

     used

     eth

    ically

     and

     responsibly

    .
    


    2

    .

     Increased

     automation

     and

     specialization

    :

     As

     AI

     systems

     become

     more

     capable

    ,

     there

     will

     be

     an

     increased

     demand

     for

     automation

     and

     specialization

    .

     This

     could

     lead

    



```python
llm.shutdown()
```
