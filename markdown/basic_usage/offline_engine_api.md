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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.65it/s]


    2026-05-15 17:39:55,374 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 17:39:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.50it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s] 

    Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 21.70it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 31.17it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.65 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.57 GB):  31%|███       | 18/58 [00:00<00:01, 33.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.57 GB):  31%|███       | 18/58 [00:00<00:01, 33.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.57 GB):  31%|███       | 18/58 [00:00<00:01, 33.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.55 GB):  31%|███       | 18/58 [00:00<00:01, 33.90it/s]

    Capturing num tokens (num_tokens=960 avail_mem=69.57 GB):  31%|███       | 18/58 [00:00<00:01, 33.90it/s] Capturing num tokens (num_tokens=960 avail_mem=69.57 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=896 avail_mem=69.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=832 avail_mem=69.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=768 avail_mem=69.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=704 avail_mem=69.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=704 avail_mem=69.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.36it/s]Capturing num tokens (num_tokens=640 avail_mem=69.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.36it/s]Capturing num tokens (num_tokens=576 avail_mem=69.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.36it/s]Capturing num tokens (num_tokens=512 avail_mem=69.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.36it/s]Capturing num tokens (num_tokens=480 avail_mem=69.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.36it/s]

    Capturing num tokens (num_tokens=448 avail_mem=69.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.36it/s]Capturing num tokens (num_tokens=448 avail_mem=69.55 GB):  53%|█████▎    | 31/58 [00:00<00:00, 31.85it/s]Capturing num tokens (num_tokens=416 avail_mem=74.13 GB):  53%|█████▎    | 31/58 [00:00<00:00, 31.85it/s]Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.85it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  71%|███████   | 41/58 [00:01<00:00, 39.23it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  71%|███████   | 41/58 [00:01<00:00, 39.23it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  71%|███████   | 41/58 [00:01<00:00, 39.23it/s]Capturing num tokens (num_tokens=144 avail_mem=74.08 GB):  71%|███████   | 41/58 [00:01<00:00, 39.23it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  71%|███████   | 41/58 [00:01<00:00, 39.23it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  71%|███████   | 41/58 [00:01<00:00, 39.23it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.20it/s] Capturing num tokens (num_tokens=80 avail_mem=74.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.37it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.53it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 37.02it/s]


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
    Generated text:  Xan and I'm a programmer. I'm only 21 years old and I'm trying to become a software engineer. I'm a well-liked person with a sense of humor and I have a high school diploma in IT. I have been trying to find a job for the past few years and I am currently working at a startup called SparkCloud. My job description is to assist developers in creating software. The best part about my job is that I get to work with the latest programming languages and software technologies. My current problem is that I am struggling with creating a software system that is not only functional but also user-friendly.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful individual, but in the world of cricket, a senior cricketer is a lesser status. So how can the world cricket organization (WCO) ensure that these members of the cricketer’s support system are at the highest level? 
    
    The WCO has already seen a lot of progress in ensuring that cricketers have access to the best possible training, coaching and support. But there are still some steps that need to be taken to ensure that these individuals are able to perform at their best and meet the high standards that the organization expects of them.
    
    Here are a few things that the organization can do to ensure that
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which of the following statements is correct? (　　)  
    A: It's located in the northern hemisphere, and it's in the western hemisphere.  
    B: It's located in the southern hemisphere, and it's in the eastern hemisphere.  
    C: It's located in the northern hemisphere, and it's in the eastern hemisphere.  
    D: It's located in the southern hemisphere, and it's in the western hemisphere.
    To determine the correct statement about the capital of France, Paris, we need to consider the geographical coordinates of the capital city. The capital city of France is Paris, and it is located in
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon, and the market is now exploring a new way of thinking, and a new way of knowing what a machine can do. While most people think of machines as being static, they are actually open to a variety of different abilities.
    With the shift to the cloud, AI has become an integral part of modern digital businesses. The cloud is used by both businesses and consumers to store, process, and share data. By building the right cloud solutions for your business, you can quickly and easily access this valuable resource.
    One of the biggest differences between cloud and local data centers is the ability to store data on a network, rather than


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


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture, and is a popular tourist destination. It is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a city of contrasts, with its modern and historic elements blending together to create a unique and fascinating place. The city is also home to many important institutions and organizations, including the French Academy of Sciences and the French National Library. Paris is a city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and artificial intelligence: As automation and AI continue to advance, we are likely to see more and more jobs being automated, leading to a shift towards more human-like AI systems. This could result in a more efficient and productive workforce, but it could also lead to job displacement for some workers.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be an increased risk of data
    


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
    Generated text:  [Your Name], and I am a [Job Title] at [Company Name]. I love [Favorite Activity/Interest]. I have always been [What motivates me], and [Your Experience] has helped me achieve this. I am [Age/Location] years old, and [Your Previous Experience/Background] will help me understand the complexities of your role. I have a strong work ethic and a commitment to excellence, and I am always looking for ways to improve my skills and knowledge. I am excited to work with you and help you achieve your goals.
    
    Sure, I can add some more details to make the self-introduction
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the Eiffel Tower stands and the Louvre Museum is housed.
    Paris is the capital city of France. It is the largest city in the country and the seventh-largest urban area in the world. The city is known for its rich history, art, and culture. Paris is also an important financial and political center in Europe, with many of the country's major banks and financial institutions located there. It is also home to many important museums, including the Louvre Museum and the Museum of Modern Art (MoMA). Finally, Paris is a hub for transportation and tourism, with many of the country's major cities
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a combination of several trends that are expected to continue or evolve over the next several years. Here are some of the most likely trends that will shape the AI landscape in the coming years:
    
    1. Increased use of AI in healthcare: AI will continue to play an increasingly important role in healthcare, with applications ranging from personalized medicine to predicting the likelihood of diseases. AI will also be used in research and development, with potential applications in drug discovery and biotechnology.
    
    2. Improved AI safety: As AI systems become more complex and sophisticated, there will be an increasing need for systems that can be more secure and reliable. This


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

    job

     title

     or

     profession

    ]

     who

     has

     always

     been

     passionate

     about

     [

    interest

     or

     hobby

    ].

     I

     enjoy

     exploring

     new

     experiences

     and

     trying

     new

     things

    ,

     and

     I

     believe

     in

     using

     my

     unique

     talents

     to

     make

     the

     world

     a

     better

     place

    .

     Whether

     I

    'm

     helping

     a

     friend

     with

     a

     problem

     or

     watching

     a

     movie

     with

     a

     group

    ,

     I

     always

     try

     to

     bring

     a

     fresh

     perspective

     and

     a

     positive

     outlook

     to

     the

     situation

    .

     I

    'm

     looking

     forward

     to

     learning

     more

     about

     you

     and

     sharing

     my

     stories

     with

     you

    !

     [

    Include

     any

     other

     personal

     details

     or

     background

     information

     about

     yourself

     that

     you

    'd

     like

     to

     include

    .

    ]


    [

    Insert

     a

     brief

     introduction

     to

     the

     reader

    ,

     such

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     and

     the

     oldest

     continuously

     inhabited

     city

     in

     Europe

    ,

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     iconic

     architecture

    ,

     including

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

    .

     
    


    France

    's

     capital

    ,

     Paris

    ,

     is

     a

     bustling

     and

     dynamic

     city

     that

     has

     been

     the

     heart

     of

     French

     culture

    ,

     politics

    ,

     and

     diplomacy

     for

     centuries

    .

     Its

     iconic

     landmarks

    ,

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

     Arc

     de

     Tri

    omp

    he

    ,

     make

     it

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    .

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     the

     time

     of

     the

     Roman

     Empire

    ,

     with

     ancient

     ruins

     and

     monuments

     like

     the

     Col

    os

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     multif

    ac

    eted

    ,

     with

     many

     potential

     trends

     to

     watch

     for

    .

     Here

     are

     a

     few

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Self

    -driving

     cars

     are

     becoming

     more

     and

     more

     prevalent

    ,

     with

     many

     companies

     investing

     heavily

     in

     AI

     for

     this

     technology

    .
    


    2

    .

     Personal

    ized

     healthcare

    :

     AI

     is

     being

     used

     to

     develop

     more

     advanced

     medical

     diagnoses

    ,

     predictive

     analytics

    ,

     and

     treatment

     plans

    .
    


    3

    .

     Virtual

     assistants

    :

     AI

    -powered

     virtual

     assistants

     are

     expected

     to

     become

     more

     ubiquitous

    ,

     with

     more

     assistants

     available

     to

     customers

     for

     a

     wider

     range

     of

     tasks

    .
    


    4

    .

     Machine

     learning

     in

     education

    :

     AI

     is

     being

     used

     to

     personalize

     learning

     experiences

    ,

     providing

     more

     accurate

     assessments

    ,

     and

     improving

     student

     engagement

    .
    


    5

    .

     Autonomous

     crops

    



```python
llm.shutdown()
```
