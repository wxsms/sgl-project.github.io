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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]


    2026-05-11 20:13:30,616 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 20:13:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:44,  3.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:44,  3.95s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:44,  3.95s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:44,  3.95s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:44,  3.95s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.77it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 25.46it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.22it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.22it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.22it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.22it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.98it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.98it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.98it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.98it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.98it/s]

    Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.98it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  60%|██████    | 35/58 [00:00<00:00, 46.42it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  60%|██████    | 35/58 [00:00<00:00, 46.42it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  60%|██████    | 35/58 [00:00<00:00, 46.42it/s]Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:00<00:00, 46.42it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  60%|██████    | 35/58 [00:00<00:00, 46.42it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  60%|██████    | 35/58 [00:00<00:00, 46.42it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.59it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.98it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.75it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.67it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 42.02it/s]


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
    Generated text:  Aisha and I am a 9th grader from 9th Street Middle School in Portland, OR. I am in the 6th grade and have been working hard at school, taking a lot of tests, and following the school's assignments. I have been learning math and science at school and I have been doing well. I have been to the doctor, and all of the reports are in good shape, and my friends are positive of me.
    I have been working on the math club and have had a lot of fun with my fellow students. I have been working on some science experiments and I have read some books and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office. The president of the United States is elected for a six-year term, which ends on January 20, 2024. It is currently the 44th president of the United States, and they are the 20th president to serve two consecutive terms. If the president is elected every four years, how many years will it take for the next president to be elected? To determine how many years it will take for the next president to be elected, we need to follow these steps:
    
    1. Identify the current year in which the next president will be elected.
    2. Calculate the number of
    ===============================
    Prompt: The capital of France is
    Generated text:  (　　)  
    A: Paris  
    B: London  
    C: Rome  
    D: Berlin
    
    To determine the capital of France, we need to recall the standard name for the capital of France. The capital of France is Paris.
    
    Let's break it down step by step:
    
    1. Identify the capital of France: The capital of France is Paris.
    2. Recognize that the capital is simply the name given to the country's capital.
    
    Therefore, the capital of France is Paris.
    
    The correct choice is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  here. But do we know if it will be beneficial to society? This is the question we need to answer by focusing on a few key areas. The first area is the technology itself. The latest innovations in AI are making progress in recognizing the scope of a wide range of problems, from healthcare to finance to transportation. These advancements will enable more efficient and effective solutions to the most significant challenges of our time. Additionally, as these technologies become more advanced, they will also make them easier to use and integrate, reducing the time and effort required to solve complex problems. The second area is the social and economic impact of AI. Many people are


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working in this field for [number of years] years. I am passionate about [reason for interest in the field]. I enjoy [reason for interest in the field]. I am always looking for [reason for interest in the field]. I am [age] years old. I am [gender] and I am [race]. I am [nationality]. I am [language]. I am [religion]. I am [political affiliation]. I am [interests]. I am [hobbies]. I am [personal qualities]. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. It is also known for its fashion industry and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a city that has a unique blend of old-world charm and modernity, making it a popular destination for tourists and locals alike. The city is also known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will lead to increased efficiency, productivity, and cost savings for businesses.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more sophisticated applications in healthcare, such as personalized medicine, drug discovery, and predictive
    


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
    Generated text:  [Name], and I work for [Company]. I'm excited to be here today and contribute to the team. What can you tell me about yourself? As an [insert 1-2 words about your profession], I've been working in [insert 1-2 words about your role]. I'm always looking to learn new things and help others, so I'm excited to share my knowledge with you. What can you tell me about your career? As a [insert 1-2 words about your profession], I've been working in [insert 1-2 words about your role]. I'm always looking to learn new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest city in the country and its political, economic, and cultural center. It was founded in 789 AD by Charlemagne and has become an international city with a rich cultural heritage and a high standard of living. Paris has many famous landmarks such as Notre Dame Cathedral, the Eiffel Tower, and the Louvre Museum. It is also the birthplace of many famous French artists and writers, including Pablo Picasso and Oscar Wilde. Paris is home to many museums, art galleries, and other cultural institutions, and the city is known for its cuisine, fashion, and music. Despite being a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of potential and possibilities, and there are many exciting developments in the tech industry that are shaping the way that we live, work, and communicate. Here are some of the most likely future trends in AI:
    
    1. Autonomous vehicles: Autonomous vehicles are becoming more common in the industry, and it's likely that they will continue to become more advanced and efficient in the years to come. AI will play a critical role in the development of self-driving cars, with advanced algorithms and machine learning techniques that can help vehicles to navigate and safely navigate their surroundings.
    
    2. Personalized health care: AI will play an increasingly important role in personalized health care


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

     an

     [

    occupation

    ],

     [

    Job

     Title

    ]

     [

    Job

     Title

    ].

     I

    've

     been

     a

     [

    current

     occupation

    ]

     for

     [

    number

    ]

     years

    .

     I

    've

     always

     been

     [

    an

    xiety

     level

    ]

     about

     my

     future

    .

     If

     you

    're

     a

     young

     person

     going

     through

     difficult

     times

    ,

     I

    'm

     here

     for

     you

    .

     It

    's

     nice

     to

     meet

     you

    !

     Can

     you

     share

     more

     about

     your

     current

     occupation

     and

     any

     challenges

     you

    've

     faced

     so

     far

    ?


    Sure

    ,

     I

    'm

     an

     [

    occupation

    ],

     [

    Job

     Title

    ]

     [

    Job

     Title

    ].

     I

    've

     been

     a

     [

    current

     occupation

    ]

     for

     [

    number

    ]

     years

    .

     I

    've

     always

     been

     [

    an

    xiety

     level

    ]

     about

     my

     future

    
    
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

     and

     is

     the

     largest

     city

     in

     the

     country

     by

     population

    .

     It

     is

     also

     the

     seat

     of

     the

     French

     government

    ,

     the

     French

     parliament

    ,

     and

     the

     French

     Supreme

     Court

    .

     The

     city

     is

     famous

     for

     its

     medieval

     architecture

    ,

     art

    ,

     and

     cuisine

    .

     It

     is

     also

     home

     to

     the

     Lou

    vre

    ,

     the

     largest

     art

     museum

     in

     the

     world

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     and

     is

     known

     for

     its

     cultural

     and

     artistic

     achievements

    .

     The

     city

     is

     also

     known

     for

     its

     annual

     fashion

     and

     sport

     events

    .

     Paris

     has

     a

     rich

     history

     and

     is

     a

     major

     economic

     and

     political

     center

     of

     France

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     progress

     in

     areas

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

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Greater

     integration

     with

     human

     cognition

    :

     One

     of

     the

     key

     future

     trends

     is

     the

     increasing

     integration

     of

     AI

     with

     human

     cognition

    .

     This

     could

     lead

     to

     the

     development

     of

     AI

     systems

     that

     can

     mimic

     human

     cognitive

     functions

    ,

     such

     as

     decision

    -making

    ,

     problem

    -solving

    ,

     and

     learning

    .
    


    2

    .

     Improved

     ethics

     and

     accountability

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     it

    's

     important

     to

     address

     ethical

     concerns

     and

     ensure

     that

     they

     are

     designed

     and

     used

     eth

    ically

    .

     This

     includes

     ensuring

     that

     AI

     systems

     are

     transparent

    ,

     accountable

    ,

     and

     equitable

    ,

    



```python
llm.shutdown()
```
