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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.42it/s]


    2026-05-13 21:56:39,403 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 21:56:39] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:04<00:13,  3.58it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]

    Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:04,  8.57it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:02, 14.71it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:04<00:00, 22.00it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:03, 17.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:03, 17.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:03, 17.81it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.87it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.06 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.87it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.87it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  21%|██        | 12/58 [00:00<00:01, 30.02it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.06 GB):  21%|██        | 12/58 [00:00<00:01, 30.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  21%|██        | 12/58 [00:00<00:01, 30.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:01, 30.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:01, 30.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:01, 30.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.52it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.52it/s] Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=896 avail_mem=72.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=832 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=640 avail_mem=72.01 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=640 avail_mem=72.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=576 avail_mem=72.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=480 avail_mem=72.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.39it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=320 avail_mem=72.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=288 avail_mem=71.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=224 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.57it/s]

    Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s] Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  81%|████████  | 47/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  81%|████████  | 47/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  81%|████████  | 47/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  81%|████████  | 47/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  81%|████████  | 47/58 [00:01<00:00, 45.53it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.94 GB):  81%|████████  | 47/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=28 avail_mem=71.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=20 avail_mem=71.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s] Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 39.83it/s]


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
    Generated text:  Anatoly. I'm a computer science student at the University of São Paulo. I have a Bachelor's degree in Computer Science, with minors in Software Engineering and Artificial Intelligence. In my spare time, I am a passionate musician, having been playing the piano since I was a teenager. I also have a passion for coding, which I've always been drawn to because it requires problem-solving skills and creativity. 
    
    In my free time, I also enjoy reading, cooking, and spending time with my family. I hope to continue to grow as a software engineer and a musician, and to pursue further opportunities in both fields. 
    
    So, what
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful person in the world. He or she is responsible for shaping the country’s direction and can have a significant impact on its development. The president is also the head of the executive branch of the government and is responsible for making important decisions on a wide range of issues, including foreign policy, defense, and economic policy.
    
    The President of the United States is elected every four years by the people who vote for them in a national election. The president’s term of office is two years and they are required to be at least 35 years old to be eligible to run for president. The president is supported by a group of 5
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. One of its most famous landmarks is the Eiffel Tower, a steeply inclined tower that stands at the edge of the Champ de Mars. As a result of its height, the Eiffel Tower became an international symbol of the French nation. Today, the tower stands at an elevation of 324 m above sea level and is a popular attraction for tourists visiting Paris.
    The Eiffel Tower was first constructed in 1887 and is the world’s tallest man-made structure. It was designed by Gustave Eiffel, a French engineer and architect. The tower was designed to be a symbol of
    ===============================
    Prompt: The future of AI is
    Generated text:  in motion, and emerging AI technologies are shaping up to make our world smarter, safer and more interconnected. This new world of AI is also bringing with it a number of challenges for us as we develop, deploy and manage AI systems. This is where the White House’s new AI Council on Cybersecurity and Security in the Digital Age comes in. The White House’s AI Council on Cybersecurity and Security in the Digital Age brings together the United States’ top cyber security experts to help shape the path to a smarter and safer world. As cyber security becomes a key driver of AI, it is essential that the White House’s AI Council is centered


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your favorite hobby or activity? I love [favorite hobby or activity]. I'm always looking for new experiences and I'm always eager to try new things. What's your favorite book or movie? I love [favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is also home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a vibrant and diverse city with a rich cultural scene, and is a popular destination for business and leisure activities. The city is also home to many important institutions, including the French Academy of Sciences and the French National Library. Paris is a city of contrasts
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration with human decision-making: AI systems are likely to become more integrated with human decision-making processes, allowing for more complex and nuanced decision-making.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more integrated into
    


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
    Generated text:  [Your Name] and I’m a [Your Age] year old [Your Profession or Title]. I’ve always been [your short bio]. I enjoy spending time with my family and friends, reading books, and trying new restaurants. My favorite hobby is [what’s your favorite hobby?]. I’m an [occupation] and I look forward to [what you’d like to tell your readers about yourself] today.
    I hope you can meet me in person soon! 
    Take care and have a great day! [Your Name] 
    Remember, you’re a treasure and worth more than any treasure you’ve ever owned. 
    You are
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and one of the most populous in the world, with a population of over 2 million people. It is known for its historical architecture, rich culture, and vibrant city life. Paris is known for its iconic landmarks such as Notre-Dame Cathedral and the Eiffel Tower, as well as for its famous cuisine and fashion scene. It is a major cultural center and a major transportation hub, with important ports and airports located nearby. Paris is a popular tourist destination and a major economic hub for France and the world. It is also home to numerous universities and research institutions. Additionally, Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, and several trends are likely to drive progress in the years ahead. Here are some of the most likely future trends in AI:
    
    1. Increased use of machine learning and deep learning: As AI systems become more complex, we can expect to see an increased use of machine learning and deep learning algorithms to develop increasingly sophisticated models. This will likely lead to even more accurate and effective solutions to a wide range of problems.
    
    2. Improved generalization: One of the key challenges in AI is ensuring that models generalize well to new data. As AI systems become more sophisticated, they will become better able to learn from more diverse and complex examples


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

    ],

     and

     I

     am

     a

     [

    career

    ,

     hobby

    ,

     etc

    .

    ].

     I

     enjoy

     [

    why

     you

     enjoy

     your

     work

    ].

     I

     hope

     you

     enjoy

     your

     time

     at

     the

     office

     and

     can

     learn

     from

     you

    !

     H

    ola

    ,

     mi

     nombre

     es

     [

    Tu

     Nombre

    ],

     y

     soy

     un

     [

    Pro

    yecto

    ,

     Cargo

    ,

     etc

    .

    ].

     Me

     s

    iento

     org

    ullo

    so

     de

     estar

     aquí

    .

     A

    precio

     que

     est

    és

     disp

    uesto

     a

     aprender

     de

     mí

    .

     ¡

    H

    asta

     pronto

    !

     
    


    Este

     es

     un

     ejemplo

     simple

     de

     un

     breve

    ,

     neutral

     self

    -int

    rodu

    cción

    .

     A

    seg

    ú

    rate

     de

     adapt

    arlo

     a

     tu

     propio

     estilo

     de

     escrit

    ura

     y

     público

     objetivo

    .

     
    


    Remember

     that

     you

     can

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     is

     known

     for

     its

     rich

     history

    ,

     world

    -class

     museums

    ,

     and

     fashion

     scene

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     home

     to

     many

     famous

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

     The

     city

     has

     a

     diverse

     population

     of

     over

     

    1

     million

     people

     and

     is

     a

     major

     economic

     and

     cultural

     hub

     in

     France

    .

     It

     is

     the

     seat

     of

     the

     French

     government

    ,

     and

     its

     rich

     culture

     and

     history

     have

     made

     it

     a

     cultural

     center

     in

     France

    .

     Paris

     is

     known

     for

     its

     romantic

     and

     romantic

     atmosphere

    ,

     and

     it

     is

     often

     referred

     to

     as

     the

     "

    Paris

    ian

     way

    ".

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     rapidly

    ,

     with

     new

     technologies

    ,

     algorithms

    ,

     and

     applications

     being

     developed

     and

     refined

     at

     an

     unprecedented

     pace

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

     Increased

     privacy

     and

     data

     protection

    :

     With

     the

     increasing

     amount

     of

     data

     generated

     by

     human

     activities

    ,

     AI

     is

     becoming

     more

     sophisticated

     and

     able

     to

     analyze

     and

     process

     it

     in

     a

     more

     ethical

     and

     transparent

     way

    .

     This

     will

     lead

     to

     increased

     privacy

     concerns

     and

     the

     development

     of

     new

     privacy

     protections

     and

     regulations

    .
    


    2

    .

     AI

     systems

     becoming

     more

     human

    -like

    :

     AI

     is

     becoming

     increasingly

     intelligent

     and

     capable

     of

     making

     decisions

     that

     appear

     to

     be

     human

    -like

    .

     This

     could

     lead

     to

     more

     advanced

     AI

     systems

     that

     can

     recognize

     and

     mimic

    



```python
llm.shutdown()
```
