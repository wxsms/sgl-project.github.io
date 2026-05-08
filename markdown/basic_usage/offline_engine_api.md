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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.78it/s]


    2026-05-08 11:05:05,392 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 11:05:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:14,  3.30it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s]

    Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:04<00:05,  7.74it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]

    Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:04<00:01, 19.71it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 27.01it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 27.01it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 27.01it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 27.01it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 27.01it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 27.01it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 27.01it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 27.01it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 27.01it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 27.01it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 35.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.12 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.09 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.74it/s]Capturing num tokens (num_tokens=960 avail_mem=72.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.74it/s] Capturing num tokens (num_tokens=896 avail_mem=72.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.74it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.06 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.74it/s]Capturing num tokens (num_tokens=832 avail_mem=72.06 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=768 avail_mem=72.06 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=704 avail_mem=72.06 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=640 avail_mem=72.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=576 avail_mem=72.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=480 avail_mem=72.05 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=448 avail_mem=71.77 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=416 avail_mem=71.77 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.76 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=352 avail_mem=71.76 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=352 avail_mem=71.76 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=320 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=288 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=256 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=240 avail_mem=71.74 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=224 avail_mem=71.74 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=224 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=208 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=192 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=176 avail_mem=71.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.35it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=144 avail_mem=71.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=144 avail_mem=71.73 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=128 avail_mem=71.73 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=112 avail_mem=71.72 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=96 avail_mem=71.72 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.39it/s] Capturing num tokens (num_tokens=80 avail_mem=71.72 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=64 avail_mem=71.71 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=64 avail_mem=71.71 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=48 avail_mem=71.71 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=32 avail_mem=71.71 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=28 avail_mem=71.70 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.61it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.70 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=20 avail_mem=71.69 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=20 avail_mem=71.69 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=16 avail_mem=71.69 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=12 avail_mem=71.69 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=8 avail_mem=71.69 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.28it/s] Capturing num tokens (num_tokens=4 avail_mem=71.68 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=4 avail_mem=71.68 GB): 100%|██████████| 58/58 [00:01<00:00, 40.63it/s]


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
    Generated text:  Monica and I am a professional marketing manager with over 10 years of experience. Having worked on a range of clients from small startups to big companies, I have a unique understanding of how to help businesses grow and succeed. My experience has helped me develop marketing strategies that are focused on growing customer loyalty, customer retention, and driving revenue growth. I also have a strong understanding of how to effectively manage the resources and budget needed to achieve marketing goals. I am confident that I can help you achieve your business goals. Let's talk about how we can work together. [REASONING AND CAUSE- Effectiveness AND RESULTS] Monica:
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ceremonial position, but the first lady is a real person. The first lady is a female member of the U.S. president's family. As such, she has the privilege of being the vice president's wife.
    The first lady is known for helping the president during natural disasters and crisis situations. She often travels to other countries to help the president with humanitarian assistance. The first lady is a philanthropist, and she makes donations to a variety of causes she feels are important.
    The first lady's first name is Elizabeth and her last name is Banning. She was born on July 16, 1946 in
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. How many seconds will it take for the same car to travel 3600 meters?
    To determine how many seconds it will take for the car to travel 3600 meters, we need to know the speed of the car. The speed of the car is not provided in the problem, so we will assume that the speed of the car is constant and given as 60 meters per second. Once we have the speed, we can calculate the time it takes to travel 3600 meters using the formula:
    
    \[ \text{Time} = \frac{\text{Distance}}{\text{Speed
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it is everywhere.
    AI is already used in healthcare, financial services, transportation, manufacturing, and more. It has already transformed industries and it will continue to do so.
    Our firms will need to adapt to this new reality of an AI-powered future. Our industry is already seeing major changes in the way that we work and manage. But in the future, how do we adapt to this new reality?
    In this webinar, we’ll discuss the future of AI, how it will impact the industry, and how to start using it in your firm.
    Michael Zick, Executive Vice President of Global Technology Services, is an accomplished technology


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Parliament Building. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Riviera, a popular tourist destination for its beautiful beaches and Mediterranean climate. Paris is a city that is both a cultural and historical center of France. It is also a major transportation
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can better understand and respond to human emotions and preferences.
    
    2. Greater emphasis on ethical considerations: As AI systems become more complex and sophisticated, there will be a greater emphasis on ethical considerations and responsible use of AI. This could lead to more stringent regulations and guidelines for the development and deployment of AI systems.
    
    3. Increased use of AI in healthcare: AI is already
    


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
    Generated text:  [insert name of the character], and I'm a/an [insert character's role or profession]. I'm [insert character's age], [insert character's nationality], and I currently [insert character's occupation]. Throughout the course of [insert length of time since character's debut], I've been [insert character's unique skill or personality trait]. I enjoy [insert character's hobby or activity]. Whether I'm a [insert hobby] or [insert activity], I love [insert what I enjoy most about it]. I'm [insert character's favorite color], [insert character's favorite genre of music], and [insert character's favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the cultural, political, and economic center of the nation.
    
    That is quite right. The capital city of France is indeed Paris, which has been the metropolis of the French nation since ancient times. It is one of the most important cities in the world for its history, architecture, and art. Paris is home to a diverse array of cultural institutions and attractions, from the Eiffel Tower to the Louvre Museum. The city is also known for its food, fashion, and music, with many world-renowned artists, musicians, and chefs. Paris is a city of contrasts and a city of incredible beauty,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving and there is a lot of potential for incredible advancements in the field. Here are some possible future trends in AI:
    
    1. Integration of AI and human expertise: One of the biggest trends in AI is the integration of AI with human expertise. This can include the use of AI in areas such as healthcare, finance, and transportation, where human expertise can bring valuable insights and solutions.
    
    2. Increased use of AI in customer service: AI is already being used in customer service to provide personalized responses and automate repetitive tasks. In the future, we can expect to see even more integration with human interaction, leading to more efficient and personalized customer


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

    ]

     and

     I

     am

     a

    /an

     [

    job

     title

     or

     profession

    ]

     in

     [

    industry

    ].

     I

     have

     always

     been

     passionate

     about

     [

    my

     primary

     interest

     or

     hobby

    ].

     My

     experience

     has

     been

     valuable

     in

     [

    example

     of

     experience

    ].

     Currently

    ,

     I

     am

     working

     as

     a

    /an

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

     have

     been

     involved

     in

     [

    example

     of

     a

     professional

     achievement

    ].

     As

     a

     result

    ,

     I

     am

     confident

     [

    example

     of

     confidence

     or

     self

    -ass

    essment

    ].

     I

     am

     a

     [

    example

     of

     self

    -

    identification

    ].

     My

     [

    example

     of

     a

     defining

     trait

     or

     characteristic

    ]

     is

     [

    example

     of

     self

    -

    identification

    ].

     Overall

    ,

     I

     am

     [

    example

     of

     a

     positive

     self

    -ass

    essment

    ].

    
    
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

     France

     by

     population

     and

     the

     fifth

    -largest

     city

     in

     the

     world

     by

     area

    .

     Paris

     is

     known

     for

     its

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     It

     also

     has

     a

     rich

     history

    ,

     with

     its

     heritage

     landmarks

     including

     the

     Tower

     of

     the

     Winds

     and

     the

     Tu

    il

    eries

     Gardens

    .

     Paris

     is

     a

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

     and

     plays

     an

     important

     role

     in

     French

     foreign

     policy

    .

     It

     is

     a

     major

     financial

     and

     business

     hub

     and

     is

     home

     to

     many

     top

    -tier

     universities

    ,

     including

     the

     Sor

    bon

    ne

    .

     France

    ’s

     capital

     city

     is

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     currently

     looking

     very

     promising

    ,

     and

     the

     trends

     that

     we

     can

     expect

     will

     be

    :
    


    1

    .

     Increased

     Use

     of

     AI

     in

     Healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     diagnose

     diseases

     and

     develop

     personalized

     treatment

     plans

    .

     We

     can

     expect

     that

     this

     trend

     will

     continue

     as

     AI

     becomes

     more

     efficient

     and

     less

     expensive

    .

     
    


    2

    .

     AI

     in

     Finance

    :

     AI

     will

     be

     used

     in

     finance

     to

     improve

     the

     risk

     management

     and

     trading

     algorithms

    .

     This

     will

     help

     to

     reduce

     fraud

     and

     increase

     efficiency

     in

     financial

     transactions

    .

     
    


    3

    .

     AI

     in

     Manufacturing

    :

     AI

     will

     be

     used

     in

     manufacturing

     to

     optimize

     production

     processes

     and

     reduce

     costs

    .

     AI

     can

     also

     be

     used

     to

     develop

     predictive

     models

     that

     can

     help

     to

     anticipate

     and

    



```python
llm.shutdown()
```
