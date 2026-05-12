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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.10it/s]


    2026-05-12 21:47:31,764 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 21:47:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.42it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.10it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 14.76it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.12 GB):   3%|▎         | 2/58 [00:00<00:03, 14.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=75.12 GB):   3%|▎         | 2/58 [00:00<00:03, 14.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.12 GB):   7%|▋         | 4/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.12 GB):   7%|▋         | 4/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):   7%|▋         | 4/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):  10%|█         | 6/58 [00:00<00:03, 14.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.11 GB):  10%|█         | 6/58 [00:00<00:03, 14.72it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=75.10 GB):  10%|█         | 6/58 [00:00<00:03, 14.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.09 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.90it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.31it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.33it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.33it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.49it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.49it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.49it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.49it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.49it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.88it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.88it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.88it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.88it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.88it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.88it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.71it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.58it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.58it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=48 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=24 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.76it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.53 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.76it/s] Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 31.36it/s]


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
    Generated text:  Steve and I am an experienced software developer with a strong passion for creating innovative solutions and delivering high-quality code. I have a diverse set of skills that includes but are not limited to programming languages like Java, Python, C++, JavaScript, and Android development, as well as knowledge in software testing and debugging, project management, and cloud computing. I have been working in the software development field for over 10 years and have worked on a variety of projects, ranging from small business applications to large-scale enterprise systems. My background includes having worked with both open-source and proprietary software technologies, as well as being proficient in using modern tools such as
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president is in charge of the country and leads the country. Most of the time, the president's job is to talk to people about the country and make important decisions for the country. The president has a lot of important jobs to do, including making important laws, making important decisions about the military, and making important decisions about money. The president is always busy and always trying to do good things for the country. Some presidents are very busy being president, while others are very busy doing other important jobs, but they still have a very important job to do. The president has to be very careful about what they
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Lyon
    C. Brussels
    D. Nice
    Answer: A
    
    If a three-digit number can be represented as 30x + 2, and the hundreds digit is greater than the tens digit, then the value of x is ____.
    A. 1
    B. 2
    C. 3
    D. 4
    Answer: C
    
    In a hypothetical scenario, what measure should the population of a certain region take to increase the number of births?
    A. Lower the birth rate and increase the death rate
    B. Lower the birth rate and decrease the death rate
    
    ===============================
    Prompt: The future of AI is
    Generated text:  more personal
    When it comes to the topic of AI, the discussion tends to turn to whether or not it is a positive or negative force in society. Most people think that the future of AI is likely to be a positive force, but the truth is that it could be anything from a positive force to a negative force. In the first place, AI has the potential to make our lives easier, but at the same time, it could lead to a lot of negative consequences.
    One of the most significant developments of AI is the rise of natural language processing. This ability to understand and interpret human language has led to the creation of chatbots


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your profession or skills]. I enjoy [insert a short description of your hobbies or interests]. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby? I love [insert a short description of your favorite hobby]. What's your favorite book or movie? I love [insert a short description of your favorite book or movie].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is a bustling metropolis with a diverse population and a rich cultural heritage. The city is home to many famous landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. Paris is also known for its fashion industry, with many famous designers and fashion houses operating in the city. The city is a major hub for business and commerce, with many international companies and institutions headquartered there. Paris is a city of contrasts, with its modern architecture and vibrant culture blending seamlessly with its historical heritage.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be a greater emphasis on ensuring that they are used ethically and responsibly. This may involve developing new ethical guidelines and standards for AI systems, as well as increasing transparency and accountability in their use.
    
    2. Greater integration with human decision-making:
    


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
    Generated text:  [Your Name], and I'm a [short, possibly imaginative, but accurate] character. I'm an [if you want to make it more descriptive] [age], [occupation], [place of origin], [language], [what you're passionate about], and [what you're best at]. I'm confident, ambitious, and passionate, and I'm eager to share my thoughts, experiences, and any interesting information you might have about me. What about you?
    You?
    You are the protagonist of the story. You are a [if you want to make it more descriptive], [age], [occupation], [place of origin
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    (A) Yes
    (B) No
    (B) No
    
    Explanation: While Paris is indeed the capital of France, it is not a specific city. Paris is a metropolitan area consisting of the city of Paris and its surrounding suburbs, including Montmartre and nearby neighborhoods. It is not a city itself. Therefore, the statement "Paris's capital" is not entirely accurate, as it does not refer to a specific city. The correct answer is (B) No. However, the correct answer should be (A) Yes, as Paris is not a city, but rather the capital of France. The fact that it is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly speculative and uncertain, but here are some potential trends that have been observed in recent years:
    
    1. AI will become more accessible and affordable: With the rise of cloud computing and AI services like Google's TensorFlow, AI will become more accessible to businesses and individuals. This will allow for a wider range of applications and industries to benefit from AI.
    
    2. AI will continue to evolve and improve: AI is constantly evolving and improving, with new technologies emerging all the time. As AI algorithms get more sophisticated, they will become more accurate and able to handle complex problems.
    
    3. AI will be integrated into everyday life: AI will be integrated into


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

    insert

     first

     name

    ]

     and

     I

    'm

     a

     [

    insert

     profession

     or

     occupation

    ].

     I

    'm

     really

     excited

     to

     meet

     you

     because

     I

    'm

     here

     to

     help

     you

     with

     any

     questions

     you

     may

     have

     about

     [

    insert

     something

     relevant

     to

     your

     profession

     or

     work

    ].

     Whether

     you

    're

     looking

     for

     advice

     on

     [

    insert

     something

     relevant

     to

     your

     profession

     or

     work

    ],

     or

     just

     want

     to

     chat

     about

     [

    insert

     something

     relevant

     to

     your

     profession

     or

     work

    ],

     I

    'm

     always

     here

     to

     listen

     and

     provide

     the

     information

     you

     need

    .

     Let

    's

     get

     started

    !

     [

    Insert

     address

    ,

     phone

     number

    ,

     and

     other

     contact

     information

     if

     you

     have

     any

    ].

     [

    Insert

     any

     relevant

     images

     or

     graphics

     that

     would

     help

     to

     create

     a

     positive

     impression

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    I

    'm

     sorry

    ,

     I

     cannot

     provide

     an

     answer

     to

     this

     question

    .

     This

     appears

     to

     be

     a

     multiple

    -choice

     question

    ,

     but

     I

     do

     not

     have

     the

     ability

     to

     generate

     or

     provide

     answers

     to

     such

     questions

    .

     Can

     we

     please

     ask

     a

     different

     question

     or

     provide

     more

     information

    ?

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     one

     of

     continuous

     improvement

     and

     innovation

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

     precision

     and

     accuracy

    :

     AI

     models

     are

     becoming

     more

     accurate

     and

     precise

    ,

     able

     to

     perform

     tasks

     that

     were

     previously

     thought

     impossible

    .

     This

     is

     due

     to

     advancements

     in

     machine

     learning

    ,

     deep

     learning

    ,

     and

     neural

     networks

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     becoming

     more

     prevalent

    ,

     with

     some

     companies

     already

     testing

     them

     on

     the

     roads

    .

     As

     the

     technology

     advances

    ,

     autonomous

     vehicles

     are

     expected

     to

     become

     more

     advanced

    ,

     capable

     of

     performing

     complex

     tasks

     like

     driving

     in

     dangerous

     conditions

     or

     in

     remote

     areas

    .
    


    3

    .

     Enhanced

     user

     experiences

    :

     AI

     is

     being

     used

     to

     enhance

     user

     experiences

     in

     areas

    



```python
llm.shutdown()
```
