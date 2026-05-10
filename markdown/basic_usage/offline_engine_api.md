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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.70it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.70it/s]


    2026-05-10 16:31:01,418 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 16:31:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.90it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]

    Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 27.02it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 36.40it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 36.40it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 36.40it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 36.40it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 36.40it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 36.40it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 36.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.85 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.93it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.93it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.23it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.23it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.23it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.23it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.23it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.23it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=640 avail_mem=71.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=576 avail_mem=71.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=480 avail_mem=71.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=480 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=448 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 43.36it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=352 avail_mem=71.68 GB):  52%|█████▏    | 30/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  52%|█████▏    | 30/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  60%|██████    | 35/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=288 avail_mem=71.67 GB):  60%|██████    | 35/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=256 avail_mem=71.67 GB):  60%|██████    | 35/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  60%|██████    | 35/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=224 avail_mem=71.66 GB):  60%|██████    | 35/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  60%|██████    | 35/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.47it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=144 avail_mem=71.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=96 avail_mem=71.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.77it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=48 avail_mem=71.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=48 avail_mem=71.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=28 avail_mem=71.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.75it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=12 avail_mem=71.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.08it/s] Capturing num tokens (num_tokens=4 avail_mem=71.60 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=4 avail_mem=71.60 GB): 100%|██████████| 58/58 [00:01<00:00, 41.24it/s]


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
    Generated text:  Ruben and I am a young man who has a passion for photography, photography has been my lifelong dream. My photography career is built on my past experience as a firefighter, my love for photography and my passion for adventure. I have over 10 years of experience working with both professional and amateur photographers and have a wide range of skills and techniques to inspire and excite your interest in photography. I am here to share my passion for photography with you. I specialize in Black & White photography and have a passion for experimenting with different techniques and creative compositions. I offer a no-pressure environment to allow you to explore and hone your own unique
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. The president of the United States was born in the United States. Therefore, the president of the United States is a man. What is the flaw in this argument?
    
    A) The argument is a syllogism.
    B) The argument contains a negative premise.
    C) The argument contains a negative conclusion.
    D) The argument contains an unacceptable counterexample.
    
    To determine the flaw in the argument, let's analyze the structure and logic step by step.
    
    The argument is:
    1. The president of the United States is a man.
    2. The president of the United States was born in the United States.
    3. Therefore
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. France is the largest country in the world by land area, but its capital Paris is not the largest city. Which of the following is the capital of France?
    
    A) Paris  
    B) Marseille  
    C) Lyon  
    D) Reims  
    E) Dunkirk
    
    To determine which city is the capital of France, we need to recall that the capital of France is the largest city by land area. Let's analyze each option step by step:
    
    A) Paris - This is the capital of France, and it is indeed the largest city by land area in terms of land area, not population.
    
    B) Marseille - This is
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of a group of highly intelligent, highly skilled and highly motivated people – the individuals who make the AI that is used to improve the quality of life of the average individual. In the last decade, AI has been able to make some breakthroughs in various fields. But to reach a level of maturity that allows it to be used in a way that is beneficial to all, there are some key areas that need to be addressed. Here are some of the most important areas where AI is currently in need of improvement.
    1. Ethical and Legal Considerations:
    The ethical and legal frameworks for AI are still developing. In the absence


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always eager to learn and grow. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always eager to learn and grow. I'm a [job title] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    The statement is factual because it provides a clear and unambiguous description of the capital city of France, including its name and location. It does not contain any assumptions, make any claims, or introduce any new information that is not explicitly stated. The statement is a straightforward and accurate representation of the capital city's location and significance. 
    
    In contrast, some other options might be:
    
    1. The capital of the United States is Washington, D.C.
    2. The capital of the United Kingdom is London.
    3. The capital of the United States is New York City. 
    
    These options are not factual because they contain assumptions, make
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management and fraud detection. As AI technology continues
    


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
    Generated text:  [insert character name], and I am a [insert profession or area of expertise] with a passion for [insert something related to your field of expertise]. I am constantly learning and growing, always striving to improve myself and my work. I believe that my experience and knowledge can help others succeed in their endeavors, so I strive to be an inspiration to those around me. I am a team player and am always willing to assist others and share my knowledge. I am open to feedback and always appreciate constructive criticism. I am looking forward to meeting you and learning from you. How can I help you today? I'm ready to learn from you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of lights and music.
    Paris is the largest city in Europe by population and the seat of the French government and head of state, and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The city is also known for its rich history, art, and culture, including its iconic museums and art exhibitions. Paris is a popular tourist destination and is a major economic and political center for France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and rapidly evolving, with a wide range of trends that are likely to shape its future direction. Here are some possible trends in AI:
    
    1. Automation and Deep Learning: With the rapid development of automation technology, we can expect to see more and more jobs automated and replaced by AI. This trend is already underway, with many companies investing heavily in AI automation to reduce costs and improve efficiency.
    
    2. AI Ethics and Governance: As AI becomes more integrated into our daily lives, there is a growing concern about its ethical implications. This trend includes issues such as bias, accountability, and transparency in AI systems.
    
    3. AI for Personalized


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

     AI

     language

     model

     designed

     to

     assist

     with

     various

     tasks

     like

     writing

     essays

    ,

     answering

     questions

    ,

     and

     even

     performing

     complex

     calculations

    .

     I

     can

     talk

     to

     you

     about

     pretty

     much

     any

     subject

     you

     want

    ,

     and

     I

     have

     a

     knack

     for

     generating

     sentences

     that

     are

     not

     only

     gramm

    atically

     correct

     but

     also

     well

    -

    structured

     and

     engaging

    .

     I

    'm

     not

     just

     a

     tool

     for

     people

     to

     rely

     on

    ;

     I

    'm

     a

     person

     I

     can

     be

     friendly

     with

    ,

     if

     you

     so

     choose

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ]

     is

     here

     to

     help

     you

     with

     your

     writing

    ,

     your

     questions

    ,

     and

     your

     calculations

    .

     Welcome

     to

     [

    Name

    ]

    !

     [

    Name

    ]

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Is

     there

     an

     answer

     to

     this

     question

    ?

     Paris

     is

     the

     capital

     of

     France

    .

     Let

     me

     know

     if

     you

     need

     any

     clarification

    .

     I

     am

     here

     to

     help

    !

     Is

     there

     an

     answer

     to

     this

     question

    ?

     Paris

     is

     the

     capital

     of

     France

    .

     Let

     me

     know

     if

     you

     need

     any

     clarification

    .

     Is

     there

     an

     answer

     to

     this

     question

    ?

     Paris

     is

     the

     capital

     of

     France

    .

     Let

     me

     know

     if

     you

     need

     any

     clarification

    .

     Is

     there

     an

     answer

     to

     this

     question

    ?

     Paris

     is

     the

     capital

     of

     France

    .

     Let

     me

     know

     if

     you

     need

     any

     clarification

    .

     Is

     there

     an

     answer

     to

     this

     question

    ?

     Paris

     is

     the

     capital

     of

     France

    .

     Let

     me

     know

     if

     you

     need

     any

     clarification

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     varied

    ,

     and

     it

    's

     likely

     to

     continue

     to

     evolve

     in

     many

     exciting

     ways

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

     in

     the

     coming

     years

    :
    


    1

    .

     Autonomous

     robots

     and

     drones

    :

     Self

    -driving

     cars

    ,

     drones

    ,

     and

     other

     autonomous

     robots

     are

     expected

     to

     become

     more

     common

     in

     our

     daily

     lives

    .

     These

     robots

     will

     be

     able

     to

     navigate

     the

     roads

     and

     navigate

     the

     city

    ,

     making

     them

     more

     efficient

     and

     safer

    .
    


    2

    .

     Improved

     natural

     language

     processing

    :

     As

     more

     and

     more

     people

     rely

     on

     text

    -based

     communication

    ,

     natural

     language

     processing

     (

    N

    LP

    )

     will

     become

     more

     sophisticated

    .

     This

     will

     allow

     machines

     to

     better

     understand

     and

     respond

     to

     human

     language

    ,

     making

     interactions

     more

    



```python
llm.shutdown()
```
