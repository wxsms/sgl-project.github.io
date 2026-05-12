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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]


    2026-05-12 21:10:17,096 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 21:10:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:45,  3.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:45,  3.95s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:45,  3.95s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:45,  3.95s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:45,  3.95s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.87it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 24.27it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 34.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.54 GB):   3%|▎         | 2/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.54 GB):   3%|▎         | 2/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.54 GB):   3%|▎         | 2/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.54 GB):   3%|▎         | 2/58 [00:00<00:02, 19.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.54 GB):   9%|▊         | 5/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.53 GB):   9%|▊         | 5/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.50 GB):   9%|▊         | 5/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.50 GB):   9%|▊         | 5/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.50 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.50 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.49 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.49 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.49 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.36it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=56.49 GB):  21%|██        | 12/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.48 GB):  21%|██        | 12/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.48 GB):  21%|██        | 12/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.48 GB):  21%|██        | 12/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.48 GB):  21%|██        | 12/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.47 GB):  21%|██        | 12/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.46 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.44 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=960 avail_mem=56.46 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=56.46 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=896 avail_mem=56.46 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=832 avail_mem=56.45 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=768 avail_mem=56.45 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=704 avail_mem=56.45 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=640 avail_mem=56.44 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=640 avail_mem=56.44 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.07it/s]Capturing num tokens (num_tokens=576 avail_mem=56.44 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.07it/s]Capturing num tokens (num_tokens=512 avail_mem=56.38 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.07it/s]Capturing num tokens (num_tokens=480 avail_mem=56.39 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.07it/s]

    Capturing num tokens (num_tokens=448 avail_mem=56.39 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.07it/s]Capturing num tokens (num_tokens=448 avail_mem=56.39 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=416 avail_mem=56.39 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=384 avail_mem=56.39 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=352 avail_mem=56.38 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=320 avail_mem=56.37 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=288 avail_mem=56.37 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=288 avail_mem=56.37 GB):  62%|██████▏   | 36/58 [00:00<00:00, 41.75it/s]Capturing num tokens (num_tokens=256 avail_mem=56.37 GB):  62%|██████▏   | 36/58 [00:00<00:00, 41.75it/s]Capturing num tokens (num_tokens=240 avail_mem=56.37 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=224 avail_mem=56.36 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.75it/s]

    Capturing num tokens (num_tokens=208 avail_mem=56.36 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=192 avail_mem=56.36 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=192 avail_mem=56.36 GB):  71%|███████   | 41/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=176 avail_mem=56.35 GB):  71%|███████   | 41/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=160 avail_mem=56.35 GB):  71%|███████   | 41/58 [00:01<00:00, 32.73it/s]

    Capturing num tokens (num_tokens=144 avail_mem=56.35 GB):  71%|███████   | 41/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=128 avail_mem=56.35 GB):  71%|███████   | 41/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=128 avail_mem=56.35 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.64it/s]Capturing num tokens (num_tokens=112 avail_mem=56.34 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.64it/s]Capturing num tokens (num_tokens=96 avail_mem=56.34 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.64it/s] Capturing num tokens (num_tokens=80 avail_mem=56.34 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.64it/s]

    Capturing num tokens (num_tokens=64 avail_mem=56.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.64it/s]Capturing num tokens (num_tokens=64 avail_mem=56.33 GB):  84%|████████▍ | 49/58 [00:01<00:00, 24.69it/s]Capturing num tokens (num_tokens=48 avail_mem=56.33 GB):  84%|████████▍ | 49/58 [00:01<00:00, 24.69it/s]Capturing num tokens (num_tokens=32 avail_mem=56.33 GB):  84%|████████▍ | 49/58 [00:01<00:00, 24.69it/s]Capturing num tokens (num_tokens=28 avail_mem=56.32 GB):  84%|████████▍ | 49/58 [00:01<00:00, 24.69it/s]

    Capturing num tokens (num_tokens=28 avail_mem=56.32 GB):  90%|████████▉ | 52/58 [00:01<00:00, 23.11it/s]Capturing num tokens (num_tokens=24 avail_mem=56.32 GB):  90%|████████▉ | 52/58 [00:01<00:00, 23.11it/s]Capturing num tokens (num_tokens=20 avail_mem=56.32 GB):  90%|████████▉ | 52/58 [00:01<00:00, 23.11it/s]Capturing num tokens (num_tokens=16 avail_mem=56.32 GB):  90%|████████▉ | 52/58 [00:01<00:00, 23.11it/s]Capturing num tokens (num_tokens=16 avail_mem=56.32 GB):  95%|█████████▍| 55/58 [00:01<00:00, 24.09it/s]Capturing num tokens (num_tokens=12 avail_mem=56.31 GB):  95%|█████████▍| 55/58 [00:01<00:00, 24.09it/s]Capturing num tokens (num_tokens=8 avail_mem=56.31 GB):  95%|█████████▍| 55/58 [00:01<00:00, 24.09it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=56.30 GB):  95%|█████████▍| 55/58 [00:01<00:00, 24.09it/s]Capturing num tokens (num_tokens=4 avail_mem=56.30 GB): 100%|██████████| 58/58 [00:02<00:00, 22.74it/s]Capturing num tokens (num_tokens=4 avail_mem=56.30 GB): 100%|██████████| 58/58 [00:02<00:00, 28.52it/s]


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
    Generated text:  Tyler, and I’m a 25-year-old software engineer. I’m currently working on a project that involves utilizing a specific programming language, which I’ll refer to as "language X." As a beginner in programming, I am seeking guidance on what programming languages to use in my projects. Please provide me with a list of at least 5 programming languages that I should consider for my project, and explain why I should choose each language. Additionally, please provide examples of real-world applications where each language would be particularly useful.
    
    Sure, I'd be happy to help you with that! Here are 5 programming languages that you should consider
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a country with a population of 300 million. If the population is 2.5 times the size of the country, how many people are there in the country?
    
    To determine the population of the country, we start by noting the given information:
    
    1. The population of the United States is 300 million.
    2. The population of the United States is 2.5 times the size of the country.
    
    First, we calculate the population of the country by dividing the population of the United States by 2.5:
    
    \[
    \text{Population of the country} = \frac{\text{Population
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of the European Union, that’s one of the most important countries in the world. But what is the capital of the European Union? The European Union is a supranational organization that was created in the 1950s to promote unity and cooperation among its member countries in Europe. The capital of France is the capital of France, and it is called Paris. Here are some interesting facts about the capital of France and the European Union:
    
    1. The capital of France is Paris, which is the largest city in France and one of the most important cities in Europe. It is located in the north of France and is
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the people and the people are the key to the future of AI.
    
    Could this be a logical argument?
    The argument is valid because it is based on a premise that supports a conclusion, the conclusion being that the future of AI is in the hands of the people. The argument is valid because it is based on a premise that supports a conclusion, the conclusion being that the future of AI is in the hands of the people.
    This argument is not valid because it is based on a premise that does not support a conclusion, the conclusion being that the future of AI is in the hands of the people. The argument is not


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your personality or skills here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity here]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I love
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. It is a major center of politics, science, and art, and is a major player in global affairs. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to play an increasingly important role in shaping the future of society, from improving healthcare and education to creating new forms of entertainment and entertainment. Finally, the development of AI is likely to be driven by a combination of technological advances, economic factors, and social and political changes. Overall, the future of AI is likely to be characterized by continued innovation and growth,
    


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
    Generated text:  [Name] and I'm a [Occupation]! I'm a [Brief description of your occupation] who, through my [Three or Four Key Skills], have been able to achieve success and make a positive impact in the world. Whether it's through [Your First Achievement], [Your Second Achievement], or [Your Last Achievement], I'm determined to keep moving forward and pursue my passions and goals. I believe that the key to success is always striving for excellence, and that's why I'm always looking for ways to improve myself and keep learning. I'm eager to share my experiences and knowledge with others and help them achieve their
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La République, " the most populous city in the European Union, and the third largest city in the world. Paris is home to many iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Champs-Élysées. The city is also known for its rich history and culture, which can be seen in its many museums, festivals, and cultural events throughout the year. Overall, Paris is a vibrant and diverse city with a rich history and a strong sense of community.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  poised for significant growth and advancement, driven by a combination of technological innovations, data processing capabilities, and evolving societal needs. Here are some potential future trends in AI:
    
    1. Enhanced AI capabilities: As AI technologies advance, their capabilities are likely to grow even more advanced and powerful. This could include improved learning, understanding, and generalization capabilities, as well as more sophisticated decision-making and problem-solving capabilities.
    
    2. Deeper AI: With the increasing availability of large amounts of data and the availability of more powerful computing resources, AI models will likely become more adept at understanding and processing complex information. This could include more complex reasoning and decision-making


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

    'm

     [

    Your

     Age

    ].

     I

    'm

     a

     [

    Your

     Profession

    ]

     who

     has

     been

     working

     for

     [

    Your

     Company

    /

    Role

    ]

     for

     [

    Your

     Length

     of

     Service

    ]

     years

    .

     I

    'm

     passionate

     about

     [

    Your

     Personal

     Passion

     or

     Hobby

    ],

     and

     I

     enjoy

     [

    Your

     Love

    /L

    ove

     of

     Sport

    /

    Interest

    ].

     I

     believe

     in

     [

    Your

     Core

     Values

     or

     Philosophy

    ].

     I

    'm

     a

     [

    Your

     Character

     Trait

     or

     Character

     -

     such

     as

     kind

    ,

     adventurous

    ,

     humorous

    ,

     etc

    .

    ].

     I

    'm

     a

     [

    Your

     Inter

    ests

    ,

     passions

    ,

     or

     hobbies

    ]

     who

     I

     love

    .

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    Your

     Inter

    ests

    ,

     hobbies

    ,

     or

     activities

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    (A

    )

     The

     capital

     of

     France

     is

     in

     the

     mountains

    .


    (B

    )

     The

     capital

     of

     France

     is

     in

     the

     city

    .


    (C

    )

     The

     capital

     of

     France

     is

     in

     the

     country

    .


    (D

    )

     The

     capital

     of

     France

     is

     in

     the

     sea

    .

     


    (E

    )

     The

     capital

     of

     France

     is

     in

     a

     country

    .

     
    


    My

     answer

     is

     B

    .

     The

     capital

     of

     France

     is

     in

     the

     city

    .

     The

     correct

     answer

     is

     B

    .

     The

     capital

     of

     France

     is

     in

     the

     city

    .

     
    


    This

     is

     a

     fact

    -based

     question

     about

     the

     capital

     city

     of

     France

    ,

     which

     is

     Paris

    .

     The

     other

     options

     are

     either

     incorrect

     (

    like

     the

     country

     being

     in a

     sea

    ,

     mountains

     being

     in

     the

     sea

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     increasingly

     optimistic

    .

     Here

     are

     some

     possible

     trends

     to

     watch

     for

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     One

     of

     the

     key

     trends

     we

    're

     seeing

     in

     AI

     is

     an

     increased

     integration

     of

     machine

     learning

     with

     human

     intelligence

    .

     This

     means

     that

     AI

     systems

     are

     becoming

     more

     capable

     of

     understanding

     and

     making

     decisions

     based

     on

     human

     values

     and

     perspectives

    .
    


    2

    .

     AI

     is

     becoming

     more

     sophisticated

     and

     capable

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     sophisticated

     and

     capable

     AI

     systems

     that

     can

     perform

     tasks

     beyond

     what

     humans

     can

     do

     today

    .
    


    3

    .

     AI

     is

     becoming

     more

     human

    -like

    :

     As

     AI

     becomes

     more

     sophisticated

     and

     capable

    ,

     we

     can

     expect

     to

     see

     more

     and

     more

     AI

    



```python
llm.shutdown()
```
