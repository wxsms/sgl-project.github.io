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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.04it/s]


    2026-05-09 03:19:22,814 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 03:19:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]

    Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 11.47it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 17.93it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 17.93it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 17.93it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 17.93it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 17.93it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 17.93it/s]

    Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 17.93it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 17.93it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 17.93it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 25.29it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]

    Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 34.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 46.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.50it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.50it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.50it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.50it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.50it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 43.11it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.09it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.18it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 21.99it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 21.99it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 21.99it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 21.99it/s] Capturing num tokens (num_tokens=4 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 21.99it/s]Capturing num tokens (num_tokens=4 avail_mem=74.01 GB): 100%|██████████| 58/58 [00:01<00:00, 32.30it/s]


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
    Generated text:  Frank and I'm a biologist at the University of California, Santa Barbara. I'm studying the influence of plants and the fungi that grow on them on the environment.
    I don't have a favorite species or any favorite place on earth. But I do enjoy the idea of visiting places like the Great Basin, where I've been on many trips, and the end of the world, where I've been to see the most diverse and awesome places in the entire world.
    Some of the things I find most interesting about the environment and most exciting about living here are the animals, plants, and fungi. They all play a role in shaping the environment
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what to do with the $100 million budget shortfall for the federal government. The budget is divided into four categories: military, education, infrastructure, and healthcare. The president has identified three potential solutions to the problem:
    
    1. Increase the military budget by $10 million.
    2. Decrease the education budget by $5 million.
    3. Increase the infrastructure budget by $20 million.
    4. Decrease the healthcare budget by $15 million.
    
    If the military budget is currently $70 million, the education budget is $50 million, the infrastructure budget is $80 million, and the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. However, the old capital of France, not Paris, is Toulouse. Which of the following statements can be drawn from the above statement?
    A: Toulouse is not a capital of France.
    B: Paris is the capital of France.
    C: Toulouse is a capital of France.
    D: Paris is the capital of Toulouse.
    To determine which statement can be drawn from the given information, let's analyze the statement step by step.
    
    1. The capital of France is Paris.
    2. The old capital of France, not Paris, is Toulouse.
    
    From these two pieces of information, we can deduce the following
    ===============================
    Prompt: The future of AI is
    Generated text:  in humans and it’s not going to come from Silicon Valley anymore. As the demand for AI continues to grow, so do the value of the technical workforce and the skills required to excel in the field. Whether you’re in the software or data science industry, becoming an AI expert is a must-have skill in the modern workplace. In this article, we’ll cover the skills and qualifications required for an AI expert.
    1. AI background: An AI background is key to developing an AI expert. A basic understanding of AI is necessary to develop knowledge of the latest AI technologies and how to incorporate them into the workforce.
    2. Programming skills:


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique skill or trait] that I've honed over the years. What's your favorite hobby or activity? I enjoy [insert a hobby or activity]. What's your favorite book or movie? I love [insert a favorite book or movie]. What's your favorite place to relax? I love [insert a favorite place to relax]. What's your favorite color? I love [insert a favorite color]. What's your favorite food
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its cuisine, fashion, and art scene. It is also a major center for business, finance, and tourism in Europe. Paris is a popular tourist destination and a cultural hub for many people around the world. The city is home to many museums, theaters, and other cultural institutions, and is a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from experience and improve their performance over time.
    
    2. Enhanced privacy and security: As AI systems become more complex and sophisticated, there will be a growing need for measures to protect user privacy and security. This could include measures such as encryption, access controls, and regular audits of AI systems.
    
    3. Greater emphasis on ethical considerations: As AI systems become more complex
    


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
    Generated text:  [Name] and I am a [occupation] who has been [number of years in occupation] for [number of years]. I am confident in [reason for confidence] and I am [gender] to [gender neutral term]. I have a keen sense of [profession], and I strive to be the best [occupation] I can be. I am always ready to learn, adapt, and grow, and I am committed to staying up-to-date on the latest trends and developments in my field. I am also [personality trait] and I am [non-binary term]. I believe in [value or belief] and I strive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La République Animée," a historic city known for its beautiful architecture, rich history, and vibrant culture. The city is home to many museums, including the Louvre, where works of art are displayed, and is also a popular destination for tourists. Paris has a rich cultural heritage and is a major hub for business, politics, and fashion. The city is also known for its annual Carnival, a week-long festival that features live music, parades, and colorful costumes. Paris is a city of contrasts, with its opulent architecture, chic fashion, and delicious cuisine. It is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but there are some possible trends that are likely to shape it. Here are some potential developments:
    
    1. Increasing integration of AI with other technologies: AI will continue to be integrated into other technologies such as smart homes, self-driving cars, and smart healthcare systems.
    
    2. AI will become more ethical: AI systems will become more transparent, accountable, and responsible in their decisions, which will require ethical guidelines and regulations.
    
    3. AI will be integrated into everyday life: AI will be more integrated into our lives, such as through voice-activated assistants, self-driving cars, and virtual assistants like Alexa or Siri.
    
    4. AI


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

    'm

     a

     [

    occupation

    ]

     who

     has

     always

     been

     passionate

     about

     [

    career

     goal

     or

     hobby

    ].

     I

    'm

     always

     looking

     for

     opportunities

     to

     [

    action

     or

     achievement

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     Start

     with

     a

     brief

     introduction

     and

     explain

     your

     unique

     background

     and

     experiences

    .

     To

     me

    ,

     my

     journey

     has

     been

     both

     challenging

     and

     exciting

    .

     What

     led

     you

     to

     this

     path

     and

     what

    's

     your

     favorite

     experience

     so

     far

    ?

     Start

     with

     your

     personal

     and

     professional

     goals

     and

     how

     you

     aim

     to

     achieve

     them

    .

     Finally

    ,

     reflect

     on

     what

     you

     can

     learn

     from

     the

     experiences

     of

     others

     and

     how

     you

     plan

     to

     apply

     them

     in

     your

     personal

     or

     professional

     life

    .

     To

     me

    ,

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     "

    City

     of

     Love

    "

     and

     the

     "

    City

     of

     Light

    ".


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     the

     country

    .

     It

     is

     renowned

     for

     its

     rich

     history

    ,

     art

    ,

     cuisine

    ,

     fashion

    ,

     and

     music

    ,

     which

     have

     made

     it

     one

     of

     the

     world

    's

     most

     famous

     cities

    .

     The

     city

     is

     located

     in

     the

     South

     East

     of

     France

     at

     the

     mouth

     of

     the

     Se

    ine

     River

    .

     It

     is

     home

     to

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

     Notre

     Dame

     Cathedral

    ,

     among

     many

     other

     landmarks

    .


    In

     

    2

    0

    1

    9

    ,

     Paris

     had

     a

     population

     of

     over

     

    1

    1

     million

     people

    ,

     making

     it

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    :
    


    1

    .

     Increased

     AI

     ethics

     and

     regulation

    :

     As

     AI

     becomes

     more

     integrated

     into

     society

    ,

     there

     will

     likely

     be

     increased

     scrutiny

     and

     regulation

     of

     its

     development

     and

     deployment

    .

     This

     will

     include

     considerations

     of

     privacy

    ,

     safety

    ,

     and

     fairness

    .
    


    2

    .

     AI

     autonomy

     and

     self

    -

    organization

    :

     AI

     systems

     are

     likely

     to

     become

     more

     autonomous

     and

     capable

     of

     making

     decisions

     and

     actions

     without

     human

     intervention

    .

     This

     will

     require

     significant

     developments

     in

     AI

     technology

    ,

     including

     improvements

     in

     neural

     networks

    ,

     reinforcement

     learning

    ,

     and

     machine

     learning

    .
    


    3

    .

     AI

     for

     better

     health

     care

    :

     AI

    -powered

     healthcare

     systems

     are

     likely

     to

     become

     more

     advanced

     and

     effective

     in

     diagn

    osing

     and

     treating

     diseases

    ,

    



```python
llm.shutdown()
```
