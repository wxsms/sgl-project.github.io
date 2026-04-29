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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.58it/s]


    2026-04-29 11:33:12,119 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 11:33:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]Compiling num tokens (num_tokens=2816):  10%|█         | 6/58 [00:04<00:30,  1.69it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03,  9.47it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03,  9.47it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03,  9.47it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03,  9.47it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03,  9.47it/s]

    Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03,  9.47it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03,  9.47it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:05<00:03,  9.47it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.08it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.32 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.29 GB):   3%|▎         | 2/58 [00:00<00:03, 16.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.29 GB):   3%|▎         | 2/58 [00:00<00:03, 16.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.29 GB):   3%|▎         | 2/58 [00:00<00:03, 16.45it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.29 GB):   3%|▎         | 2/58 [00:00<00:03, 16.45it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.28 GB):   9%|▊         | 5/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.27 GB):   9%|▊         | 5/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.27 GB):   9%|▊         | 5/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.27 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.27 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.26 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=116.26 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.25 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.25 GB):  21%|██        | 12/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.20 GB):  21%|██        | 12/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.20 GB):  21%|██        | 12/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.18 GB):  21%|██        | 12/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.17 GB):  21%|██        | 12/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.17 GB):  21%|██        | 12/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.81it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=116.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=960 avail_mem=116.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.81it/s] Capturing num tokens (num_tokens=960 avail_mem=116.16 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=896 avail_mem=116.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=832 avail_mem=116.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=768 avail_mem=116.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=704 avail_mem=116.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=640 avail_mem=116.12 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=640 avail_mem=116.12 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.50it/s]Capturing num tokens (num_tokens=576 avail_mem=116.12 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.50it/s]

    Capturing num tokens (num_tokens=512 avail_mem=116.08 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.50it/s]Capturing num tokens (num_tokens=480 avail_mem=116.10 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.50it/s]Capturing num tokens (num_tokens=448 avail_mem=116.10 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.50it/s]Capturing num tokens (num_tokens=416 avail_mem=116.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.50it/s]Capturing num tokens (num_tokens=416 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=352 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=320 avail_mem=116.08 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.75it/s]

    Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=256 avail_mem=116.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=240 avail_mem=116.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=224 avail_mem=116.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=208 avail_mem=116.06 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=208 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=192 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=176 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=160 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=144 avail_mem=116.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.75it/s]

    Capturing num tokens (num_tokens=144 avail_mem=116.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.25it/s]Capturing num tokens (num_tokens=128 avail_mem=116.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.25it/s]Capturing num tokens (num_tokens=112 avail_mem=116.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.25it/s]Capturing num tokens (num_tokens=96 avail_mem=116.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.25it/s] Capturing num tokens (num_tokens=80 avail_mem=116.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.25it/s]Capturing num tokens (num_tokens=80 avail_mem=116.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=64 avail_mem=116.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=48 avail_mem=116.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=32 avail_mem=116.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=28 avail_mem=116.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.66it/s]

    Capturing num tokens (num_tokens=28 avail_mem=116.03 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=24 avail_mem=116.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=20 avail_mem=116.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=16 avail_mem=116.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=12 avail_mem=116.01 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=12 avail_mem=116.01 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=8 avail_mem=116.01 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.75it/s] Capturing num tokens (num_tokens=4 avail_mem=116.01 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=4 avail_mem=116.01 GB): 100%|██████████| 58/58 [00:01<00:00, 35.14it/s]


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
    Generated text:  Bess, I'm a 22 year old girl from New Zealand. I'm from an Aborigine culture. I was born with a severe genetic condition and I'm now 16 years old. I have a very big, thick set of freckles on my face and I also have a large, flat set of freckles on my back. I also have three small, round, not flat, round freckles on my chest. Is it possible that I have scarring from previous injuries? I'm asking this because I have an older friend of mine, she was born with similar genetic conditions and it
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use a new policy that will either increase or decrease government spending. The policy will have a total impact on the economy of $x$ dollars and will be implemented for $t$ years. The president is considering two scenarios: 
    
    1. The policy will be implemented immediately, and then the impact will be reflected in the next year.
    2. The policy will not be implemented at all.
    
    If the president wants to minimize the economic impact, which scenario should he choose? Provide a mathematical explanation for your answer.
    
    To determine which scenario minimizes the economic impact, we need to compare the total impact of the policy in
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. Rome
    D. Madrid
    Answer:
    A
    
    The principal form of sexual reproduction in lower plants is ____.
    A. Fission
    B. Budding
    C. Spore reproduction
    D. Fertilization
    Answer:
    C
    
    The function of the clutch is to ____.
    A. Apply and disengage friction between the two halves of the clutch
    B. Generate a torque that can stop the vehicle
    C. Ensure safe operation of the vehicle
    D. Ensure smooth operation of the vehicle
    Answer:
    A
    
    The core content of the Chinese Dream is ____.
    A
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with many promising applications in fields as diverse as healthcare, finance, and even entertainment. AI has the potential to revolutionize how we live and work, but it also raises important questions about privacy, bias, and accountability. As AI technology continues to advance, it is crucial that we address these challenges to ensure that it is used in a way that benefits society as a whole.
    
    One way to address these challenges is through the development of AI ethics guidelines. These guidelines can help us ensure that AI systems are fair, transparent, and accountable. They can also help us understand the ethical implications of AI technology and make informed decisions about its use


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] and [Country]. I'm [Age] years old. I'm [Height] inches tall, [Weight] pounds, and [Hobbies]. I'm [Occupation] at [Company Name]. I'm [Age] years old. I'm [Height] inches tall, [Weight] pounds, and [Hobbies]. I'm [Occupation] at [Company Name]. I'm [Age] years old. I'm [Height] inches tall, [Weight] pounds, and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Academy of Sciences. Paris is a cultural and economic hub, known for its rich history, diverse cuisine, and vibrant nightlife. It is a popular tourist destination, attracting millions of visitors each year. The city is also home to the French Riviera, a popular tourist destination for its beautiful beaches and Mediterranean climate. Paris is a city that has been a center of power and culture for centuries, and continues to be a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration with human intelligence: One of the most significant trends in AI is the increasing integration of AI with human intelligence. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Enhanced privacy and security: As AI systems become more complex and sophisticated, there will be a need for greater privacy and security measures to protect the data and personal information that they collect and process. This
    


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
    Generated text:  [insert first name], and I am a [insert profession or occupation] with [insert qualifications or experience]. I have been working for [insert employer or company name] for [insert number of years] years, and I am currently seeking a new opportunity. I am [insert any relevant skills or experience that is unique to your profession or occupation]. What can I do for you today?
    Remember to include any relevant information that differentiates you from other candidates, and don't include any personal information that could be misleading or damaging to your reputation. Also, make sure to tailor your self-introduction to the specific needs of the company you are
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its stunning architecture and iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. 
    
    (Note: I am assuming you are referring to the capital city of France, Paris.) The city of Paris, also known as Paris, is a metropolitan area in western France, with its capital city being Paris, known for its iconic landmarks and cultural attractions. Paris is the most populous city in France and is home to many of the country's major museums, festivals, and art galleries. It is also home to the Eiffel Tower, a symbol of the city and a major tourist attraction. The French government
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends that are expected to shape the evolution of the field:
    
      1. Increasing reliance on machine learning and artificial intelligence: As machine learning and artificial intelligence continue to advance, the ability of machines to learn from data and make decisions will become more and more sophisticated.
      2. Emergence of more complex cognitive abilities: There is growing evidence that the human brain has cognitive abilities that are not yet fully understood, and that could potentially be harnessed to develop AI that exhibits even greater complexity and creativity.
      3. Greater integration with human decision-making: AI systems are likely to become more integrated


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

    'm

     a

     [

    Type

     of

     Work

    ]

     specialist

    .

     I

    've

     been

     working

     for

     [

    Company

     Name

    ]

     for

     [

    Number

     of

     Years

    ]

     years

     now

    ,

     and

     I

     enjoy

     solving

     problems

     and

     helping

     people

     to

     achieve

     their

     goals

    .

     I

    'm

     known

     for

     my

     creative

     thinking

     and

     ability

     to

     come

     up

     with

     solutions

     that

     go

     beyond

     the

     obvious

    .

     Please

     feel

     free

     to

     ask

     me

     anything

     you

    'd

     like

     to

     know

    .

     Let

    's

     connect

    !

     [

    Your

     Name

    ]

     [

    Optional

     Biography

     or

     Skills

    ]

     [

    Optional

     Image

     or

     Video

    ]

     [

    Optional

     Twitter

     Handle

     or

     LinkedIn

     Profile

    ]

     [

    Optional

     Location

    ]

     [

    Optional

     E

    -mail

    ]

     [

    Optional

     Phone

     Number

    ]

     [

    Optional

     Social

     Media

     Links

    ]

     [

    Optional

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    France

    's

     capital

     city

    ,

     Paris

    ,

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

     and

     Notre

    -D

    ame

     Cathedral

    ,

     as

     well

     as

     its

     vibrant

     culture

     and

     culinary

     scene

    .

     The

     city

    's

     skyline

     is

     filled

     with

     tall

     buildings

     and

     monuments

    ,

     and

     it

     is

     a

     popular

     tourist

     destination

     for

     millions

     of

     visitors

     each

     year

    .

     Paris

     is

     also

     home

     to

     a

     rich

     history

     and

     cultural

     heritage

    ,

     including

     the

     ancient

     Roman

     Forum

     and

     the

     Renaissance

     Pal

    ais

     Royal

    .

     Overall

    ,

     Paris

     is

     a

     beautiful

     and

     diverse

     city

     that

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     Paris

     is

     the

     capital

     city

     of

     France

     and

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

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     there

     are

     many

     potential

     paths

     and

     technologies

     that

     could

     lead

     to

     advancements

     in

     AI

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Ub

    iqu

    itous

     AI

    :

     AI

     could

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     the

     devices

     we

     use

     to

     our

     homes

    .

     It

     could

     make

     our

     lives

     easier

    ,

     like

     voice

     assistants

     for

     our

     phones

    ,

     self

    -driving

     cars

    ,

     and

     more

    .
    


    2

    .

     Human

    -A

    I

     collaboration

    :

     AI

     could

     learn

     to

     improve

     its

     performance

     by

     interacting

     and

     learning

     from

     humans

    ,

     which

     could

     lead

     to

     more

     effective

     AI

     systems

    .
    


    3

    .

     AI

     ethics

    :

     As

     AI

     systems

     become

     more

     autonomous

    ,

     they

     could

     become

     more

     capable

     of

     acting

     in

     certain

     ethical

     situations

    



```python
llm.shutdown()
```
