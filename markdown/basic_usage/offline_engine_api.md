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


    2026-05-12 05:49:35,494 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 05:49:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.93it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.93it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.93it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.66it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 16.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 16.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.74 GB):   9%|▊         | 5/58 [00:00<00:02, 20.00it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=76.74 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.74 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.74 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.64 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.15it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:01<00:02, 16.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:01<00:01, 20.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:01<00:01, 20.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:01<00:01, 20.69it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:01<00:01, 20.69it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:01<00:01, 20.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.33it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.33it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.33it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.33it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.33it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.33it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.69it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]

    Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.49it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.49it/s]

    Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.86it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.86it/s]

    Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:02<00:00, 28.95it/s]


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
    Generated text:  Alex and I am a 25 year old male. I was told that the only thing I have to do is to eat more vegetables and exercise more to lose weight. This made me feel a little scared at first but I was surprised at how easy it was to follow the instructions. 
    
    I was wondering if there are any other reasons why I should eat more vegetables and exercise more? What are the health benefits of eating more vegetables and exercising more? I also wondered what would happen if I continued to follow these habits and achieved the results. The more you follow these habits, the more weight you will lose? I also wanted to know
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man, what is the probability that the president of the United States is female?
    To determine the probability that the president of the United States is female, we need to know the number of women who have held the position of President of the United States and the total number of women who have held the position. Without specific data, we cannot calculate the exact probability. However, we can state the general approach to solving this problem.
    
    1. Identify the number of women who have held the position of President of the United States.
    2. Identify the total number of women who have held the position of President of the United States.
    3. Use
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. Brussels
    C. London
    D. Vienna
    
    To determine the capital of France, let's consider the typical capital cities of European countries. The capital of France is typically Paris. Here is the reasoning:
    
    1. Paris is the capital of France.
    2. The capital of France is a major European city.
    3. Commonly, French capitals are named after the country or its most important city, such as Paris after the French Revolution, or Brussels after the Belgian city.
    4. No other city in Europe has been commonly named as the capital after the country it is located in.
    
    Given these
    ===============================
    Prompt: The future of AI is
    Generated text:  in its diversity, and it is important for the ethical consideration of the world to be represented in the AI algorithms. The advancements in AI are enabling us to see new possibilities and work towards more ethical and responsible practices. However, the widespread misuse of AI technologies can have devastating impacts on society and the environment.
    
    AI is an incredibly powerful tool that can be used for good or for bad, depending on how it is designed. However, the responsibility of using AI in a responsible way rests with us as individuals, organizations, and governments. This means that we must ensure that we are using AI ethically and responsibly, and that we are making sure


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and fashion, and is home to many world-renowned museums, theaters, and restaurants. The city is also known for its cuisine, with dishes like croissants, escargot, and bouillabaisse being popular among locals and tourists alike. Paris is a city of contrasts, with its modern architecture and cultural heritage blending with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced human-machine collaboration: AI is likely to become more integrated with human-machine collaboration, allowing machines to work alongside humans in a more collaborative and efficient manner.
    
    3. Increased ethical considerations: AI is likely to face increased ethical considerations, with concerns about bias, transparency, and accountability.
    
    4. Advancements in AI technology: AI technology is likely to continue to advance, with new algorithms and techniques being developed to improve performance
    


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
    Generated text:  [Name], and I'm a [job title or occupation] with [number] years of experience. I'm passionate about [why you're passionate about your job], and I thrive on [why you're passionate about your job]. I'm a [specific skill or interest] and I'm always seeking to learn more about [relevant topic]. I'm always looking for ways to [what you do to help others, or how you make a positive impact]. I'm also [what you do to make your character stand out, or what sets you apart]. I love [why you love your job, or why you enjoy your work].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris, the historical and cultural heart of the country, is often referred to as the "City of Love" and is famous for its architecture, art, and iconic landmarks such as Notre-Dame Cathedral and the Louvre Museum. Its vibrant atmosphere and charming French culture make it a popular tourist destination. Paris is also known for its food and wine, with its famous Paris steak, ciabatta sandwiches, and the renowned red roux coffee. With its annual fashion and art fairs, Paris is a hub of creativity and entertainment in the world. The city's use of technology and innovation in transportation and energy production is a key
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by an increase in the sophistication and complexity of AI systems. Some potential trends include:
    
    1. AI will become more personalized: AI systems will be able to learn from the data and preferences of individual users or individuals, making them more personalized and relevant to their needs.
    
    2. AI will be used in more areas: AI is already being used in areas such as healthcare, finance, and transportation, but it is likely that AI will be used in more diverse and specialized areas as it becomes more widely adopted.
    
    3. AI will be integrated into more devices: As more devices become connected to the internet, AI will become more


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

    'm

     an

     experienced

     [

    occupation

     or

     profession

    ]

     with

     over

     [

    number

    ]

     years

     of

     experience

     in

     [

    industry

    ]

     and

     [

    language

    ].

     I

     have

     a

     deep

     appreciation

     for

     the

     challenges

     and

     opportunities

     in

     [

    field

    ]

     and

     a

     strong

     drive

     to

     continue

     learning

     and

     growing

    .

     I

    'm

     dedicated

     to

     [

    purpose

     or

     goal

    ]

     and

     am

     always

     eager

     to

     push

     myself

     to

     achieve

     it

    .

     If

     you

     need

     advice

    ,

     support

    ,

     or

     a

     chance

     to

     learn

     more

     about

     my

     background

    ,

     please

     don

    't

     hesitate

     to

     reach

     out

     to

     me

    .

     At

     heart

    ,

     I

     am

     [

    persona

    ].

     You

     can

     trust

     me

     to

     be

     a

     reliable

    ,

     trustworthy

    ,

     and

     trustworthy

     friend

     to

     you

    .

     So

    ,

     what

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

    ,

     and

     is

     the

     largest

     city

     in

     both

     Europe

     and

     the

     world

    .

     The

     city

     is

     famous

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culture

    ,

     and

     has

     been

     a

     major

     economic

     hub

     since

     the

     

    1

    4

    th

     century

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

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     also

     has

     a

     diverse

     population

    ,

     with

     many

     French

    -speaking

     communities

     and

     immigrants

     from

     around

     the

     world

    .

     Paris

     is

     a

     cultural

     and

     intellectual

     hub

    ,

     attracting

     tourists

     from

     around

     the

     world

     and

     hosting

     numerous

     festivals

     and

     events

     throughout

     the

     year

    .

     Its

     status

     as

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     for

     significant

     changes

    ,

     driven

     by

     the

     ever

    -growing

     capabilities

     and

     applications

     of

     artificial

     intelligence

    .

     Here

     are

     some

     potential

     future

     trends

    :
    


    1

    .

     **

    Increased

     Integration

     and

     Complexity

    **:

     AI

     systems

     are

     becoming

     more

     integrated

     with

     hardware

     and

     software

    ,

     and

     their

     complexity

     is

     increasing

    .

     This

     integration

     will

     likely

     lead

     to

     the

     development

     of

     new

     AI

     architectures

     and

     frameworks

     that

     can

     handle

     more

     complex

     tasks

    .
    


    2

    .

     **

    Deep

     Learning

     and

     Neural

     Networks

    **:

     Deep

     learning

     and

     neural

     networks

     will

     continue

     to

     advance

    ,

     making

     them

     more

     efficient

    ,

     accurate

    ,

     and

     capable

     of

     handling

     increasingly

     complex

     tasks

    .

     These

     advancements

     will

     also

     lead

     to

     the

     development

     of

     new

     AI

     models

     that

     can

     learn

     from

     data

     in

     ways

     that

     are

     not

     possible

     with

    



```python
llm.shutdown()
```
