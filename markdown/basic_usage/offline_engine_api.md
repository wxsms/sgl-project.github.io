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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.34it/s]


    2026-05-13 23:45:37,644 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 23:45:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.76it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.73it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.92it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 41.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 17.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 19.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 19.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 19.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 19.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.33it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.84it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.16it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.20it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.83it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.83it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.83it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.83it/s]

    Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.83it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.66it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.66it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.66it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.66it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.66it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.42it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.42it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.42it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.42it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.42it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.75it/s]

    Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.49it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.49it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.49it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.49it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.49it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.37it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.37it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.37it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.37it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.37it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 29.34it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 29.34it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 29.34it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 29.34it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 28.68it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 28.68it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.68it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.68it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:02<00:00, 27.79it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:02<00:00, 27.79it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  97%|█████████▋| 56/58 [00:02<00:00, 27.79it/s]

    Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:02<00:00, 26.80it/s]


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
    Generated text:  Elva and I am a 16 year old girl. I am from an African country and I grew up in the country. I feel like I have a right to be a part of this country, but I have to deal with discrimination and hate. I was born in the country but I was raised in a different country. My home was in a different town. I have a school education but I do not feel very prepared for this country. I am the only person in my family. I was very good at sports, but I was unsuccessful in my studies. I am going to go to college in the United States, and
    ===============================
    Prompt: The president of the United States is
    Generated text:  very popular with the public. As a result, his government has many programs that help the people. They help people who are sick, poor, old, and in need of other help. The government of the United States is often called the "big brother" of a country. 
    
    What does the government of the United States do? Government is a group of people who make laws and take care of the people. The government of the United States is made up of many different parts. 
    
    1. What is the name of the country that the United States is part of? It is called the United States. 
    
    2. What is the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located in the center of the country, in the southeast of the French Alps, between the Indien and the Atlantic Ocean. It’s the country’s second largest city, after the capital of Germany.
    The city was founded in the 8th century by a Viking invasion. In the Middle Ages, Paris was one of the most important European cities, and it was then, the largest city in Europe. However, in the 16th century, the French Revolution started, which made Paris the most important city in the world. From 1789 to 1799, there was a period of 
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it will have a profound impact on the world around us. From personal assistants to self-driving cars, AI is revolutionizing the way we live, work, and interact with each other. But what happens to the data we collect from these advanced technologies? With the increasing reliance on AI, we are constantly running out of space for storing and processing vast amounts of data. In this blog post, we will explore the implications of the data we produce from AI on our planet.
    As AI becomes more advanced, the amount of data it produces is also increasing. According to a study by IBM, the amount of data generated by AI systems


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


    Generated text:  [Name] and I am a [occupation] who has been [number of years] in the industry. I am passionate about [reason for interest in the industry]. I am always looking for new opportunities to [reason for interest in the industry]. I am [number of years] in the industry and have [number of years] of experience in [occupation]. I am a [number of years] in the industry and have [number of years] of experience in [occupation]. I am a [number of years] in the industry and have [number of years] of experience in [occupation]. I am a [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the country's cultural and political center. Paris is a vibrant and diverse city with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major economic and financial center in Europe. The city is home to many famous museums, theaters, and restaurants, and is known for its fashion and food scenes. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. Enhanced personalization: AI will continue to improve our ability to personalize our experiences with technology. This will involve the use of machine learning algorithms to analyze user
    


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
    Generated text:  [name]. I am a [occupation] and I enjoy [reason for being there]. I am [age] years old and have [number of years] years of experience in [occupation]. I am [gender] and I am [religion]. I am [language] and [country of origin]. I am [height] inches tall and [weight] pounds heavy. I have [number of tattoos] tattoos on my body and I enjoy [reason for tattoos]. I have a [physical characteristic] and I am [personality] and [habits]. I am [place of origin] and I am [language]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city is the cultural and political center of the country. It is known for its stunning architecture, rich history, and lively nightlife. The city is also famous for its annual music festival, the Ode to Joy. The city is home to many museums, including the Louvre, and is the capital of the French Riviera. Paris is also known for its wine and cheese, and is a popular destination for tourists and locals alike. French people love the city and it is considered one of the most important cities in the world. Paris is an important center of French culture and is a UNESCO World Heritage Site. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and rapidly evolving, with many potential developments and applications. Here are some possible future trends in AI:
    
    1. Autonomous vehicles: Self-driving cars are expected to become more common and widespread, leading to significant benefits in transportation and efficiency.
    
    2. Wearable technology: Wearable devices will become increasingly popular, offering features like health monitoring, fitness tracking, and remote monitoring of medical conditions.
    
    3. Smart homes: Home automation and smart home devices will become more prevalent, with automated lighting, heating, and security systems.
    
    4. Language translation and interpretation: AI will be able to translate and interpret human language, making communication more accessible and efficient


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

     Emily

    ,

     I

    'm

     a

     person

     with

     a

     passion

     for

     writing

     and

     photography

    .

     Photography

     has

     been

     my

     true

     love

     since

     I

     was

     a

     child

    .

     I

     love

     capturing

     the

     essence

     of

     the

     moment

    ,

     and

     I

     believe

     in

     the

     power

     of

     images

     to

     transport

     us

     to

     another

     world

    .

     I

     have

     an

     eye

     for

     detail

     and

     a

     keen

     sense

     of

     color

    .

     I

    'm

     also

     a

     big

     believer

     in

     the

     importance

     of

     preserving

     and

     sharing

     what

     I

    'm

     learning

     and

     experiencing

     with

     others

    .

     I

     love

     sharing

     my

     work

     with

     friends

     and

     family

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     express

     myself

     through

     my

     photography

     and

     writing

    .

     I

    'm

     a

     proud

     member

     of

     the

     indie

     film

     and

     photography

     community

     and

     love

     to

     explore

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     light

     and

     music

    .

     It

     is

     renowned

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

     the

     Lou

    vre

     Museum

    ,

     and

     Notre

     Dame

     Cathedral

    ,

     as

     well

     as

     for

     its

     rich

     cultural

     heritage

     and

     contemporary

     art

     scene

    .

     The

     city

     also

     has

     a

     diverse

     population

    ,

     including

     people

     of

     different

     national

    ities

    ,

     religions

    ,

     and

     cultures

    ,

     making

     it

     a

     welcoming

     and

     multicultural

     neighborhood

    .

     Paris

     is

     a

     vibrant

     and

     exciting

     city

     with

     a

     vibrant

     nightlife

     and

     numerous

     museums

     and

     galleries

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     It

     is

     known

     for

     its

     gastr

    onomy

     and

     cuisine

    ,

     with

     its

     famous

     dishes

     like

     cro

    iss

    ants

    ,

     ch

    ou

    cr

    oute

    ,

     and

     past

    is

    .

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     rapidly

     evolving

    ,

     with

     many

     possibilities

     for

     the

     technology

     to

     advance

     in

     a

     variety

     of

     ways

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

     Machine

     learning

     will

     become

     more

     sophisticated

    :

     As

     AI

     continues

     to

     learn

     and

     improve

    ,

     we

     can

     expect

     it

     to

     become

     more

     adept

     at

     recognizing

     patterns

     and

     making

     predictions

     based

     on

     large

     amounts

     of

     data

    .

     This

     could

     lead

     to

     the

     development

     of

     more

     advanced

     and

     accurate

     models

    ,

     such

     as

     those

     that

     can

     recognize

     faces

     in

     photos

    ,

     identify

     medical

     images

    ,

     and

     even

     make

     decisions

     based

     on

     a

     variety

     of

     factors

    .
    


    2

    .

     AI

     will

     become

     more

     widespread

    :

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     we

     can

     expect

     it

     to

     become

     more

     integrated

     into

    



```python
llm.shutdown()
```
