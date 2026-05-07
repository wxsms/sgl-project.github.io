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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]


    2026-05-07 02:34:56,116 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 02:34:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:37,  4.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:37,  4.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:37,  4.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:37,  4.86s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:37,  4.86s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.49it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.49it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.49it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.49it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.49it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.49it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.49it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.49it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:10<00:11,  2.59it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:10<00:11,  2.59it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:10<00:11,  2.59it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:10<00:11,  2.59it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:10<00:11,  2.59it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:10<00:11,  2.59it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:11<00:11,  2.59it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:11<00:11,  2.59it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:11<00:11,  2.59it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]

    Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:11<00:05,  4.11it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:11<00:01,  6.72it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:11<00:00, 10.22it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:11<00:00, 10.22it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:11<00:00, 10.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.07 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.07 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]

    Capturing num tokens (num_tokens=960 avail_mem=71.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s] Capturing num tokens (num_tokens=960 avail_mem=71.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=896 avail_mem=71.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=832 avail_mem=71.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=768 avail_mem=71.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=704 avail_mem=71.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=640 avail_mem=71.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=640 avail_mem=71.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=576 avail_mem=71.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=512 avail_mem=71.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=480 avail_mem=71.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=448 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.99it/s]

    Capturing num tokens (num_tokens=416 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=416 avail_mem=71.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=384 avail_mem=71.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=352 avail_mem=71.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=320 avail_mem=71.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=240 avail_mem=71.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=224 avail_mem=71.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.52it/s]

    Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=160 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=112 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=96 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.80it/s] Capturing num tokens (num_tokens=96 avail_mem=70.98 GB):  81%|████████  | 47/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=80 avail_mem=70.97 GB):  81%|████████  | 47/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=64 avail_mem=70.97 GB):  81%|████████  | 47/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 44.69it/s]

    Capturing num tokens (num_tokens=32 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=28 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=28 avail_mem=70.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=24 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=20 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s] Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=4 avail_mem=70.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=4 avail_mem=70.94 GB): 100%|██████████| 58/58 [00:01<00:00, 39.69it/s]


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
    Generated text:  Alex. I'm very excited to meet you here in this beautiful city of Rome. Here we'll talk about some of the most famous landmarks in the city, as well as some interesting places you should definitely visit in the city. 
    As you can see, Rome is a city of history and culture, and it's a city where you can experience the best of both worlds - the modernity and the heritage of the past. As a matter of fact, you can even find themselves in the middle of the Italian countryside and be transported to the past.
    When I first visited this city, I was thoroughly impressed with the beauty of the city
    ===============================
    Prompt: The president of the United States is
    Generated text:  a major employer in many countries, and they have a very important position. They are responsible for making decisions that affect millions of people on a daily basis. They are also responsible for ensuring that the country is safe and secure. This is a very important job, and it is very hard to be a president. To be a president, you need to have a lot of skills and qualifications. The most important skill is probably being able to communicate clearly and effectively with people. Being able to read and understand the news is also very important. Another important skill is being able to make decisions. The president needs to be able to make decisions that will
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the capital of France. Paris is the capital of France.
    You are a world class trivia AI. Provide a detailed answer so the user understands everything better. Sure, I'd be happy to provide a detailed answer about the capital of France, Paris. Let's begin!
    
    ### The Capital of France: Paris
    
    **Location:**
    The capital of France, Paris, is located in the south-central region of France. It is situated in the Hauts-de-France region, which spans from the lowlands to the highlands of the region.
    
    **Physical Geography:**
    Paris is situated in the heart of the Paris Basin
    ===============================
    Prompt: The future of AI is
    Generated text:  bright with more and more companies using it to improve their products and services. For those who want to get started, there are a few key steps to follow. First, identify your goals and objectives. Then, create a clear vision for the AI that will support those goals. Next, determine the necessary tools and technologies to develop your AI. Finally, identify a team of people who can help you implement and optimize your AI project. This will take time and effort, but with the right resources and dedication, you can make significant progress towards your goals. As AI technology continues to evolve, the potential applications and benefits will only grow. So,


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your interests and what you're looking for in a job. What can I help you with today? [Name] is looking for a [Job Title] at [Company Name]. I'm excited to hear about your experience and what you're looking for in a job. [Name] is looking for a [Job Title] at [Company Name]. I'm excited to hear about your experience and what you're looking for in a job. [Name] is looking for a [Job Title] at [Company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its cuisine, fashion, and art scene. Paris is a popular tourist destination and a cultural hub for France and the world. It is home to many world-renowned museums, theaters, and landmarks. Paris is a city that has a unique blend of history, art, and culture, making it a must-visit destination for anyone interested in France. 
    
    Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in more complex and personalized ways, with the goal of improving the quality of care for patients.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As AI becomes more advanced, it is likely to be used in more complex and personalized
    


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
    Generated text:  [First Name] and I am [Last Name] a versatile, empathetic, and skilled [occupation or profession] who has always been a go-to resource for anyone in need. I am a wordsmith and storyteller, able to craft compelling narratives and convey complex ideas in a way that is easy to understand. Whether it's tackling tough conversations or solving complex problems, I'm here to lend a listening ear and a helping hand. My goal is to make people feel understood and heard, and I believe that anyone can find strength in my experiences and insights. Thank you for considering me for your next project! [Your Name] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and the world's seventh-largest city by population. Paris is known for its historical landmarks such as the Eiffel Tower and Louvre Museum, and for its vibrant arts and culture scene. The city is also home to the Eiffel Tower, the Louvre Museum, the Arc de Triomphe, the Notre-Dame Cathedral, and many other notable structures and attractions. As the seat of government and the largest city in France, Paris plays a crucial role in the country's culture, economy, and politics. It is the oldest capital city in the world and a major destination
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to involve rapid advancements in both hardware and software, with a focus on developing more sophisticated and intelligent machine learning algorithms. Here are some possible future trends in AI:
    
    1. Increased focus on ethics and transparency: AI systems are becoming more complex and complex, and the ethical implications of how AI is used are becoming more significant. There will be a growing focus on the ethical implications of AI, including issues such as bias, privacy, and accountability.
    
    2. AI for healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and the potential for AI to become more sophisticated in the future will be significant. AI systems could


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

    职业

    ]

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     [

    occupation

    ].

     I

     enjoy

     [

    job

     responsibilities

    ],

     and

     I

    'm

     passionate

     about

     [

    personal

     interest

     or

     hobby

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     open

     to

     challenges

     and

     opportunities

     to

     improve

     myself

    .

     I

     am

     patient

    ,

     organized

    ,

     and

     always

     strive

     to

     be

     a

     good

     team

     player

    .

     I

     am

     confident

     in

     my

     abilities

     and

     I

     am

     determined

     to

     succeed

     in

     whatever

     I

     set

     out

     to

     do

    .

     Please

     let

     me

     know

     if

     you

     would

     like

     to

     schedule

     a

     call

     to

     chat

     more

     about

     how

     I

     can

     help

     you

     achieve

     your

     goals

    .

     [

    Your

     Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     historical

     landmarks

     and

     cultural

     significance

    .

     It

     is

     located

     on

     the

     Mediterranean

     coast

     and

     is

     one

     of

     the

     largest

     cities

     in

     the

     world

     by

     population

    .

     Paris

     is

     famous

     for

     its

     art

    ,

     architecture

    ,

     and

     cuisine

    ,

     as

     well

     as

     for

     being

     the

     birth

    place

     of

     many

     famous

     French

     figures

    ,

     including

     Napoleon

     Bon

    ap

    arte

     and

     É

    d

    ou

    ard

     Gl

    is

    .

     The

     city

     is

     also

     home

     to

     many

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    ’

    Or

    say

    ,

     and

     hosts

     numerous

     festivals

     and

     cultural

     events

     throughout

     the

     year

    .

     It

     is

     an

     important

     center

     for

     politics

    ,

     culture

    ,

     and

     entertainment

     in

     France

     and

     has

     played

     a

     key

     role

     in

     shaping

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     revolutionary

    ,

     with

     numerous

     potential

     applications

     and

     developments

     shaping

     the

     landscape

     of

     modern

     technology

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     are

     likely

     to

     shape

     the

     industry

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     social

     concerns

    :

     AI

     is

     already

     making

     significant

     strides

     in

     areas

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     law

     enforcement

    ,

     but

     there

     is

     still

     much

     to

     be

     done

     to

     ensure

     that

     it

     is

     used

     eth

    ically

     and

     responsibly

    .

     As

     AI

     becomes

     more

     integrated

     into

     everyday

     life

    ,

     there

     is

     likely

     to

     be

     increased

     focus

     on

     addressing

     social

     and

     ethical

     concerns

     such

     as

     bias

    ,

     privacy

    ,

     and

     human

     rights

    .
    


    2

    .

     Adv

    ancements

     in

     machine

     learning

     algorithms

    :

     AI

     algorithms

     are

    



```python
llm.shutdown()
```
