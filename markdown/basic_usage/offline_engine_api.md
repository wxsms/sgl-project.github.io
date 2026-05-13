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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.94it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.94it/s]


    2026-05-13 18:51:52,933 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 18:51:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:40,  4.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 12.97it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 35.89it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 35.89it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 35.89it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 35.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.10it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=75.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.80 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.79 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.77 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.09 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.08 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.83it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=75.07 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.07 GB):  31%|███       | 18/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.07 GB):  31%|███       | 18/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.07 GB):  31%|███       | 18/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.05 GB):  31%|███       | 18/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=960 avail_mem=75.06 GB):  31%|███       | 18/58 [00:00<00:01, 27.65it/s] Capturing num tokens (num_tokens=896 avail_mem=75.06 GB):  31%|███       | 18/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=896 avail_mem=75.06 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=832 avail_mem=75.06 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=768 avail_mem=75.05 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=704 avail_mem=75.05 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.63it/s]

    Capturing num tokens (num_tokens=640 avail_mem=75.05 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=576 avail_mem=75.05 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=576 avail_mem=75.05 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.22it/s]Capturing num tokens (num_tokens=512 avail_mem=75.03 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.22it/s]Capturing num tokens (num_tokens=480 avail_mem=75.05 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.22it/s]Capturing num tokens (num_tokens=448 avail_mem=75.04 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.22it/s]Capturing num tokens (num_tokens=416 avail_mem=75.04 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=384 avail_mem=75.04 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=384 avail_mem=75.04 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=352 avail_mem=75.03 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=320 avail_mem=75.03 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.45it/s]

    Capturing num tokens (num_tokens=288 avail_mem=75.03 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=256 avail_mem=75.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=240 avail_mem=75.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=240 avail_mem=75.02 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=208 avail_mem=75.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=192 avail_mem=74.84 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.37it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.37it/s] Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.78it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.75it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 35.43it/s]


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
    Generated text:  John. I'm a psychology professor at the University of Illinois at Chicago. I love music, and I think you do, too. I have a diverse collection of books, and I collect books in a sense. For example, one of my favorite books is The Story of Books, by John Trahants. It tells a history of books, from their beginnings through to today's three million books available online. My wife and I have a small library. The library is not our only collection. I have books in a spare room on the library desk. It is a little room, and there is a bed on the other side.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. At first, he was very young, but he grew up to be a very wise and successful president. But he didn't have enough time to have a vacation. He had a lot of work to do, but he couldn't have time for himself. One day, he said, "All I have to do is to get 30 years of my life!" Everyone was surprised, but the president didn't have enough time for himself. He said, "All I have to do is to get 30 years of my life!" Everybody was surprised, but the president didn't have enough
    ===============================
    Prompt: The capital of France is
    Generated text:  (      )
    A: Paris
    B: Lille
    C: Lyon
    D: Nice
    To determine the capital of France, we need to identify the official capital of the country. The capital of France is Paris. Let's go through the options step by step:
    
    A: Paris - This is the correct answer because Paris is indeed the capital of France, and it is the official capital.
    
    B: Lille - Lille is the capital of the Kingdom of France, but it is not the official capital.
    
    C: Lyon - Lyon is the capital of the Canton of Lyon, but it is not the official capital.
    
    D
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    Innovation in AI has been on the rise and the future looks bright. In fact, the future of AI is brighter than ever. In this article, we'll explore the future of AI and how it will transform the way we live, work, and interact with the world around us. By the end of this article, you'll be able to make an informed decision on how to utilize AI in your day-to-day life. Get ready to be amazed at the possibilities of this powerful technology! Who knows what amazing things will happen in the future? AI has already shown us how powerful and innovative it can be, and it looks


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or goal]. I'm always eager to learn and grow, and I'm always willing to take on new challenges. I'm confident in my abilities and I'm always ready to help others. I'm a [reason for confidence] and I'm always looking for ways to [action or goal]. I'm excited to meet you and learn more about you. What's your name? What's your occupation? What's your reason for passion
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Quarter. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is also known for its cuisine, including French cuisine, and its fashion industry. The French capital is a vibrant and dynamic city that is a must-visit for anyone interested in French culture and history.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Greater use of machine learning: Machine learning is becoming increasingly powerful, allowing AI to learn from data and improve its performance over time.
    
    3. Enhanced natural language processing: Natural language processing is becoming more advanced, allowing AI to better understand and respond to human language.
    
    4. Increased reliance on AI for decision-making: As AI becomes more integrated with human intelligence, it is likely to become more heavily
    


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
    Generated text:  [Name], and I work as a [Position] at [Company]. I have [Number of Years] years of experience in [Industry] and [Number of Years] years of experience in [Industry]. I have a passion for [Interest or Hobby] that I am always eager to share. I am always looking for new experiences and learning opportunities to grow as a professional and personal. I am a [Nostalgic or Open] person who values relationships and enjoys making new connections. I am confident, capable, and always willing to take on new challenges. My goal is to achieve success and have a fulfilling career that aligns
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Riviera" and "The City of Light." It is a cosmopolitan city with a rich history, culture, and vibrant nightlife, known for its landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, including gourmet restaurants, French wine, and a wide variety of food stalls. It's a popular tourist destination for its beautiful architecture, annual festivals, and romantic evening parties. Paris is a major hub for both domestic and international tourism, with many local businesses and residents living in the city. Its medieval architecture and stunning
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and uncertain, with many potential applications and developments. Here are some possible trends that could shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose diseases, provide personalized treatment plans, and improve patient outcomes. We may see even greater use of AI in the future, with the development of more advanced AI models that can analyze large amounts of medical data more quickly and accurately.
    
    2. Artificial intelligence in transportation: AI is already being used in self-driving cars, and we may see even more widespread use of AI in transportation in the future. Autonomous vehicles could potentially reduce traffic accidents and


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

     Sarah

     and

     I

    'm

     a

     

    2

    5

    -year

    -old

     marketing

     manager

    .

     I

    'm

     a

     creative

     problem

     solver

     with

     a

     keen

     eye

     for

     detail

     and

     a

     strong

     understanding

     of

     consumer

     behavior

    .

     I

    'm

     passionate

     about

     helping

     businesses

     increase

     their

     brand

     awareness

     and

     sales

     through

     effective

     marketing

     strategies

    .

     I

    'm

     also

     an

     expert

     in

     the

     use

     of

     data

     and

     analytics

     to

     inform

     my

     work

    .

     I

     enjoy

     working

     in

     a

     fast

    -paced

     environment

    ,

     using

     my

     creative

     thinking

     to

     solve

     complex

     problems

     and

     come

     up

     with

     innovative

     solutions

    .

     In

     my

     free

     time

    ,

     I

     enjoy

     running

    ,

     reading

     books

    ,

     and

     spending

     time

     with

     my

     family

     and

     friends

    .

     Thank

     you

    .

     What

     are

     some

     examples

     of

     creative

     problem

    -solving

     strategies

     that

     Sarah

     could

     use

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     and

     the

     economic

     and

     cultural

     center

     of

     the

     country

    .

     Its

     history

     is

     closely

     tied

     to

     the

     French

     Revolution

     and

     is

     known

     for

     its

     rich

     culture

    ,

     art

    ,

     and

     historical

     landmarks

    .

     Paris

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

     the

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

     It

     is

     also

     a

     popular

     tourist

     destination

     and

     host

     to

     many

     world

    -f

    amous

     festivals

     and

     events

    .

     Its

     economy

     is

     also

     highly

     developed

     and

     plays

     a

     significant

     role

     in

     French

     society

    .

     With

     its

     diverse

     culture

     and

     historical

     importance

    ,

     Paris

     has

     been

     a

     cultural

     hub

     for

     centuries

     and

     continues

     to

     be

     a

     significant

     city

     in

     France

    .

     It

     is

     also

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     technological

    ,

     societal

    ,

     and

     ethical

     factors

    ,

     as

     well

     as

     ongoing

     research

     and

     development

    .

     Some

     potential

     trends

     that

     are

     currently

     being

     explored

     or

     proposed

     include

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     AI

     is

     likely

     to

     continue

     to

     improve

     and

     expand

     its

     capabilities

    ,

     particularly

     in

     areas

     such

     as

     natural

     language

     processing

    ,

     image

     recognition

    ,

     and

     decision

    -making

    .

     However

    ,

     as

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     may

     also

     see

     its

     limitations

     become

     more

     apparent

    .
    


    2

    .

     Autonomous

     and

     self

    -driving

     vehicles

    :

     Autonomous

     and

     self

    -driving

     vehicles

     are

     likely

     to

     become

     more

     common

     in

     the

     future

    ,

     as

     AI

     technology

     continues

     to

     improve

     and

     becomes

     more

     integrated

     into

     our

     daily

    



```python
llm.shutdown()
```
