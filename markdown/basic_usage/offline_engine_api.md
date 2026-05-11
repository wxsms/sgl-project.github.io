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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.62it/s]


    2026-05-11 05:56:52,053 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 05:56:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 23.31it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 33.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:04, 13.98it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:04, 13.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:04, 13.98it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):  10%|█         | 6/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):  10%|█         | 6/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):  10%|█         | 6/58 [00:00<00:03, 15.29it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.18it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 26.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 26.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 26.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  31%|███       | 18/58 [00:00<00:01, 26.66it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.28it/s]Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.28it/s] Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.28it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.28it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.95it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.95it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.95it/s]Capturing num tokens (num_tokens=640 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.95it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.95it/s]

    Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.04it/s]

    Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.83it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.83it/s]

    Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=32 avail_mem=72.45 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.78it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.78it/s]

    Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 32.36it/s]


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
    Generated text:  Tianyi. I'm very happy to be here today. Let's get started with a general question. How do you say "Great! Thank you for your time, Tianyi. " in Chinese?
    
    A. 中国真好！谢谢你的耐心，谢谢。
    
    B. 中国真好！谢谢，谢谢。
    
    C. 中国真好！谢谢，非常感谢。
    
    D. 中国真好！非常感谢，谢谢。
    
    Your choice is:
    
    A. 中国真好！谢谢，谢谢。
    
    B. 中国真好！谢谢，谢谢。
    
    C. 中国真好！谢谢，非常感谢。
    
    D
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He has a very important job. That job is to lead the country. The president is like the leader of the country. Some people think that the president is like a king. He has a lot of power. He can change laws and make important decisions. He can make all the rules of the country. The president is also like a mayor. He leads the city. The president is very important because he leads the country. The president is like a king because he can make all the rules of the country. 
    What question would one ask from this paragraph? The president is like a king because he can make
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Where is the capital of South Africa?
    
    The capital of South Africa is Cape Town.
    
    Is the following statement true?
    "Every night, a person will be able to watch a television show."
    Options:
    - yes
    - no
    
    Let me think through this step-by-step:
    
    1. We need to consider the definition of television shows, which typically involve programming that is broadcast on television.
    2. The statement claims that every night, a person will be able to watch a television show.
    3. However, television shows are not limited to broadcasting on television. They can be broadcast on various forms of media, such as radio, movies
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but the tools for ensuring that AI is used ethically and responsibly are also growing. This morning, we’re going to take a look at some of the leading companies that are working on this important issue.
    On this week’s episode of The AI Podcast, we’re speaking with three leaders in the AI field. They are David Sterne, the CEO of the AI Research Foundation at the University of Oxford; Andrew Winters, the founder and CEO of OpenAI; and Bob Kowalski, the Chief Innovation Officer at the Fraunhofer Institute of Systems and Software Systems.
    We’ll be talking about the ethical implications of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry], and I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling metropolis with a diverse population and is a major center for art, culture, and commerce. It is also home to many famous landmarks and attractions, including the Palace of Versailles and the Champs-Élysées. Paris is a city that has a rich history and continues to be a major cultural and economic center in France. It is also known for its delicious cuisine and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability. This could lead to more robust regulations and standards for AI development and deployment.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to
    


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
    Generated text:  [Name], and I am a/an [role] at [Company]. I started my career at [Previous Company], and I'm currently [Role]. I'm a/an [skill] in my field and I strive to always strive for excellence. What do you think is the most important thing for a career to be successful? As an AI language model, I can't have a personal experience or career history, but based on research, the most important thing for a career to be successful is to find a job that aligns with your skills, interests, and values. This means that you need to be willing to put in the work
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital and largest city of France. It is known for its iconic landmarks, such as the Eiffel Tower and Notre-Dame Cathedral, as well as its rich cultural heritage, stunning architecture, and diverse population. The city is also home to several world-renowned museums, such as the Louvre and the Musée d'Orsay. Paris is a popular tourist destination and a popular subject for literature, film, and art. Its long history and cultural importance make it one of the most important cities in the world. Paris is often considered a symbol of French culture and identity. For more
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting, and there are many possibilities and directions AI is headed in. Here are some possible trends:
    
    1. Increased Robustness and Generalization: The most important future trend for AI will be the development of more robust and generalized AI. This means that AI will be able to learn and adapt to new situations and data, rather than just memorize and reproduce patterns. This will be crucial for applications such as autonomous vehicles, medical diagnosis, and fraud detection.
    
    2. Emphasis on Ethics and Bias: As AI becomes more ubiquitous, there will be more pressure on developers to address ethical concerns and avoid bias. This will require the development


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

     name

    ]

     and

     I

    'm

     a

     [

    insert

     occupation

     or

     background

     here

    ].

     I

    'm

     passionate

     about

     [

    insert

     hobbies

     or

     interests

     here

    ],

     and

     I

     enjoy

     [

    insert

     something

     positive

     about

     myself

     here

    ].

     I

     love

     to

     [

    insert

     something

     positive

     about

     my

     personality

     here

    ],

     and

     I

     believe

     in

     [

    insert

     something

     positive

     about

     my

     beliefs

     here

    ].

     I

     believe

     that

     [

    insert

     something

     positive

     about

     my

     values

     here

    ],

     and

     I

     enjoy

     [

    insert

     something

     positive

     about

     my

     work

     ethic

     here

    ].

     I

    'm

     excited

     to

     [

    insert

     what

     I

     hope

     to

     achieve

     here

    ],

     and

     I

    'm

     looking

     forward

     to

     [

    insert

     anything

     that

     makes

     me

     happy

     here

    ].

     I

    'm

     a

     [

    insert

     type

     of

     person

     here

    ]

     and

     I

     thrive

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

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

    .

     
    


    Note

    :

     This

     answer

     is

     fact

    ually

     correct

     and

     provides

     a

     brief

     description

     of

     Paris

    's

     history

    ,

     culture

    ,

     and

     popular

     attractions

    .

     It

     does

     not

     contain

     any

     personal

     opinions

     or

     speculation

    .

     Please

     let

     me

     know

     if

     you

     need

     any

     other

     information

     or

     if

     there

     is

     anything

     else

     I

     can

     assist

     you

     with

    .

     
    


    Paris

    :

     The

     City

     of

     Love

    


    By

     Mark

     De

    egan

    


    *

     *

     *
    


    Today

    ,

     Paris

     is

     a

     buzzing

     met

    ropolis

     that

     reflects

     the

     diversity

     of

     its

     citizens

    '

     cultures

    .

     But

     it

     began

     as

     a

     small

     community

     with

     a

     population

     of

     just

     over

     

    1

    0

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     potential

    ,

     but

     it

     is

     impossible

     to

     predict

     with

     certainty

     what

     the

     world

     will

     look

     like

     in

     the

     next

     decade

     or

     more

    .

     Here

     are

     some

     of

     the

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Improved

     transparency

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     we

     will

     see

     a

     greater

     emphasis

     on

     transparency

     in

     their

     design

     and

     operation

    .

     This

     means

     that

     we

     will

     see

     more

     detailed

     and

     detailed

     documentation

     of

     the

     algorithms

     and

     models

     used

     to

     make

     decisions

    ,

     as

     well

     as

     more

     public

     discussion

     about

     how

     they

     work

    .
    


    2

    .

     Increased

     automation

    :

     With

     the

     rise

     of

     automation

    ,

     we

     can

     expect

     to

     see

     more

     AI

    -powered

     systems

     take

     over

     some

     tasks

     that

     are

     currently

     performed

    



```python
llm.shutdown()
```
