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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.75it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.74it/s]


    2026-05-08 04:13:23,667 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 04:13:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:44,  1.22it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:44,  1.22it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:44,  1.22it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:44,  1.22it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:44,  1.22it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.55it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 19.11it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 19.11it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 19.11it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 19.11it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 19.11it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 19.11it/s]

    Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 19.11it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 19.11it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 25.36it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 34.70it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 34.70it/s]

    Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 34.70it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 34.70it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 34.70it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 45.95it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 45.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.35 GB):   3%|▎         | 2/58 [00:00<00:04, 13.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.34 GB):   3%|▎         | 2/58 [00:00<00:04, 13.02it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.34 GB):   3%|▎         | 2/58 [00:00<00:04, 13.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.34 GB):   7%|▋         | 4/58 [00:00<00:03, 14.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.24 GB):   7%|▋         | 4/58 [00:00<00:03, 14.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.33 GB):   7%|▋         | 4/58 [00:00<00:03, 14.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.33 GB):  10%|█         | 6/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.31 GB):  10%|█         | 6/58 [00:00<00:03, 16.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.31 GB):  10%|█         | 6/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.30 GB):  10%|█         | 6/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.30 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.30 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.27 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.23 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.61it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.23 GB):  21%|██        | 12/58 [00:00<00:02, 19.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.26 GB):  21%|██        | 12/58 [00:00<00:02, 19.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.26 GB):  21%|██        | 12/58 [00:00<00:02, 19.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.26 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.49it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=73.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.73 GB):  31%|███       | 18/58 [00:00<00:01, 23.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.62 GB):  31%|███       | 18/58 [00:00<00:01, 23.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.57 GB):  31%|███       | 18/58 [00:00<00:01, 23.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.55 GB):  31%|███       | 18/58 [00:00<00:01, 23.44it/s]Capturing num tokens (num_tokens=960 avail_mem=73.56 GB):  31%|███       | 18/58 [00:00<00:01, 23.44it/s] Capturing num tokens (num_tokens=896 avail_mem=73.56 GB):  31%|███       | 18/58 [00:00<00:01, 23.44it/s]Capturing num tokens (num_tokens=896 avail_mem=73.56 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.89it/s]Capturing num tokens (num_tokens=832 avail_mem=73.55 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.89it/s]Capturing num tokens (num_tokens=768 avail_mem=73.54 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.89it/s]Capturing num tokens (num_tokens=704 avail_mem=73.54 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.89it/s]

    Capturing num tokens (num_tokens=640 avail_mem=73.53 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.89it/s]Capturing num tokens (num_tokens=576 avail_mem=73.53 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.89it/s]Capturing num tokens (num_tokens=576 avail_mem=73.53 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.62it/s]Capturing num tokens (num_tokens=512 avail_mem=73.50 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.62it/s]Capturing num tokens (num_tokens=480 avail_mem=73.52 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.62it/s]Capturing num tokens (num_tokens=448 avail_mem=73.51 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.62it/s]Capturing num tokens (num_tokens=416 avail_mem=73.51 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.62it/s]Capturing num tokens (num_tokens=384 avail_mem=73.51 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.62it/s]Capturing num tokens (num_tokens=384 avail_mem=73.51 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.10it/s]Capturing num tokens (num_tokens=352 avail_mem=73.50 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.10it/s]Capturing num tokens (num_tokens=320 avail_mem=73.47 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.10it/s]

    Capturing num tokens (num_tokens=288 avail_mem=73.47 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.10it/s]Capturing num tokens (num_tokens=256 avail_mem=73.46 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.10it/s]Capturing num tokens (num_tokens=240 avail_mem=73.45 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.10it/s]Capturing num tokens (num_tokens=240 avail_mem=73.45 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.02it/s]Capturing num tokens (num_tokens=224 avail_mem=73.47 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.02it/s]Capturing num tokens (num_tokens=208 avail_mem=73.46 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.02it/s]Capturing num tokens (num_tokens=192 avail_mem=73.46 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.02it/s]Capturing num tokens (num_tokens=176 avail_mem=73.45 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.02it/s]Capturing num tokens (num_tokens=160 avail_mem=73.44 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.02it/s]Capturing num tokens (num_tokens=160 avail_mem=73.44 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.14it/s]Capturing num tokens (num_tokens=144 avail_mem=73.43 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.14it/s]

    Capturing num tokens (num_tokens=128 avail_mem=73.43 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.14it/s]Capturing num tokens (num_tokens=112 avail_mem=73.43 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.14it/s]Capturing num tokens (num_tokens=96 avail_mem=73.42 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.14it/s] Capturing num tokens (num_tokens=96 avail_mem=73.42 GB):  81%|████████  | 47/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=80 avail_mem=73.41 GB):  81%|████████  | 47/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=64 avail_mem=73.40 GB):  81%|████████  | 47/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=48 avail_mem=73.40 GB):  81%|████████  | 47/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=32 avail_mem=73.39 GB):  81%|████████  | 47/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=28 avail_mem=73.39 GB):  81%|████████  | 47/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=28 avail_mem=73.39 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.65it/s]Capturing num tokens (num_tokens=24 avail_mem=73.39 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.65it/s]

    Capturing num tokens (num_tokens=20 avail_mem=73.38 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.65it/s]Capturing num tokens (num_tokens=16 avail_mem=73.38 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.65it/s]Capturing num tokens (num_tokens=12 avail_mem=73.38 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.65it/s]Capturing num tokens (num_tokens=8 avail_mem=73.37 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.65it/s] Capturing num tokens (num_tokens=8 avail_mem=73.37 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB): 100%|██████████| 58/58 [00:01<00:00, 31.24it/s]


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
    Generated text:  David. My father was an actor and was successful in the business world. I enjoyed watching his movies and his performances. But I didn't really understand the business world very well. For example, he was quite a businessman and always helping people. He would always tell us to be honest and to not lie. The thing that I didn't understand about business was the way people make money and what the companies were trying to do. What was the company trying to do? And what was the purpose of the company? I went to the University and studied what kind of company was successful and what kind of company was not. I came out with
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. He has to deal with many problems in his work. Some of them are very important. He needs to visit many important places, and he needs to take many important trips. He has to deal with a lot of different people and meet with them. In short, he has a very busy day. He is often away from home for many days. He has to deal with a lot of people. He has to deal with many problems. Sometimes he feels very tired. He is often busy. He has to go on a trip to a foreign country. He has to deal with many people. He has to deal with many
    ===============================
    Prompt: The capital of France is
    Generated text: ____
    A. Paris
    B. Lille
    C. Lorient
    D. Strasbourg
    Answer:
    A
    
    What is the core value of the Internet?
    A. Information sharing
    B. Internet neutrality
    C. Law and order
    D. Information release
    Answer:
    A
    
    Please select the correct Chinese translation: "I won't go to the party until tomorrow."
    A. I will not go to the party tomorrow.
    B. I won't go to the party tomorrow.
    Answer:
    B
    
    Which of the following is NOT a correct way to manage team meetings?
    A. Involve key team members in the planning and
    ===============================
    Prompt: The future of AI is
    Generated text:  in the future, and it is in the hands of the next generation of researchers. AI is not just a technology, it is an approach to solving the complex challenges of today, and it is evolving at an unprecedented pace. The potential applications of AI are boundless, from healthcare to financial services, from education to transportation, and from entertainment to manufacturing. By the year 2025, it is likely that at least 10% of all research jobs will be in AI-related fields. In addition, by 2050, it is estimated that AI will be responsible for over 30% of all research


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. I'm always looking for new opportunities to grow and learn, and I'm always eager to learn more about the world around me. What's your favorite hobby or activity? I love [insert a short, positive description of your favorite hobby or activity]. I also enjoy [insert a short, positive description
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for business, finance, and tourism, and is a popular tourist destination. The city is home to many important institutions such as the French Academy of Sciences and the French National Library. Paris is a city of contrasts, with its rich history and culture on one hand,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective use of resources, as well as more accurate and reliable predictions of human behavior.
    
    3.
    


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
    Generated text:  [Your Name] and I am a [Your Profession] who has been [Your Current Position] for [Your Years in the Industry]. I am always on the lookout for new challenges and opportunities to grow and learn. I believe in the power of collaboration and teamwork, and I am always open to new experiences and ideas. I am a person who values professionalism, punctuality, and efficiency in all that I do. If you have any questions or need assistance, please do not hesitate to reach out to me. Thank you. How about we change the word order of the sentence a bit to make it sound more natural and descriptive? Certainly
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, the iconic French capital, is renowned for its stunning architecture, vibrant culture, and rich history, making it a global cultural and tourist attraction. Known as the "City of Love" due to its romantic vibe, Paris is home to iconic landmarks like the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and more. Its cuisine, such as pastries, cheese, and wine, is also a major draw, making it one of the world's most famous cities. Additionally, Paris is home to several museums, including the Louvre, the Musée d'Orsay, and the Galerie Georges
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and uncertain, with many potential areas of development and innovation. Here are some possible trends in AI that could shape the field in the years to come:
    
    1. AI in Healthcare: AI will play a crucial role in healthcare, particularly in the areas of diagnostics, treatment planning, and disease prediction. AI algorithms will be able to analyze large datasets and provide more accurate and personalized diagnoses than human doctors.
    
    2. Autonomous Vehicles: AI will continue to become more advanced and autonomous, with vehicles able to navigate roads and respond to traffic conditions on their own. Autonomous vehicles will be used for transportation, delivery, and other applications, reducing traffic congestion and


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

    Age

    ]

     year

     old

     who

     works

     as

     a

     [

    Your

     job

     title

    ]

     at

     [

    Company

     name

    ].

     I

     enjoy

     [

    Your

     hobby

    ,

     interest

    ,

     or

     passion

    ],

     and

     I

    'm

     passionate

     about

     [

    Your

     favorite

     hobby

    ,

     interest

    ,

     or

     passion

    ].

     I

     try

     my

     best

     to

     make

     my

     workplace

     a

     [

    Your

     company

     culture

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     [

    Your

     area

     of

     interest

     or

     challenge

    ].

     I

    'm

     always

     here

     to

     help

     those

     around

     me

    ,

     and

     I

     appreciate

     all

     the

     [

    positive

     attributes

     or

     qualities

    ]

     that

     they

     bring

     to

     the

     workplace

    .

     I

    'm

     excited

     to

     meet

     everyone

     and

     contribute

     to

     the

     success

     of

     [

    Company

     name

    ].

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     European

     Union

     and

     is

     the

     largest

     city

     in

     Western

     Europe

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     architecture

    .

     The

     city

     has

     a

     diverse

     population

    ,

     with

     a

     mix

     of

     French

    ,

     African

    ,

     and

     other

     immigrant

     groups

    .

     It

     is

     also

     a

     major

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     annually

    .

     Paris

     is

     home

     to

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

     many

     other

     iconic

     landmarks

    .

     The

     city

     is

     also

     an

     important

     center

     for

     politics

    ,

     culture

    ,

     and

     industry

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     and

     is

     a

     major

     hub

     for

     global

     business

     and

     commerce

    .

     It

     has

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     rapidly

     evolving

     and

     diverse

     area

    ,

     with

     many

     potential

     trends

     and

     developments

     shaping

     how

     this

     technology

     will

     play

     out

     over

     the

     coming

     decades

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

     Personal

    ization

    :

     One

     of

     the

     most

     significant

     future

     trends

     in

     AI

     is

     personal

    ization

    .

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     be

     able

     to

     analyze

     large

     amounts

     of

     data

     to

     identify

     patterns

     and

     preferences

    ,

     allowing

     machines

     to

     tailor

     their

     behavior

     to

     individual

     users

    .

     This

     could

     lead

     to

     more

     efficient

     use

     of

     resources

     and

     improved

     customer

     experiences

    .
    


    2

    .

     Learning

     and

     adapt

    ability

    :

     AI

     will

     continue

     to

     become

     more

     sophisticated

    ,

     with

     the

     ability

     to

     learn

     from

     mistakes

     and

     adapt

     to

     new

    



```python
llm.shutdown()
```
