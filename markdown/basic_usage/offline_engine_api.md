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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.79it/s]


    2026-05-21 01:05:23,744 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-21 01:05:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.92it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.92it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.91it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.91it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.91it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.91it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.91it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.91it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.91it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 35.83it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 35.83it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 35.83it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 35.83it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 35.83it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 35.83it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.77it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.83it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.73it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 37.46it/s]


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
    Generated text:  Nalini and I'm a computer science student. Recently, I have been studying about object-oriented programming. One concept I'm struggling to understand is inheritance. Could you please explain what inheritance means in the context of object-oriented programming? Inheritance in object-oriented programming allows a class to inherit properties and methods from another class. However, I don't understand how to implement it properly, especially when dealing with complex objects. Could you provide some examples and guidance on how to write a proper inheritance code for a complex object? Sure, I can help you understand inheritance in the context of object-oriented programming. Inheritance is a way to create a
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy and he is often not able to have a proper rest. What does the word "rest" in this sentence most probably mean?
    
    A) A vacation
    B) A nap
    C) A vacation
    D) A rest
    
    C) A vacation
    D) A rest
    D) A rest
    The word "rest" in this sentence most likely means a vacation. The president is often not able to have proper rest because of his busy schedule and the long hours he works. The other options, such as a nap or a vacation, are not as directly related to the meaning of the word "rest" in this context
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. Rome
    D. Berlin
    Answer:
    
    A
    
    For the function $y=\frac{x}{x-1}$, when $x\neq1$, the range of $y$ is ____
    A. $y\geq1$
    B. $y\leq-1$
    C. $y\leq1$
    D. $y\leq-2$
    Answer:
    
    C
    
    Xiaohong's mother bought 30 kilograms of apples at a price of 2 yuan per kilogram. After selling some of the apples, she had
    ===============================
    Prompt: The future of AI is
    Generated text:  predictably what it is, but there are a few things that must be questioned and changed to make sure that the future of AI is a seamless success. One of the main changes that must be made is to make sure that all AI systems and applications are secure from the outset.
    AI systems and applications are an integral part of modern society and they are used in a variety of fields from healthcare to transportation to finance. However, as with any technology, there is the potential for it to be used for negative purposes. In order to ensure that AI systems and applications are secure, it is important to make sure that they are designed to minimize the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your personality or skills]. I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby or activity? I love [insert a short description of your favorite activity]. What's your favorite book or movie? I love [insert a short description of your favorite book or movie]. What's your favorite place to go? I love [insert a short description of your favorite place]. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also the country's largest city and the second most populous. Paris is a cultural and economic center with a diverse population and a rich history dating back to the Roman Empire. The city is known for its art, architecture, and cuisine, and is a major tourist destination. It is also home to many famous landmarks and museums, including the Louvre and the Notre-Dame Cathedral. Paris is a city that has played a significant role in French history and continues to be a major cultural and economic center in the country.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. More integration with other technologies: AI is already being integrated into a wide range of other technologies, including healthcare, finance, and transportation. As these technologies continue to evolve, we can expect to see even more integration between AI and other technologies.
    
    3. Greater use of AI in
    


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
    Generated text:  [Your Name], and I’m [Your Age], [Your Occupation]. I’m originally from [Your Place of Birth]. I’ve lived in various parts of the world, but my most memorable journey was the [Year of Your Birth]. I’m a [Your Strength], [Your Weakness], and [Your Personality]. I’m passionate about [Your Passion] and always strive to [Your Goal]. If you wanted to know more about me, I’d be happy to tell you more about me. [Your Name] [Your Age], [Your Occupation], born in [Your Place of Birth], have you had the opportunity to meet
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and a major political, cultural, and economic center. Paris is known for its historic landmarks, such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also famous for its cuisine, art, and fashion. Paris is a major tourist destination and is home to numerous museums, theaters, and other attractions. The city is also known for its coffee culture, which is a major part of its identity. The city is home to many important political and cultural institutions, such as the French National Radio and Television Network, the French National Library, and the Palace of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a diverse range of trends and developments, including:
    
    1. Increased automation: With the widespread adoption of machine learning and deep learning, AI systems are becoming more adept at performing repetitive tasks and performing complex calculations. This trend is likely to continue as AI becomes more efficient and cost-effective.
    
    2. Enhanced privacy and security: As AI systems are integrated into everyday life, there will be an increasing focus on ensuring that their data is secure and protected. AI systems will need to be designed to respect user privacy and take steps to prevent unauthorized access.
    
    3. Enhanced creativity: AI systems are already being used to generate creative content,


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

     am

     [

    Age

    ]

     years

     old

    .

     I

     am

     a

     [

    Occup

    ation

     or

     Profession

    ]

     who

     has

     always

     been

     passionate

     about

     [

    What

     interests

     you

     in

     life

    ].

     I

     enjoy

     [

    What

     hobby

     or

     activity

    ]

     and

     am

     always

     learning

     new

     things

    .

     I

     believe

     in

     [

    What

     you

     believe

     in

     most

    ],

     and

     I

     strive

     to

     [

    What

     you

     want

     to

     achieve

     in

     life

    ].

     I

     am

     someone

     who

     is

     [

    What

     you

     are

    ].

     I

     am

     [

    Name

    ]

     and

     I

     am

     passionate

     about

     [

    What

     you

     are

     passionate

     about

    ].

     I

     hope

     to

     continue

     growing

     and

     learning

     new

     things

    ,

     and

     to

     be

     someone

     who

     inspires

     others

     to

     do

     the

     same

    .

     I

     am

     excited

     to

     have

     the

     opportunity

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     where

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

    -D

    ame

     Cathedral

     stand

     tall

    .

     It

     is

     the

     largest

     and

     most

     populous

     city

     in

     Europe

    .

     The

     city

     is

     known

     for

     its

     fashion

    ,

     art

    ,

     and

     food

    ,

     as

     well

     as

     its

     rich

     history

     and

     iconic

     landmarks

    .

     Paris

     has

     been

     a

     cultural

     hub

     for

     centuries

     and

     is

     home

     to

     many

     famous

     artists

    ,

     writers

    ,

     and

     musicians

    .

     It

     is

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     atmosphere

     and

     its

     focus

     on

     love

    ,

     love

    ,

     love

    .

     With

     a

     population

     of

     over

     

    2

     million

    ,

     Paris

     is

     a

     major

     economic

     and

     cultural

     center

     in

     Europe

     and

     plays

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     full

     of

     possibilities

    ,

     and

     the

     trends

     that

     are

     shaping

     it

     will

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     data

     availability

    ,

     and

     ethical

     considerations

    .

     Here

     are

     some

     potential

     trends

     that

     may

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     Machine

     Learning

    :

     With

     the

     availability

     of

     massive

     amounts

     of

     data

    ,

     machine

     learning

     algorithms

     are

     becoming

     more

     sophisticated

     and

     capable

     of

     learning

     from

     data

    .

     As

     a

     result

    ,

     AI

     systems

     may

     be

     able

     to

     recognize

     patterns

     and

     make

     decisions

     that

     are

     more

     accurate

     and

     automated

     than

     ever

     before

    .
    


    2

    .

     De

    eper

     Understanding

     of

     Human

     Intelligence

    :

     As

     AI

     continues

     to

     learn

     more

    ,

     it

     may

     be

     able

     to

     better

     understand

     human

     intelligence

     and

    



```python
llm.shutdown()
```
