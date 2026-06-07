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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  3.94it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.85it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.91it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.13it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.13it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.13it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.13it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 22.13it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 22.13it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 22.13it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 22.13it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 22.13it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 22.13it/s] 

    Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 22.13it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 31.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.86 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.83 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.83 GB):   3%|▎         | 2/58 [00:00<00:03, 16.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.82 GB):   3%|▎         | 2/58 [00:00<00:03, 16.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.82 GB):   3%|▎         | 2/58 [00:00<00:03, 16.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=70.82 GB):   3%|▎         | 2/58 [00:00<00:03, 16.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.82 GB):   9%|▊         | 5/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.82 GB):   9%|▊         | 5/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.81 GB):   9%|▊         | 5/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.80 GB):   9%|▊         | 5/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.15it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=68.82 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.72 GB):  21%|██        | 12/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.72 GB):  21%|██        | 12/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.72 GB):  21%|██        | 12/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.71 GB):  21%|██        | 12/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.71 GB):  21%|██        | 12/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.98it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=68.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.68 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=960 avail_mem=68.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.98it/s] Capturing num tokens (num_tokens=960 avail_mem=68.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=896 avail_mem=68.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=832 avail_mem=68.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=768 avail_mem=68.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=704 avail_mem=68.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=640 avail_mem=68.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=640 avail_mem=68.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=576 avail_mem=68.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=512 avail_mem=68.66 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.46it/s]

    Capturing num tokens (num_tokens=480 avail_mem=68.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=448 avail_mem=68.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=416 avail_mem=68.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=416 avail_mem=68.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=384 avail_mem=68.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=352 avail_mem=68.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=320 avail_mem=68.66 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=288 avail_mem=68.66 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=256 avail_mem=68.66 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=256 avail_mem=68.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=240 avail_mem=68.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=224 avail_mem=68.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.83it/s]

    Capturing num tokens (num_tokens=208 avail_mem=68.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=192 avail_mem=68.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=176 avail_mem=68.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=176 avail_mem=68.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=160 avail_mem=68.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=144 avail_mem=68.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=128 avail_mem=68.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=112 avail_mem=68.63 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=96 avail_mem=68.63 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.55it/s] Capturing num tokens (num_tokens=96 avail_mem=68.63 GB):  81%|████████  | 47/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=80 avail_mem=68.63 GB):  81%|████████  | 47/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=64 avail_mem=68.62 GB):  81%|████████  | 47/58 [00:01<00:00, 45.87it/s]

    Capturing num tokens (num_tokens=48 avail_mem=68.62 GB):  81%|████████  | 47/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=32 avail_mem=68.62 GB):  81%|████████  | 47/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=28 avail_mem=68.61 GB):  81%|████████  | 47/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=28 avail_mem=68.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.15it/s]Capturing num tokens (num_tokens=24 avail_mem=68.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.15it/s]Capturing num tokens (num_tokens=20 avail_mem=68.60 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.15it/s]Capturing num tokens (num_tokens=16 avail_mem=68.60 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.15it/s]Capturing num tokens (num_tokens=12 avail_mem=68.60 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.15it/s]Capturing num tokens (num_tokens=8 avail_mem=68.60 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.15it/s] Capturing num tokens (num_tokens=8 avail_mem=68.60 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=4 avail_mem=68.59 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=4 avail_mem=68.59 GB): 100%|██████████| 58/58 [00:01<00:00, 39.96it/s]


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
    Generated text:  Janet Murphy, and I've recently retired from the Department of Chemistry at the University of Houston. I still do research in the Department and write a daily blog at Intellectual Ventures.
    
    ### Dissecting a Metal Ion
    
    Molecular spectroscopy is a way to look at the vibrational states of molecules, where the atoms that make up the molecule vibrate in a way that is intimately related to their electronic structure. I won't go into the details here, but the principle is that a molecule absorbs radiation and then re-emits radiation that is identical in frequency, but with slightly different wavelengths. This process is called "absorption."
    
    The main
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. He has 500 employees. If he wants to give a bonus to each of the employees, how many dollars will he spend on bonuses if each employee's salary is $600000?
    
    To determine how much money the president will spend on bonuses, we need to follow these steps:
    
    1. Identify the number of employees the president has.
    2. Identify the salary of each employee.
    3. Calculate the total bonus amount by multiplying the number of employees by the salary per employee.
    
    Given:
    - The president has 500 employees.
    - Each employee's salary is $60
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. 
    
    The capital of France is the capital city of France. 
    
    The capital of France is the capital of a country.
    
    Can we conclude that the capital of a country is the capital of France? 
    
    Choose from: 1). yes 2). it is not possible to tell 3). no 1). yes
    
    The capital of France is the capital city of France. Therefore, the capital of a country is also the capital city of that country. So, the correct answer is 1) yes. The capital of a country is indeed the capital city of that country. 
    
    In addition, while the capital of a
    ===============================
    Prompt: The future of AI is
    Generated text:  about equality and transparency. AI technology cannot exist in isolation. It is an integral part of technology, but it cannot exist in isolation. This is because AI technology is not just about logic and numbers. The future of AI is about equality and transparency. AI is a tool, not a person. It can be used to do good or bad, and in the right hands, it can bring us closer to equality and transparency. Equality means that everyone is treated with the same level of respect and fairness. Transparency means that information is shared in a way that is fair and unbiased. In today’s world, AI is becoming more and more important.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I love [job title] because [reason for passion]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new experiences and adventures to try. What's your favorite book or movie? I love [book
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its rich history, including the French Revolution and the French Revolution Square. Paris is a popular tourist destination and a major economic center in France. It is home to many world-renowned museums, theaters, and restaurants. The city is also known for its fashion industry and its role in the French Revolution. Paris is a vibrant and diverse city with a rich cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and artificial intelligence: As automation and AI become more prevalent in various industries, we are likely to see an increase in the use of AI in manufacturing, healthcare, transportation, and other sectors. This will lead to the automation of repetitive tasks and the creation of new jobs that require specialized skills.
    
    2. Enhanced privacy and security: As AI becomes more advanced, we are likely to see an increase in the
    


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
    Generated text:  [Name], I'm a [Type of Character], and I'm from [Country]. I'm [Number] of years old.
    I come from a [Country], but I've always been [的性格 or characteristic]. I'm known for my [优势或技能/爱好], and I enjoy [what I like to do]. I also love to [what you enjoy doing] and I'm [what you think of yourself]. I'm really passionate about [what you love doing]. I believe that everyone can achieve something great if they [what you would like to see in someone]. I'm always looking for new experiences and learning new things
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city with the Eiffel Tower, the Louvre, and the Notre-Dame Cathedral. Paris is known for its stunning architecture, rich history, and vibrant culture. It is the largest city in Europe and a popular tourist destination. Many famous people, including Napoleon Bonaparte and Charles II of Portugal, have been born and raised in Paris. The city is a cultural and economic hub, and it is home to many of the world's top museums, art galleries, and restaurants. Paris has become synonymous with luxury and celebrity, and it remains an important cultural and economic center of France. According to the latest population estimates
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and full of possibilities, and it is certain that it will continue to evolve and expand in many ways. Here are some possible future trends in AI:
    
    1. Increased integration with other technologies: The integration of AI with other technologies such as robotics, drones, and natural language processing is expected to increase. This will enable the development of more advanced AI systems that can perform tasks that are currently limited to humans.
    
    2. Greater automation and cognitive capabilities: AI is expected to become more capable of performing tasks that require decision-making and problem-solving. This will lead to more automation and cognitive capabilities in various industries, from healthcare to manufacturing.
    
    3.


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

     am

     a

     [

    Your

     profession

    ,

     such

     as

     "

    teacher

    ,"

     "

    engine

    er

    ,"

     "

    business

     owner

    ,"

     "

    doctor

    ,"

     etc

    .

    ].

     I

     am

     currently

     [

    Your

     age

    ],

     and

     I

     love

     [

    Your

     hobby

    ,

     personal

     interest

    ,

     or

     a

     specific

     skill

    ],

     and

     I

     am

     always

     looking

     for

     new

     challenges

     to

     take

     me

     further

    .

     Thank

     you

    !

     To

     answer

     any

     questions

    ,

     please

     feel

     free

     to

     ask

     me

     anything

    ,

     and

     I

    'm

     here

     to

     provide

     helpful

     and

     informative

     responses

    .

     I

     look

     forward

     to

     chatting

     with

     you

    .

     Good

    bye

    !

     Dear

     [

    Recipient

    's

     Name

    ],
    


    I

     am

     a

     [

    Your

     name

    ]

     at

     [

    Your

     workplace

     or

     organization

    ],

     and

     I

     am

     very

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Re

    ine

     des

     Fle

    urs

    ."

     It

     is

     a

     cultural

    ,

     historical

    ,

     and

     economic

     center

    ,

     and

     serves

     as

     the

     country

    's

     political

     and

     administrative

     capital

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     landmarks

    ,

     including

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

    .

     Paris

     is

     also

     known

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     art

    ,

     including

     its

     iconic

     fashion

     designer

    ,

     Christian

     D

    ior

    .

     The

     city

     has

     a

     diverse

     population

    ,

     with

     over

     

    3

     million

     residents

     and

     a

     rich

     cultural

     and

     artistic

     heritage

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     is

     the

     

    1

    1

    th

     most

     populous

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     significant

     advancements

     in

     several

     key

     areas

    ,

     including

    :
    


    1

    .

     Automation

     and

     re

    config

    urable

     hardware

    :

     Advances

     in

     computing

     power

     and

     micro

    ch

    ips

     are

     expected

     to

     drive

     the

     development

     of

     smarter

    ,

     more

     efficient

     machines

    ,

     such

     as

     AI

    -based

     autonomous

     vehicles

    ,

     drones

    ,

     and

     other

     robots

     that

     can

     perform

     tasks

     with

     minimal

     human

     intervention

    .
    


    2

    .

     Natural

     language

     processing

     (

    N

    LP

    ):

     N

    LP

     is

     expected

     to

     continue

     its

     evolution

    ,

     with

     new

     breakthrough

    s

     in

     understanding

     human

     language

    ,

     language

     generation

    ,

     and

     language

     translation

    .
    


    3

    .

     Explain

    able

     AI

    :

     Advances

     in

     AI

     models

     and

     techniques

     will

     be

     used

     to

     make

     AI

     systems

     more

     transparent

     and

     explain

    able

    ,

     allowing

     users

     to

     better

    



```python
llm.shutdown()
```
