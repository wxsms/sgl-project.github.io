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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.41it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.40it/s]


    2026-04-15 20:28:00,690 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 20:28:00] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:31,  5.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:31,  5.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:31,  5.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:31,  5.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:31,  5.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:47,  1.12it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:47,  1.12it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:47,  1.12it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:47,  1.12it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:47,  1.12it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:06<00:47,  1.12it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:06<00:47,  1.12it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:06<00:15,  3.06it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:06<00:05,  6.93it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:06<00:02, 11.82it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]

    Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:06<00:01, 17.81it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:06<00:00, 24.84it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 31.60it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 31.60it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 31.60it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 31.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.13it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.13it/s] Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.90it/s]Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.90it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.90it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.90it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.90it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.90it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]

    Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.18it/s] Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 41.25it/s]

    Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  81%|████████  | 47/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  81%|████████  | 47/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  81%|████████  | 47/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.57it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.57it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.57it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.57it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.57it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.57it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 37.75it/s]


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
    Generated text:  Sonya, a 16-year-old anime girl who is learning to use a computer and has recently learned about machine learning and deep learning. In this task, you are asked to write a program in Python that will display the contents of a CSV file, specifically a "students.csv" file, and its contents as a string. The program should also include a function that will prompt the user to input their name, a list of names, and a list of their favorite subjects, and will use the input to update the contents of the CSV file. The function should also handle exceptions and ensure that the file is opened and read in the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, so he gets a lot of money. Some people think that the president should be paid as much as an actor. But others think that the president should be paid as little as an actor. Most people think that the president should be paid as much as an actor. 
    
    What is the main idea of the passage?
    The passage mainly tells us that people have different opinions about how much money the president of the United States should be paid. Some people think that the president should be paid as much as an actor, while others think that he should be paid as little as an actor. Most people think that the president should
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the river Seine, in the department of Seine-et-Marne.
    A. True
    B. False
    Answer: A
    
    If a user wants to know about the current version of a certain software, which component should they use? ____
    A. Device Manager
    B. Control Panel
    C. Registry
    D. Software Update
    Answer: D
    
    In the following program, what will be printed when running? #include "stdio.h" main() { int a=4,b=3; printf("a=%d,b=%d",a++,b--); } A. 5,2 B. 
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but our current AI tools and environments are flawed and vulnerable. In addition, our data is often incomplete, biased, and overused in our decision-making. Many are looking to new technologies like AI to improve our society and allow us to live better lives.
    
    However, there is a fundamental ethical issue that must be addressed before any new AI tools can be implemented. This issue is the need to ensure that AI algorithms do not perpetuate or exacerbate harmful social biases that harm society as a whole. 
    
    To address this ethical issue, researchers are creating new algorithms that are more unbiased and fair. These new algorithms use machine learning techniques to


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [What I like to do]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. It is a major transportation hub and a major tourist destination, with its famous landmarks and museums attracting millions of visitors each year. Paris is a city of contrasts, with its modern architecture and vibrant nightlife, as well as its traditional French charm and historical significance. The city is also known for its cuisine, with its famous dishes such
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased automation: AI is likely to become more integrated into various industries, leading to increased automation of tasks and processes. This could result in the creation of new jobs, but also potentially leading to job displacement.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be an increased need for measures to protect user data and prevent cyber attacks. This could lead to the development of new privacy and security standards.
    
    3. Enhanced human-computer interaction: AI is likely to become more integrated into human-computer interaction, allowing for more natural and intuitive interactions between humans
    


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
    Generated text:  [Your Name], and I'm a [Role/Job] at [Company Name]. I'm really passionate about [Your Passion/Interest/Responsibility], and I enjoy helping people achieve their goals and make the world a better place. I'm always on the lookout for new challenges and opportunities to grow and learn, and I'm always looking for ways to improve my skills and knowledge. I love to collaborate with others, learn from them, and be open to new ideas and perspectives. I'm a team player, and I'm happy to work with people of all backgrounds and personalities. Whether I'm working on a project alone or with
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    What is the role of Paris in France's economy? Paris is the economic center of France and is the largest city by population. It is a major hub for industries, finance, trade, and tourism. Paris is home to many of the world's leading businesses, including those in the aerospace, automotive, fashion, and media industries. The city is also home to many international companies, including those in the fashion, media, and food industries. Paris is an important economic hub for France, and its economy has been an important driver of growth for much of the country's history. The French economy as a whole is also heavily dependent
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a mix of disruptive and transformative developments that will reshape the way we live, work, and interact with technology. Here are some potential trends that could shape the future of AI:
    
    1. Increased automation and AI-powered service: As AI continues to improve, it will be able to perform more complex tasks and automate repetitive tasks, leading to more efficient and effective service delivery. This could mean that we will see more automation in industries like healthcare, finance, and transportation.
    
    2. Improved healthcare: AI-powered diagnostic tools and predictive analytics could help doctors make more accurate diagnoses and recommend more personalized treatment plans. Additionally, AI-powered chatbots and


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

     am

     a

     professional

     with

     over

     [

    number

    ]

     years

     of

     experience

     in

     the

     [

    industry

    ].

     I

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    industry

    ],

     and

     I

     have

     been

     in

     this

     industry

     since

     [

    date

    ].

     I

     am

     a

     skilled

     professional

     with

     a

     passion

     for

     [

    industry

    ]

     and

     I

     am

     always

     looking

     to

     learn

     and

     grow

    .

     I

     have

     a

     strong

     work

     ethic

     and

     a

     team

    -oriented

     attitude

    .

     I

     am

     always

     eager

     to

     learn

     new

     things

     and

     to

     help

     others

     in

     my

     profession

    .

     I

     am

     confident

     in

     my

     ability

     to

     contribute

     to

     the

     success

     of

     [

    company

    /

    organization

    ],

     and

     I

     am

     committed

     to

     delivering

     exceptional

     results

    .

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    
    
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

     and

     vibrant

     culture

    .


    You

     are

     to

     answer

     this

     question

    :

     Has

     Paris

     been

     selected

     as

     a

     Spaceport

    ?

     To

     answer

     this

     question

    ,

     I

     will

     perform

     the

     following

     steps

    :


    1

    .

     Identify

     the

     space

    port

     in

     Paris


    2

    .

     Check

     if Paris

     is

     considered

     a

     space

    port

    


    3

    .

     If

     yes

    ,

     state

     if

     it

    's

     currently active

     or

     in

     progress

    


    4.

     If

     no

    , explain

     why

     Paris

     is

     not

     a

     space

    port

    


    Step

     

    1

    :

     Ident

    ifying

     the

     space

    port

     in Paris

    


    After

     researching

    ,

     I

     found

     that

     Paris

     is not

     currently

     considered

     a

     space

    port

    .

     However

    ,

     the

     city

     does

     have

     a

     growing

     presence

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    ,

     with

     numerous

     potential

     developments

     and

     advancements

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

    :

     With

     the

     help

     of

     advanced

     machine

     learning

     algorithms

    ,

     AI

     is

     likely

     to

     become

     more

     capable

     and

     efficient

    .

     This

     will

     enable

     AI

     to

     learn

     from

     data

     more

     effectively

    ,

     making

     it

     better

     at

     recognizing

     patterns

     and

     making

     predictions

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     a

     variety

     of

     healthcare

     applications

    ,

     including

     diagnosis

    ,

     treatment

     planning

    ,

     and

     drug

     development

    .

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     to

     be

     used

     in

     even

     more

     precise

     and

     accurate

     ways

    .
    


    3

    .

     Integration

     of

     AI

     into

     consumer

     electronics

    :

    



```python
llm.shutdown()
```
