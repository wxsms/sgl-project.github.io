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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.34it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.34it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:22,  2.34it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:22,  2.34it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:22,  2.34it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:22,  2.34it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]

    Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:08,  5.37it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:03, 11.51it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]

    Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:04<00:01, 18.65it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:00, 27.58it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]

    Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:04<00:00, 36.78it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 47.53it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 47.53it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 47.53it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 47.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.16 GB):   2%|▏         | 1/58 [00:00<00:05,  9.99it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.14 GB):   2%|▏         | 1/58 [00:00<00:05,  9.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   2%|▏         | 1/58 [00:00<00:05,  9.99it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   5%|▌         | 3/58 [00:00<00:04, 11.66it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.37 GB):   5%|▌         | 3/58 [00:00<00:04, 11.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.36 GB):   5%|▌         | 3/58 [00:00<00:04, 11.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.36 GB):   9%|▊         | 5/58 [00:00<00:03, 13.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.35 GB):   9%|▊         | 5/58 [00:00<00:03, 13.31it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.34 GB):   9%|▊         | 5/58 [00:00<00:03, 13.31it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.34 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.33 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.33 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.32 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.32 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.31 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.31 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.41it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.30 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.29 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.28 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.27 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.69it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.24 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.59it/s]Capturing num tokens (num_tokens=960 avail_mem=74.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.59it/s] Capturing num tokens (num_tokens=896 avail_mem=74.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.59it/s]Capturing num tokens (num_tokens=832 avail_mem=74.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.59it/s]Capturing num tokens (num_tokens=768 avail_mem=74.23 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.59it/s]Capturing num tokens (num_tokens=704 avail_mem=74.23 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.59it/s]Capturing num tokens (num_tokens=704 avail_mem=74.23 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.70it/s]Capturing num tokens (num_tokens=640 avail_mem=74.21 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.70it/s]Capturing num tokens (num_tokens=576 avail_mem=74.23 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.70it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.21 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.70it/s]Capturing num tokens (num_tokens=480 avail_mem=74.22 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.70it/s]Capturing num tokens (num_tokens=448 avail_mem=74.21 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.70it/s]Capturing num tokens (num_tokens=448 avail_mem=74.21 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=384 avail_mem=74.20 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=352 avail_mem=74.17 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=320 avail_mem=74.16 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=288 avail_mem=74.16 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=288 avail_mem=74.16 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=256 avail_mem=74.16 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=224 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=208 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=192 avail_mem=74.13 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=192 avail_mem=74.13 GB):  71%|███████   | 41/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=176 avail_mem=74.12 GB):  71%|███████   | 41/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=160 avail_mem=74.12 GB):  71%|███████   | 41/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=144 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=128 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=112 avail_mem=74.10 GB):  71%|███████   | 41/58 [00:01<00:00, 39.37it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=96 avail_mem=74.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.95it/s] Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=48 avail_mem=74.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=24 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=16 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.15it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.15it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.15it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 31.28it/s]


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
    Generated text:  Natasha and I’m a high school teacher at a large school in the Midwest. I have a master's degree in education, which I earned through the University of Illinois at Chicago. I’ve taught middle school science, biology, and earth science. I’m currently teaching high school science and mathematics. I’ve also taught math and computer science at the University of Illinois at Chicago.
    My experience as a teacher includes teaching one-on-one, group instruction, and a variety of advanced courses. My students are often English speakers, students in special education, or students with various learning styles. I believe in a student-centered approach and believe that each student has
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing to spend $300 billion on a new program to reduce poverty. If the program is successful, it is estimated to increase the poverty rate by 10 percentage points. What is the estimated percentage change in the poverty rate if the president's proposal is implemented?
    To determine the estimated percentage change in the poverty rate if the president's proposal is implemented, we can follow these steps:
    
    1. Identify the initial poverty rate.
    2. Identify the increase in the poverty rate due to the program.
    3. Calculate the new poverty rate after the program.
    4. Determine the percentage change in the poverty rate.
    
    The initial poverty rate is
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located in the southeast of the country. It is the most visited city in the world and is the center of government, of culture, of science, of politics, of fashion, of literature and of philosophy.
    
    Paris is situated in the Seine River valley, in the Paris Basin and in the Paris Basin, in the Île de France.
    
    The Paris Basin is a 1600 square-kilometer area bounded by the Paris Basin, the Vosges mountains, the Loire river and the Seine river. It is the most densely populated area in the world and is the birthplace of the French people.
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and rapidly evolving, and at the same time, the notion of AI being a potential threat to humanity has been present for a long time. This post will discuss the current state of AI, its impact on society, and the future implications of AI.
    
    The current state of AI:
    
    AI is now considered a necessary tool for businesses and organizations to achieve their goals. AI systems are used to automate processes, improve efficiency, and deliver services in a more efficient manner. AI can be used for a wide range of applications such as healthcare, finance, transportation, and education.
    
    The impact of AI on society:
    
    AI has the potential to change the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant culture, including its annual Eiffel Tower Festival and its famous fashion and food scenes. The city is a major transportation hub and is home to many of France's major cities and regions. It is a popular tourist destination and is known for its cuisine, art, and music. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more efficient and effective decision-making, as well as more personalized and context-aware interactions.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased pressure to consider the ethical implications of its use. This could lead to more stringent regulations and guidelines for
    


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
    Generated text:  [insert first name], and I am an AI assistant. I was created by [insert name of company], and I am a tool designed to help people with their daily tasks. I have been trained on vast amounts of data, and I can assist in a variety of tasks such as answering questions, providing information, and even assisting in writing. My goal is to help you achieve your goals and make your life easier. How can I help you today? [insert enthusiasm and positive tone] Here’s to a great day! [insert final line of self-introduction] Cheers! [insert positive final line of self-introduction] Cheers!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    This statement is accurate. Paris is the capital city of France and serves as the political, economic, and cultural center of the country. It is also the largest city in the European Union and one of the most visited cities in the world. The city is known for its rich history, architectural wonders, vibrant culture, and access to various regions of the country. As the capital, it plays a crucial role in governance and administration.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by exponential growth in capabilities and applications of AI, driven by continued advances in technology, new data sources, and increased focus on ethical considerations. Some possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used in a variety of healthcare applications, such as personalized medicine, disease prediction, and patient monitoring. As AI becomes more advanced, we can expect to see even more sophisticated applications, such as virtual assistants, robots, and self-driving vehicles.
    
    2. Personalized education: AI is already being used to personalize education, such as by tailoring learning paths based on a student's


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

    Your

     Profession

    ]

     at

     [

    Your

     Organization

    /

    Company

    ],

     and

     I

    'm

     excited

     to

     meet

     you

     today

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     You

    'll

     find

     that

     your

     journey

     with

     us

     is

     different

     from

     any

     you

    've

     ever

     experienced

     before

    .

     What

     brings

     you

     here

    ?

     Can

     you

     tell

     me

     a

     little

     about

     your

     background

    ?

     You

    'll

     find

     that

     as

     we

     learn

     more

     about

     you

    ,

     you

     will

     reveal

     something

     that

     you

    've

     never

     told

     anyone

     before

    .

     So

    ,

     what

    's

     your

     story

    ?

     And

     what

    's

     the

     most

     exciting

     thing

     about

     the

     opportunity

     to

     meet

     you

    ?

     What

     do

     you

     look

     forward

     to

     the

     most

    ?

     I

    'm

     looking

     forward

     to

     spending

     time

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

    .

     It

     is

     one

     of

     the

     oldest

     continuously

     inhabited

     cities

     in

     the

     world

     and

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     culture

    ,

     art

    ,

     and

     fashion

    ,

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     It

     is

     a

     bustling

     city

     with

     a

     diverse

     population

     of

     around

     

    2

    .

    4

     million

     inhabitants

    ,

     and

     is

     known

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     wine

    .

     The

     city

     is

     a

     popular

     tourist

     destination

     and

     is

     home

     to

     many

     museums

    ,

     galleries

    ,

     and

     restaurants

    .

     It

     is

     also

     home

     to

     many

     famous

     landmarks

    ,

     such

     as

     the

     Arc

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     advancements

    ,

     innovation

    ,

     and

     integration

     with

     various

     industries

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

     Increased

     Automation

    :

     AI

     is

     expected

     to

     continue

     autom

    ating

     tasks

     that

     are

     repetitive

    ,

     mundane

    ,

     and

     time

    -consuming

    .

     This

     includes

     tasks

     such

     as

     data

     entry

    ,

     routine

     tasks

    ,

     and

     routine

     maintenance

    .

     Automation

     will

     also

     be

     used

     to

     improve

     efficiency

    ,

     reduce

     costs

    ,

     and

     increase

     productivity

    .
    


    2

    .

     Custom

    ization

     and

     Personal

    ization

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     is

     expected

     to

     be

     able

     to

     understand

     and

     analyze

     user

     behavior

    ,

     preferences

    ,

     and

     habits

    ,

     enabling

     it

     to

     provide

     personalized

     and

     context

    -aware

     experiences

    .

     This

     will

     lead

     to

     more

     accurate

     and

    



```python
llm.shutdown()
```
