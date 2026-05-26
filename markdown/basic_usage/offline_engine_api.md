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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.31it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.67it/s]

    Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 13.62it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 13.62it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 13.62it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.62it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.62it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.62it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.62it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 17.69it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 17.69it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 17.69it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 17.69it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 17.69it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 17.69it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 17.69it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.41it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.07it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.35 GB):  21%|██        | 12/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.10 GB):  21%|██        | 12/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.34 GB):  21%|██        | 12/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.33 GB):  21%|██        | 12/58 [00:00<00:02, 22.00it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.33 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.33 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.13 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.31 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.31 GB):  31%|███       | 18/58 [00:00<00:02, 18.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.30 GB):  31%|███       | 18/58 [00:00<00:02, 18.32it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.30 GB):  31%|███       | 18/58 [00:00<00:02, 18.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.30 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.27 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.14it/s]Capturing num tokens (num_tokens=960 avail_mem=74.18 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.14it/s] Capturing num tokens (num_tokens=960 avail_mem=74.18 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.56it/s]Capturing num tokens (num_tokens=896 avail_mem=74.28 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.56it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.27 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.56it/s]Capturing num tokens (num_tokens=768 avail_mem=74.27 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.56it/s]Capturing num tokens (num_tokens=768 avail_mem=74.27 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.68it/s]Capturing num tokens (num_tokens=704 avail_mem=74.26 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.68it/s]Capturing num tokens (num_tokens=640 avail_mem=74.25 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.68it/s]Capturing num tokens (num_tokens=576 avail_mem=74.25 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.68it/s]Capturing num tokens (num_tokens=576 avail_mem=74.25 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.53it/s]Capturing num tokens (num_tokens=512 avail_mem=74.24 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.53it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.25 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.53it/s]Capturing num tokens (num_tokens=448 avail_mem=74.24 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.53it/s]Capturing num tokens (num_tokens=448 avail_mem=74.24 GB):  53%|█████▎    | 31/58 [00:01<00:01, 22.70it/s]Capturing num tokens (num_tokens=416 avail_mem=74.22 GB):  53%|█████▎    | 31/58 [00:01<00:01, 22.70it/s]Capturing num tokens (num_tokens=384 avail_mem=74.23 GB):  53%|█████▎    | 31/58 [00:01<00:01, 22.70it/s]Capturing num tokens (num_tokens=352 avail_mem=74.21 GB):  53%|█████▎    | 31/58 [00:01<00:01, 22.70it/s]Capturing num tokens (num_tokens=352 avail_mem=74.21 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.79it/s]Capturing num tokens (num_tokens=320 avail_mem=74.20 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.79it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.19 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.79it/s]Capturing num tokens (num_tokens=256 avail_mem=74.20 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.79it/s]Capturing num tokens (num_tokens=240 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.79it/s]Capturing num tokens (num_tokens=240 avail_mem=74.18 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=224 avail_mem=74.18 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=208 avail_mem=74.19 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=192 avail_mem=74.19 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.20it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.19 GB):  71%|███████   | 41/58 [00:01<00:00, 26.55it/s]Capturing num tokens (num_tokens=176 avail_mem=74.18 GB):  71%|███████   | 41/58 [00:01<00:00, 26.55it/s]Capturing num tokens (num_tokens=160 avail_mem=74.18 GB):  71%|███████   | 41/58 [00:01<00:00, 26.55it/s]Capturing num tokens (num_tokens=144 avail_mem=74.17 GB):  71%|███████   | 41/58 [00:01<00:00, 26.55it/s]Capturing num tokens (num_tokens=128 avail_mem=74.14 GB):  71%|███████   | 41/58 [00:01<00:00, 26.55it/s]Capturing num tokens (num_tokens=128 avail_mem=74.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.61it/s]Capturing num tokens (num_tokens=112 avail_mem=74.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.61it/s]Capturing num tokens (num_tokens=96 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.61it/s] Capturing num tokens (num_tokens=80 avail_mem=74.14 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.61it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.61it/s]Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.64it/s]Capturing num tokens (num_tokens=48 avail_mem=74.13 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.64it/s]Capturing num tokens (num_tokens=32 avail_mem=74.12 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.64it/s]Capturing num tokens (num_tokens=28 avail_mem=74.09 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.64it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.64it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.41it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.41it/s]Capturing num tokens (num_tokens=16 avail_mem=74.10 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.41it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.08 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.41it/s]Capturing num tokens (num_tokens=8 avail_mem=74.08 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.41it/s] Capturing num tokens (num_tokens=8 avail_mem=74.08 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.64it/s]Capturing num tokens (num_tokens=4 avail_mem=74.07 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.64it/s]Capturing num tokens (num_tokens=4 avail_mem=74.07 GB): 100%|██████████| 58/58 [00:02<00:00, 24.86it/s]


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
    Generated text:  Luis Fernández and I'm a communication specialist. I'm currently completing my Master's degree in Communication Studies and I'll be graduating soon. I'd like to share some of the tools that I use to communicate and engage with people in my daily life. I have a keen interest in conveying my message and ideas clearly and effectively. 
    
    My communication skills are essential in my profession and also in my personal life. I believe that effective communication is key to success and for both individuals and organizations. Therefore, I aim to improve my communication skills by reading widely, practicing listening skills, and taking part in online communication forums.
    
    I also take online courses
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military helicopters he should have. He finds out that there are 360 days in a year. He decides that a helicopter must fly for 8 hours a day to be considered productive. If he randomly selects 50 helicopters from the month of July, what is the probability that all of them are productive on July 1, 2023?
    To determine the probability that all 50 randomly selected helicopters are productive on July 1, 2023, we need to follow these steps:
    
    1. **Identify the total number of days in July:**
       July has
    ===============================
    Prompt: The capital of France is
    Generated text:  _________.____
    A. Paris
    B. Nice
    C. Berlin
    D. New York
    Answer:
    
    A
    
    The capital of France is ____.
    A. Paris
    B. Nice
    C. Berlin
    D. New York
    Answer:
    
    A
    
    The capital of France is ____.
    A. Paris
    B. Nice
    C. Berlin
    D. New York
    Answer:
    
    A
    
    Which of the following statements about the capital of France is true? 
    A. Paris
    B. Nice
    C. Berlin
    D. New York
    Answer:
    
    A
    
    The capital of France is ____.
    A. Paris
    
    ===============================
    Prompt: The future of AI is
    Generated text:  highly promising, but there are several challenges and limitations that must be overcome. In this article, we will explore the different areas where AI is currently facing challenges and potential solutions. One of the main challenges is the ethical implications of AI. Ethical AI refers to AI systems that are designed to operate ethically and responsibly, rather than simply being programmed to perform tasks. However, there is a lack of clear guidelines on how to approach ethical AI, and many ethical dilemmas arise.
    Another challenge is the cost of AI. AI systems are expensive to develop, maintain, and operate, and it can be difficult to justify the cost to stakeholders.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title] at [company name]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [favorite hobby or activity]. I'm always looking for new experiences and adventures to try. What's your favorite book or movie? I love [favorite book or movie]. I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic center, with a rich history dating back to the Roman Empire and being a major hub for international trade. Paris is known for its vibrant nightlife, fashion, and art scene, and is a popular tourist destination. The city is also home to many museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has played a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be a growing emphasis on ensuring that they are used ethically and responsibly. This may involve developing new ethical guidelines and standards for AI systems, as well as increasing transparency and accountability in their development and deployment.
    
    2. Greater integration with human intelligence
    


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
    Generated text:  [Name], and I'm [Age]. I come from [Your Country]. I've always had a passion for [Your Hobby] and have always loved learning new things. I have a curious mind and love to explore and discover new things. I am always eager to learn and grow as a person. What is your favorite hobby or activity to keep you happy and relaxed? As a fictional character, my favorite hobby is learning and discovering new things. I find that it is a way to broaden my knowledge and expand my mind, and it is also a great way to keep me motivated and energized. I enjoy reading, trying new foods
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also home to some of the world's most renowned art galleries, such as the Louvre Museum and the Museum of Modern Art (MoMA). Paris is known for its rich culture, cuisine, and entertainment scene, as well as its role in French politics and society. As a result, it has been a major city for over 300 years and is a popular tourist destination for visitors from all over the world.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  extremely dynamic, with various potential trends emerging and transforming the landscape of technology. Here are some of the most likely developments that could occur in the near and long term:
    
    1. Advancements in deep learning and neural networks: As AI technology continues to evolve, we are likely to see more advanced models and algorithms that can handle more complex problems. Deep learning and neural networks are already showing great promise, but there are still many open questions that need to be answered.
    
    2. Increased integration with other technologies: AI will continue to integrate more and more with other technologies, such as IoT, blockchain, and the Internet of Things (IoT). As


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

    'm

     a

     [

    Type

    ]

     [

    Category

    ]

     who

     enjoys

     [

    Occup

    ation

    ]

     and

     has

     a

     passion

     for

     [

    Enjoy

    ment

    ].

     What

     makes

     you

     unique

     among

     your

     peers

    ?
    


    [

    Name

    ]

     is

     a

     [

    Type

    ]

     [

    Category

    ]

     who

     has

     always

     been

     passionate

     about

     [

    Occup

    ation

    ]

     and

     has

     a

     deep

     passion

     for

     [

    Enjoy

    ment

    ].

     What

     makes

     you

     unique

     among

     your

     peers

    ?

     [

    Name

    ]

     is

     not

     only

     an

     expert

     in

     [

    Occup

    ation

    ]

     but

     also

     a

     master

     in

     [

    Enjoy

    ment

    ],

     constantly

     exploring

     new

     techniques

     and

     techniques

    .

     What

     is

     the

     reward

     for

     pursuing

     your

     passion

    ?

     [

    Name

    ]

     believes

     that

     the

     rewards

     for

     pursuing

     one

    's

     passion

     are

     the

     same

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

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

    .

     It

     is

     also

     a

     bustling

     city

     with

     a

     diverse

     culture

     and

     cuisine

    ,

     which

     is

     reflected

     in

     its

     annual

     Bast

    ille

     Day

     celebrations

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

     considered

     the

     most

     important

     city

     in

     Europe

    .

     Additionally

    ,

     Paris

     is

     an

     important

     center

     of

     international

     business

     and

     finance

    ,

     with

     offices

     and

     businesses

     in

     many

     major

     countries

     around

     the

     world

    .

     Lastly

    ,

     it

     is

     home

     to

     a

     wide

     variety

     of

     museums

    ,

     art

     galleries

    ,

     and

     cultural

     events

     that

     appeal

     to

     visitors

     of

     all

     ages

     and

     backgrounds

    .

     Therefore

    ,

     Paris

     is

     one

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     there

     are

     several

     possible

     trends

     that

     could

     emerge

    .

     Here

     are

     some

     possibilities

    :
    


    1

    .

     Increased

     AI

     AI

    ,

     or

     AI

     in

     general

    ,

     is

     likely

     to

     continue

     to

     evolve

     and

     improve

    .

     This

     could

     include

     new

     algorithms

     and

     techniques

     that

     are

     specifically

     designed

     to

     tackle

     complex

     problems

    .
    


    2

    .

     AI

     in

     healthcare

     AI

     is

     likely

     to

     have

     a

     significant

     impact

     on

     the

     healthcare

     industry

    .

     Medical

     devices

     that

     can

     analyze

     and

     interpret

     patient

     data

     in

     real

    -time

     could

     help

     doctors

     make

     more

     accurate

     diagnoses

     and

     develop

     more

     effective

     treatments

    .
    


    3

    .

     AI

     in

     education

     AI

     is

     likely

     to

     play

     a

     more

     significant

     role

     in

     education

    .

     AI

    -powered

     educational

     tools

     and

     platforms

     could

     help

     students

     learn

     more

     efficiently

     and

     effectively

    ,

    



```python
llm.shutdown()
```
