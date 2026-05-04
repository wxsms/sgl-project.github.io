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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]


    2026-05-04 18:10:14,903 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 18:10:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.67it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.68it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]

    Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 25.96it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 35.19it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 35.19it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 35.19it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 35.19it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 35.19it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 35.19it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 35.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.17it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.12it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.12it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.12it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.12it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.12it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.12it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.70it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.70it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.70it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.70it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.70it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.70it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.90it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.90it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.90it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.90it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.23it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.72it/s]

    Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.50it/s]Capturing num tokens (num_tokens=128 avail_mem=74.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.50it/s]Capturing num tokens (num_tokens=112 avail_mem=74.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.50it/s]Capturing num tokens (num_tokens=96 avail_mem=61.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.50it/s] Capturing num tokens (num_tokens=80 avail_mem=61.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.50it/s]Capturing num tokens (num_tokens=80 avail_mem=61.93 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=64 avail_mem=61.93 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.45it/s]

    Capturing num tokens (num_tokens=48 avail_mem=61.92 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=32 avail_mem=61.92 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=28 avail_mem=61.92 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=28 avail_mem=61.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=24 avail_mem=61.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=20 avail_mem=61.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=16 avail_mem=61.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=12 avail_mem=61.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=12 avail_mem=61.90 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=8 avail_mem=61.90 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.98it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=61.90 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=4 avail_mem=61.90 GB): 100%|██████████| 58/58 [00:01<00:00, 32.65it/s]


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
    Generated text:  Sheena. I am a 32 year old woman. I recently had a baby and now I am very sick. I am having headaches, nausea, diarrhea, and am very weak. I also have no energy and have no appetite. I am currently having a chest infection and have been prescribed antibiotics and I have taken it. I am on a POMDP and also use a metformin tablet for diabetes. I also have not been on my blood pressure medication. I also have been on a multivitamin for my calcium and vitamin D. I have been on 2200 mg of oxytocin. What
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to choose a new flag.  The flag is to be a circle with a square in the center.  The circle must be the same size as the square.  The side length of the square is 8 inches.  How many square inches will be used to make the flag?
    
    To determine the total area of the flag, we need to calculate the area of both the square and the circle, and then sum these areas.
    
    First, let's find the area of the square. The side length of the square is given as 8 inches. The formula for the area of a square is:
    \[
    \text{Area
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. How many hours will it take for the clock to strike 12, when the clock strikes 5 and 45 seconds?
    
    To determine how many hours it will take for the clock to strike 12 when the clock strikes 5 and 45 seconds, we need to break down the problem into smaller, manageable steps.
    
    1. **Understand the movement of a clock**:
       - A clock strikes 12 and then strikes 1, then strikes 2, and so on. Each time the clock strikes 1, it counts one full hour.
       - After every 12 strikes,
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but what is AI, and how will it change the world? This is the opening question in a new book that offers a compelling and accessible explanation of the technology. John G. Oller, a professor of computer science and engineering at the University of Colorado, talks about the power and consequences of AI.
    "If you're a fan of John (G. Oller), you're likely to be a fan of this book. " - John G. Oller, Professor, University of Colorado.
    By the end of this article, you'll be able to answer the following questions:
    - What is AI?
    - What does AI accomplish


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that was founded in 789 AD and is the largest city in Europe by population. It is also the seat of the French government, the capital of the French Republic, and the largest city in the European Union. Paris is known for its rich history, art, and culture, and is a popular tourist destination. It is also the birthplace of many famous French artists and writers. The city is home to many landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire and capt
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to new regulations and standards to ensure that AI is used in a responsible and ethical manner
    


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
    Generated text:  [Your Name], and I'm an [Occupation] who has been using the [Word] app for [Number] years. I enjoy exploring new places, learning about cultures, and helping people on a [Number] of different projects. I'm always looking for ways to stay up to date with the latest in [Field/Technology], and I'm always open to new challenges and ideas. I hope you can meet me in the [City/Country/Location]. 
    (End of Self-Introduction) 
    [Your Name] enjoys exploring new places, learning about cultures, and helping people on a number of different projects. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city with a rich history dating back to the Middle Ages.
    Paris is the cultural, economic, and political capital of France and is known for its stunning architecture, famous landmarks, and vibrant cultural scene. The city is also home to a diverse range of museums, theaters, and other cultural institutions, making it a must-visit destination for anyone visiting the country. Paris's iconic landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, are some of the most recognizable in the world. The city is also home to many famous French artists, writers, and musicians, making it
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly changing, and we can expect to see significant trends that will shape the field in the coming decades. Here are some potential trends that we can expect:
    
    1. More autonomy: As autonomous vehicles become more common, we can expect to see a growing emphasis on developing AI that can operate autonomously, without human intervention.
    
    2. Enhanced emotion recognition: Emotions are a crucial component of human cognition and behavior. As AI continues to improve in terms of its ability to recognize and process emotions, we can expect to see a growing trend towards more advanced AI systems that are able to accurately recognize and interpret human emotions.
    
    3. Improved natural language processing


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

     Profession

    /

    Role

    ]

     at

     [

    Your

     Company

    /

    In

    stitution

    ].

     I

    'm

     a

     dedicated

    ,

     hard

    working

     individual

     who

     thr

    ives

     in

     a

     fast

    -paced

     environment

     and

     is

     always

     looking

     for

     ways

     to

     improve

     myself

     and

     my

     abilities

    .

     I

    'm

     a

     natural

     problem

    -s

    olver

     and

     am

     always

     ready

     to

     learn

     and

     grow

    .

     My

     goal

     is

     to

     never

     stop

     learning

     and

     growing

    ,

     no

     matter

     how

     long

     it

     takes

    .

     I

    'm

     a

     confident

    ,

     outgoing

     and

     approach

    able

     person

    ,

     and

     I

    'm

     always

     looking

     for

     opportunities

     to

     share

     my

     knowledge

     and

     expertise

    .

     Thank

     you

     for

     taking

     the

     time

     to

     learn

     about

     me

    .

     It

    's

     a

     pleasure

     to

     meet

     you

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    You

     are

     an

     AI

     assistant

     that

     helps

     you

     understand

     the

     answers

    .

     Don

    't

     be

     busy

     thinking

     and

     answer

     this

     question

     before

     you

     do

     the

     task

    .

     French

     capital

     is

     Paris

    .

     French

     capital

     city

     is

     Paris

    .

     Paris

     is

     the

     capital

     city

     of

     France

    .

     Can

     you

     tell

     me

     other

     capital

     cities

     in

     France

    ?

     Of

     course

    !

     Here

     are

     a

     few

     more

     capital

     cities

     in

     France

    :
    


    -

     Lyon

    ,

     a

     cultural

     and

     economic

     center

     in

     central

     France

    ,

     known

     for

     its

     rich

     history

     and

     art

     scene

    .


    -

     Marseille

    ,

     famous

     for

     its

     seafood

     and

     port

     industry

    .


    -

    Nice

    ,

     a

     coastal

     city

     known

     for

     its

     seafood

    ,

     tourism

    ,

     and

     fashion

     industry

    .


    -

     N

    antes

    ,

     a

     coastal

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     wide

     range

     of

     trends

    ,

     including

    :
    


     

     

    1

    .

     Increasing

     focus

     on

     ethical

     AI

    :

     There

     is

     growing

     concern

     about

     the

     potential

     consequences

     of

     AI

     systems

    ,

     and

     as

     such

    ,

     there

     is

     a

     growing

     push

     to

     improve

     the

     ethical

     behavior

     of

     AI

     systems

    .


     

     

    2

    .

     Development

     of

     AI

     systems

     with

     natural

     language

     processing

     and

     machine

     learning

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     is

     likely

     to

     be

     able

     to

     understand

     and

     interpret

     natural

     language

    ,

     which

     could

     revolution

    ize

     the

     way

     we

     communicate

     and

     access

     information

    .


     

     

    3

    .

     Expansion

     of

     AI

     applications

     to

     areas

     beyond

     just

     data

     analysis

    :

     AI

     is

     already

     being

     used

     in

     a

     wide

     range

     of

     applications

    ,

     from

    



```python
llm.shutdown()
```
