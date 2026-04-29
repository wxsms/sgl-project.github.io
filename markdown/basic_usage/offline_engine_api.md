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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.61it/s]


    2026-04-29 13:21:55,781 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 13:21:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.62it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.32it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 38.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.66it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.66it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.75it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=116.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.37 GB):  21%|██        | 12/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.36 GB):  21%|██        | 12/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.36 GB):  21%|██        | 12/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.36 GB):  21%|██        | 12/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.35 GB):  21%|██        | 12/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.35 GB):  21%|██        | 12/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.74it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=116.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=960 avail_mem=116.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.74it/s] Capturing num tokens (num_tokens=960 avail_mem=116.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=896 avail_mem=116.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=832 avail_mem=116.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=768 avail_mem=116.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=704 avail_mem=116.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=640 avail_mem=116.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=640 avail_mem=116.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.30it/s]Capturing num tokens (num_tokens=576 avail_mem=116.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.30it/s]

    Capturing num tokens (num_tokens=512 avail_mem=116.30 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.30it/s]Capturing num tokens (num_tokens=480 avail_mem=116.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.30it/s]Capturing num tokens (num_tokens=448 avail_mem=116.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.30it/s]Capturing num tokens (num_tokens=416 avail_mem=116.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.30it/s]Capturing num tokens (num_tokens=416 avail_mem=116.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.17it/s]Capturing num tokens (num_tokens=384 avail_mem=116.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.17it/s]Capturing num tokens (num_tokens=352 avail_mem=116.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.17it/s]Capturing num tokens (num_tokens=320 avail_mem=116.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.17it/s]Capturing num tokens (num_tokens=288 avail_mem=116.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.17it/s]Capturing num tokens (num_tokens=256 avail_mem=116.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.17it/s]

    Capturing num tokens (num_tokens=256 avail_mem=116.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=240 avail_mem=116.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=224 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=208 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=192 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=176 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=176 avail_mem=116.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=160 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=144 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=128 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=112 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.75it/s]

    Capturing num tokens (num_tokens=96 avail_mem=116.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.75it/s] Capturing num tokens (num_tokens=96 avail_mem=116.23 GB):  81%|████████  | 47/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=80 avail_mem=116.23 GB):  81%|████████  | 47/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=64 avail_mem=116.22 GB):  81%|████████  | 47/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=48 avail_mem=116.22 GB):  81%|████████  | 47/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=32 avail_mem=116.22 GB):  81%|████████  | 47/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=28 avail_mem=116.21 GB):  81%|████████  | 47/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=28 avail_mem=116.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=24 avail_mem=116.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=20 avail_mem=116.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=16 avail_mem=116.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.01it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=8 avail_mem=116.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.01it/s] Capturing num tokens (num_tokens=8 avail_mem=116.20 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=4 avail_mem=116.20 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=4 avail_mem=116.20 GB): 100%|██████████| 58/58 [00:01<00:00, 38.07it/s]


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
    Generated text:  Lina and I'm from Mexico. I'm an astronaut. I live in China. I'm an engineer. I'm a student. When I first met you, I thought you were very cool. At first, I had a hard time making friends. But you always said, "Just relax and get to know me." So I tried my best to make you laugh. You were kind to me, and that's why I've always liked you. It's true that in my country, we always have a good relationship with each other, but you are an engineer. You seem to work a lot. We usually spend a lot
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she is the leader of the country. The president is not elected by the people. He or she is elected by the voters. The president is president of the United States. He or she is the leader of the country. The president is president of the United States. The president is president of the United States. The president is president of the United States. The president is president of the United States. The president is president of the United States. The president is president of the United States. The president is president of the United States.
    What are some of the important roles and responsibilities of
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) Paris
    B) Lyon
    C) Bordeaux
    D) Marseille
    Answer:
    A
    
    Among the following options, which one is a derivative of the product rule?
    A. (ax+b)^2
    B. (a+b)^2
    C. (2x+3)^3
    D. (a+x)^2
    Answer:
    B
    
    On a north-south street, there are two friends: Person A, who lives between A and B, and Person B, who lives between B and C. Person A and Person B have the same income level, but Person A lives in the north of the street
    ===============================
    Prompt: The future of AI is
    Generated text:  strong and growing, and there are many innovative solutions available to address its challenges. While AI has brought many benefits, it can also pose some risks and ethical concerns. While it is important to embrace AI in order to drive innovation and progress, it is also important to be mindful of its potential impact on human society and the environment.
    AI can be used to improve decision-making in a wide range of fields, from healthcare to finance to transportation. However, there are also risks associated with AI, such as the possibility of bias in AI algorithms and the potential for unintended consequences of AI deployment.
    To address these risks, it is important to ensure that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character or personality]. And what's your background? I have a [insert a short description of your background or education]. And what's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity]. And what's your favorite book or movie? I love [insert a short description of your favorite book or movie]. And what's your favorite place to go? I love [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance in French culture and politics. It is a widely recognized and well-known fact that Paris is the capital city of France, and this statement accurately reflects this fact. The statement is simple and easy to understand, making it suitable for a wide range of audiences. It also avoids any potential confusion or ambiguity, as it clearly distinguishes between the capital city and its neighboring cities. Overall, this statement is a reliable and accurate representation of the facts surrounding Paris's role as the capital of France. 
    
    In conclusion, the statement "
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI that can better understand and respond to human needs and emotions.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and responsible AI development. This could involve creating AI that is designed to be transparent
    


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
    Generated text:  __________. I'm a(n) ____________. I love to ___________ and enjoy __________. I'm an excellent _____________. I'm a(n) _____________. I love to ___________ and enjoy ___________. I'm an excellent _____________. I'm a(n) ____________. I love to ___________ and enjoy ___________. I'm an excellent _____________. I'm a(n) ____________. I love to ___________ and enjoy ___________. I'm an excellent _____________. I'm a(n) ____________. I love to ___________ and enjoy
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is a historical and cultural center with a rich history dating back to the time of the Roman Empire, and is the second-most populous city in the European Union. Paris is known for its art, architecture, and cuisine, and is also home to numerous famous landmarks, museums, and theaters. It is a UNESCO World Heritage site and a major international hub for politics, business, and media. Paris is a vibrant and cosmopolitan city with a vibrant cultural scene and a lively nightlife. Despite its historical and cultural importance, Paris is also known for its rapid pace of life and high cost of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's evolution. Here are some possible future trends in AI:
    
    1. Increased integration with human intelligence: As AI becomes more complex, it's likely that it will be integrated with human intelligence in a way that enhances and complements it. This could mean that AI will be able to understand and interpret human emotions, cognitive functions, and decision-making processes.
    
    2. Advancements in machine learning: Machine learning is a key component of AI, and it's expected to continue to advance in the coming years. Advances in machine learning will likely lead to more sophisticated algorithms that


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

     first

     name

    ]

     and

     I

    'm

     a

     [

    insert

     profession

     or

     hobby

    ]

     who

     loves

     to

     [

    insert

     a

     personal

     trait

     or

     passion

    ]

     and

     [

    insert

     something

     related

     to

     your

     interests

     or

     hobbies

    ].

     I

    'm

     always

     looking

     for

     ways

     to

     [

    insert

     something

     related

     to

     my

     interests

     or

     hobbies

    ],

     so

     I

    'm

     always

     up

     for

     learning

     and

     growing

    .

     I

    'm

     currently

     [

    insert

     age

     and

     background

    ]

     years

     old

     and

     I

     live

     in

     [

    insert

     your

     city

     or

     country

    ].

     If

     you

    're

     interested

    ,

     I

     can

     have

     a

     chat

     about

     my

     interests

     and

     experiences

    .

     What

    's

     your

     name

    ?

     [

    insert

     your

     name

    ]

     How

     is

     it

     going

    ?

     [

    insert

     your

     name

    ]

     How

     are

     you

    ?

     [

    insert

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     and

     it

     is

     known

     for

     its

     rich

     history

    ,

     vibrant

     culture

    ,

     and

     beautiful

     architecture

    .

     The

     city

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

     the

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     known

     for

     its

     fashion

    ,

     cuisine

    ,

     and

     wine

    ,

     which

     are

     famous

     throughout

     the

     world

    .

     Paris

     is

     a

     city

     that

     has

     played

     an

     important

     role

     in

     French

     history

    ,

     and

     it

     continues

     to

     be

     a

     major

     cultural

     hub

     for

     many

     years

     to

     come

    .

     Paris

     is

     a

     truly

     unique

     and

     fascinating

     city

    ,

     and

     it

     is

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     France

    .

     
    


    (

    1

    5

    0

     words

    )

     Paris

     is

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     quite

     promising

    ,

     and

     it

     is

     likely

     to

     continue

     to

     evolve

     and

     transform

     in

     exciting

     ways

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     automation

     and

     AI

     in

     various

     industries

    :

     With

     the

     advent

     of

     AI

    ,

     automation

     in

     various

     industries

     is

     expected

     to

     continue

     to

     increase

    .

     This

     includes

     manufacturing

    ,

     healthcare

    ,

     finance

    ,

     and

     transportation

    .

     AI

    -powered

     automation

     will

     likely

     become

     more

     prevalent

     as

     it

     can

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

    ,

     such

     as

     data

     analysis

    ,

     pattern

     recognition

    ,

     and

     decision

    -making

    .
    


    2

    .

     AI

    -powered

     autonomous

     vehicles

    :

     AI

    -powered

     autonomous

     vehicles

     (

    AV

    )

     are

     expected

     to

     become

     more

     prevalent

     in

     the

     future

    ,

     particularly

    



```python
llm.shutdown()
```
