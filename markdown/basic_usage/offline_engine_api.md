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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.07it/s]


    2026-04-12 13:08:20,655 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 13:08:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.89it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.24it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.24it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.24it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.24it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.24it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.24it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.24it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.24it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.24it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.11it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.11it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.11it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.11it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.11it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.11it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.11it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.11it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.56it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.56it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.56it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.56it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.56it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.56it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.56it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.56it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.37 GB):   3%|▎         | 2/58 [00:00<00:03, 16.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.37 GB):   3%|▎         | 2/58 [00:00<00:03, 16.50it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.37 GB):   3%|▎         | 2/58 [00:00<00:03, 16.50it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.37 GB):   3%|▎         | 2/58 [00:00<00:03, 16.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.37 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.36 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.36 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.36 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.36 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=60.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.32 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.30 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.66it/s]

    Capturing num tokens (num_tokens=960 avail_mem=60.32 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.66it/s] Capturing num tokens (num_tokens=896 avail_mem=60.31 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=832 avail_mem=60.31 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=768 avail_mem=60.31 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=704 avail_mem=60.30 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=704 avail_mem=60.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=640 avail_mem=60.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=576 avail_mem=60.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=512 avail_mem=60.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=480 avail_mem=60.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=448 avail_mem=60.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.83it/s]

    Capturing num tokens (num_tokens=448 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=416 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=384 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=352 avail_mem=60.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=320 avail_mem=60.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.62it/s]

    Capturing num tokens (num_tokens=288 avail_mem=60.28 GB):  53%|█████▎    | 31/58 [00:01<00:00, 44.62it/s]Capturing num tokens (num_tokens=288 avail_mem=60.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=256 avail_mem=60.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=240 avail_mem=60.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=224 avail_mem=60.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=208 avail_mem=60.27 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=192 avail_mem=60.27 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=176 avail_mem=60.27 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=176 avail_mem=60.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.44it/s]Capturing num tokens (num_tokens=160 avail_mem=60.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.44it/s]Capturing num tokens (num_tokens=144 avail_mem=60.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.44it/s]Capturing num tokens (num_tokens=128 avail_mem=60.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.44it/s]

    Capturing num tokens (num_tokens=112 avail_mem=60.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.44it/s]Capturing num tokens (num_tokens=96 avail_mem=60.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.44it/s] Capturing num tokens (num_tokens=80 avail_mem=60.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.44it/s]Capturing num tokens (num_tokens=80 avail_mem=60.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=64 avail_mem=60.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=48 avail_mem=60.24 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=32 avail_mem=60.24 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=28 avail_mem=60.23 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=24 avail_mem=60.23 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=20 avail_mem=60.23 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=20 avail_mem=60.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.14it/s]Capturing num tokens (num_tokens=16 avail_mem=60.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.14it/s]Capturing num tokens (num_tokens=12 avail_mem=60.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.14it/s]

    Capturing num tokens (num_tokens=8 avail_mem=60.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.14it/s] Capturing num tokens (num_tokens=4 avail_mem=60.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.14it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:01<00:00, 39.04it/s]


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
    Generated text:  Raquel and I am a student at the University of the South Pacific. I have learned so much during this 100-day course and I'm happy to share my experiences with you. I am also very interested in knowing what other students have done in this course and what are they doing after completing the course. To do this, I want to ask you a series of questions. Please answer them to the best of your ability.
    
    **Question:** What was your favorite day to spend during your stay in the United States?
    
    **A:** My favorite day was on my last day because I had a great time with friends and family.
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. He has to travel the country to make sure that everything is good. He can't go to bed until after dinner time. His father has a secret place where he can keep his messages. It is a special place. He gives him permission to use it. He can read messages in it and when the president asks him to write a letter, he writes it down. He takes his secret place out to be put away. He says he will read messages from time to time when he is not at home. But he doesn't read it very often. The president likes it. His father is busy, but he hopes he
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Strasbourg
    C. Lyon
    D. Bercy
    
    To determine the capital of France, let's analyze each option step by step:
    
    A. Paris - Paris is the capital city of France and has been for centuries. It is well-known for its rich history, renowned art museums, and beautiful architecture.
    
    B. Strasbourg - Strasbourg is a city in northeastern France, known for its medieval architecture and historical importance. It is not the capital of France.
    
    C. Lyon - Lyon is a city in southwestern France and is famous for its wine industry and fortress architecture. It is not
    ===============================
    Prompt: The future of AI is
    Generated text:  looking a lot more like the future of the world as a whole. This is partly because of the incredible advances that have taken place in the past few years, as well as the ongoing evolution of the field itself.
    In fact, it may be more accurate to say that the world of AI is changing faster than the rest of the world. This is due to a number of factors, including the increasing use of artificial intelligence in the healthcare industry, the expansion of the use of AI in the financial industry, and the integration of AI in various sectors such as manufacturing, transportation, and retail.
    However, it is also important to recognize that the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is also the oldest capital city in Europe, having been founded in 789 AD. Paris is known for its rich history, art, and cuisine. It is also a major financial center and a major tourist destination. The city is home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a cultural and intellectual center, with many important museums, theaters, and art galleries. It is also a major hub for business and commerce, with many international companies and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced machine learning capabilities: AI will continue to improve its ability to learn from data and make more accurate predictions and decisions.
    
    3. Increased use of AI in healthcare: AI will be used to improve the accuracy and efficiency of medical diagnosis and treatment, as well as to develop new treatments and therapies.
    
    4. Greater integration with natural language processing: AI will become more integrated with natural language processing, allowing
    


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
    Generated text:  Sarah and I'm a stay-at-home mom with a degree in psychology. I've always been an introverted person, so my passions are usually outside the home. I enjoy spending time in nature, reading, and cooking delicious meals with my husband. How would you describe your personality type? Persona:
    50%
    Introvert
    10%
    Loyal
    30%
    Caring
    20%
    Assistent
    10%
    Question: Can you tell me a bit about yourself and your life outside of work? Sarah enjoys spending time with her family, volunteering at a local animal shelter, and trying new recipes. How do you balance
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an historic city that is the third largest in the country and one of the most populous.
    Paris, also known as "La Chapelle-Roeuil" and "Lyon de la Rochefoucauld," is the third largest city in France by population and the seventh largest by area, with a population of approximately 2.1 million people. It is the cultural and political capital of France, home to the Louvre Museum and many other attractions. Paris is also known for its vibrant nightlife and culinary scene, which attracts tourists and locals alike. The city is a melting pot of cultures and traditions, with a rich history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and challenges. Here are some possible trends that we can expect to see in the coming years:
    
    1. Increased automation and self-driving: The future of AI will see a greater focus on increasing automation and self-driving technology. This will lead to a rise in jobs, but it will also create new opportunities for those who are skilled in programming and AI.
    
    2. Enhanced cognitive abilities: AI will continue to advance and become more advanced. We may see breakthroughs in areas like language translation, healthcare, and financial modeling.
    
    3. Integration of AI into everyday life: AI will continue to become more integrated into everyday life, from


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

    ].

     I

    'm

     a

     [

    occupation

     or

     profession

    ]

     with

     a

     passion

     for

     [

    what

     you

     enjoy

     doing

    ].

     I

    'm

     someone

     who

     is

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     develop

     new

     ones

    .

     I

     am

     not

     afraid

     to

     take

     risks

     and

     push

     boundaries

    ,

     and

     I

     believe

     that

     the

     best

     way

     to

     grow

     and

     succeed

     is

     to

     embrace

     failure

     as

     a

     stepping

     stone

     to

     success

    .

     
    


    What

     inspired

     you

     to

     become

     a

     [

    occupation

     or

     profession

    ]

    ?
    


    I

     was

     inspired

     by

     [

    reason

     why

     you

     started

     this

     path

     of

     action

    ].

     And

     I

     believe

     that

     my

     journey

     is

     unique

     because

     it

     involves

     embracing

     [

    something

     new

     or

     difficult

    ]

     to

     build

     upon

     my

     existing

     skills

     and

     knowledge

    .
    


    What

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    A

    ust

    rian

     composer

     Franz

     Lis

    zt

     is

     widely

     regarded

     as

     the

     greatest

     virt

    u

    oso

     pian

    ist

     of

     the

     

    1

    9

    th

     century

    .

     What

     does

     that

     statement

     mean

    ?

     Explain

     using

     at

     least

     three

     details

     about

     Lis

    zt

    .
    


    A

    ust

    rian

     composer

     Franz

     Lis

    zt

     is

     widely

     regarded

     as

     the

     greatest

     virt

    u

    oso

     pian

    ist

     of

     the

     

    1

    9

    th

     century

    .

     Lis

    zt

     was

     born

     in

     

    1

    8

    1

    1

     in

     Moh

    r

    stadt

    ,

     Sax

    ony

     and

     was

     considered

     to

     be

     the

     greatest

     pian

    ist

     of

     the

     

    1

    9

    th

     century

     by

     the

     young

     Vi

    enn

    ese

     pian

    ist

     Franz

     Lis

    zt

    ,

     who

     he

     became

     his

     teacher

     and

     mentor

    .

     At

     age

     

    1

    9

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     potential

     trends

     that

     are

     both

     exciting

     and

     complex

    .

     Some

     of

     the

     most

     significant

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     Aut

    onomy

    :

     AI

     will

     continue

     to

     get

     more

     autonomous

    ,

     with

     machines

     able

     to

     think

     and

     make

     decisions

     without

     human

     intervention

    .

     This

     could

     lead

     to

     more

     efficient

    ,

     scalable

    ,

     and

     responsive

     AI

     systems

     that

     can

     adapt

     and

     learn

     on

     their

     own

    .
    


    2

    .

     Enhanced

     Personal

    ization

    :

     With

     AI

    ,

     people

     will

     be

     able

     to

     receive

     more

     personalized

     experiences

    ,

     from

     recommendations

     to

     personalized

     insurance

     and

     healthcare

     products

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

    ,

     as

     well

     as

     greater

     customer

     satisfaction

    .
    


    3

    .

     Development

     of

     Universal

     AI

    :

     AI

     systems

     will

    



```python
llm.shutdown()
```
