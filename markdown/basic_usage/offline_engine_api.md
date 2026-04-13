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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.44it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.44it/s]


    2026-04-13 05:52:21,810 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 05:52:21] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:25,  2.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:25,  2.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:25,  2.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:25,  2.54s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.52it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 20.93it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 20.93it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 20.93it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 20.93it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.93it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.93it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.93it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.93it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.93it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.93it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.93it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.93it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.93it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.93it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.93it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.93it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.93it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 35.57it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 35.57it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 35.57it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 35.57it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 35.57it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 35.57it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 35.57it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 35.57it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 41.79it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.58 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.58 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.58 GB):   7%|▋         | 4/58 [00:00<00:03, 17.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.58 GB):   7%|▋         | 4/58 [00:00<00:03, 17.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.58 GB):   7%|▋         | 4/58 [00:00<00:03, 17.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   7%|▋         | 4/58 [00:00<00:03, 17.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.01it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.26it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.26it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.76it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.76it/s] Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=832 avail_mem=72.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=640 avail_mem=72.51 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=640 avail_mem=72.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=576 avail_mem=72.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=240 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.62it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.33it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.33it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.33it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.33it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.33it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.33it/s]Capturing num tokens (num_tokens=32 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.33it/s]Capturing num tokens (num_tokens=32 avail_mem=72.46 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.16it/s]Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.16it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.16it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.16it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.16it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.16it/s]Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.16it/s] Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  98%|█████████▊| 57/58 [00:01<00:00, 50.12it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  98%|█████████▊| 57/58 [00:01<00:00, 50.12it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 41.87it/s]


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
    Generated text:  Raquel and I am from Bolivia. I come from a family that has always been involved in sports. I’ve always loved the power of sports and I would like to learn as many new skills as possible. I also love to travel and try new foods. I also have a passion for music and I would like to learn to play the guitar and sing in my free time.
    I’m a student and I’m currently learning to play guitar in a private guitar program. I am a fan of all kinds of music. I also love to listen to music and take photos of my favorite songs, which makes me so happy! I am currently
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to executive a new law that would give the president a veto power. He wants to know how many valid outcomes he has, given that there are three possible states of nature: 1) a normal state where the president would follow the existing law, 2) a bad state where the president would not follow the law, and 3) an extremely bad state where the president would not follow the law, and the outcomes of the states of nature are independent.
    
    The president wants to know how many possible different outcomes there are if he decides to veto the law under each state of nature. Calculate the total number of possible
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. You can see the Paris metro station “Le Marais” in the picture. 
    
    <image>
    
    The Paris metro system consists of several lines that are parallel to each other. The lines are numbered from 1 to 5 in the order of the lines opening. The metro stations are numbered from 1 to 328. Each station has a terminal, which is the last station of the line connected to the station.
    
    The stations in the same line have the same number of stations. The stations in different lines are numbered in an increasing order of numbers. The terminal of the station 1 of line 
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting and diverse, with applications ranging from robotic warfare to the development of personal assistants like Siri and Alexa. While the future is bright, it’s also filled with potential risks. Here’s a look at some of the biggest issues that need to be addressed before AI becomes a reality:
    
    1. Bias and Discrimination: One of the biggest risks of AI is the potential for bias and discrimination. If AI is trained on biased data, it can perpetuate or even exacerbate existing societal biases. This could lead to unintended consequences and discrimination in everyday life.
    
    2. Privacy and Security: As AI becomes more integrated into our daily lives, it raises


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is a city of contrasts, with its elegant architecture, vibrant nightlife, and diverse cultural scene. It is a city of art, science, and philosophy, and a city of innovation and progress
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends in AI that are expected to shape the future:
    
    1. Increased automation: As AI continues to become more advanced, it is likely to automate many of the tasks that are currently done by humans. This could lead to a significant increase in productivity and efficiency, but it could also lead to job displacement for some people.
    
    2. Improved privacy and security: As AI becomes more advanced, it is likely to require more data to function effectively. This could lead to increased privacy and
    


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
    Generated text:  [Your Name], and I’m a [character's profession or title] here at [Company Name]. My background is [summary of your professional experience and achievements], and I’ve been working at [Company Name] for [number] years. I’m passionate about [major area of interest or expertise], and I’m always looking for ways to [positive action or goal]. I thrive on [reason why you excel as a team player or problem-solver], and I’m eager to help others do the same. How would you like to get to know more about me? Feel free to ask me any questions you may have, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and the largest city in both France and Europe. It is known for its historical landmarks, vibrant culture, and annual celebrations such as the Eiffel Tower and the French New Year. It is also known for its fashion industry, food, and cuisine. Paris is a popular tourist destination, and its landmarks and cultural attractions attract millions of visitors each year. The city's population is also growing rapidly, and it is considered the most populous city in Europe. Paris is also known for its role in the French Revolution and as a symbol of the nation.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and unpredictable, but there are several trends that are likely to shape its direction:
    
    1. Increased personalization: With the advent of AI, we can expect to see a significant increase in personalization of AI-based services, tailored to individual needs and preferences. This could lead to more efficient use of resources and improved customer satisfaction.
    
    2. Improved transparency and accountability: As AI systems become more complex and sophisticated, it is likely that we will see a greater emphasis on transparency and accountability. This could lead to increased scrutiny of AI-powered systems and greater public trust in the technology.
    
    3. Increased reliance on AI for critical tasks: AI will increasingly


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

    /

    Title

    ]

     with

     [

    Your

     University

    /

    Professional

     Experience

    ]

     under

     my

     belt

    .

     I

    've

     always

     been

     passionate

     about

     [

    Your

     Passion

    /F

    avorite

     Thing

    ].

     I

    'm

     excited

     to

     take

     on

     this

     role

     and

     help

     [

    Your

     Client

    /

    Company

    ]

     achieve

     [

    Their

     Goal

    ].

     I

    'm

     always

     open

     to

     learning

     and

     improving

     my

     skills

    ,

     and

     I

    'm

     confident

     that

     I

     can

     bring

     a

     new

     and

     unique

     perspective

     to

     this

     position

    .

     What

    's

     your

     name

    ?

     And

     what

    's

     your

     profession

    /

    position

    ?


    Hello

    ,

     my

     name

     is

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

    /

    Title

    ]

     with

     [

    Your

     University

    /

    Professional

     Experience

    ]

     under

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     vibrant

     and

     historic

     city

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

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .


    Paris

    ,

     the

     capital

     of

     France

    ,

     is

     renowned

     for

     its

     iconic

     landmarks

     like

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     history

    ,

     including

     the

     Renaissance

    ,

     French

     Revolution

    ,

     and

     modern

     history

    ,

     which

     have

     shaped

     its

     culture

    ,

     art

    ,

     and

     traditions

    .

     Its

     diverse

     neighborhoods

    ,

     including

     the

     Left

     Bank

    ,

     Right

     Bank

    ,

     and

     the

     B

    ist

    rot

    ,

     offer

     a

     wide

     array

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

     that

     are

     shaping

     the

     technology

     and

     applications

     of

     artificial

     intelligence

     (

    AI

    )

     over

     the

     next

     few

     years

    .

     Some

     of

     the

     most

     significant

     trends

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     The

     use

     of

     AI

     is

     expected

     to

     continue

     to

     grow

     as

     more

     companies

     and

     governments

     seek

     to

     address

     ethical

     concerns

     related

     to

     AI

    ,

     such

     as

     bias

    ,

     transparency

    ,

     and

     accountability

    .

     AI

     researchers

     are

     also

     focusing

     on

     developing

     more

     ethical

     AI

     systems

     that

     can

     make

     decisions

     that

     are

     fair

     and

     respectful

     to

     all

     parties

     involved

    .
    


    2

    .

     Increase

     in

     specialized

     AI

    :

     AI

     systems

     are

     likely

     to

     become

     more

     specialized

     as

     they

     are

     used

     for

     specific

     tasks

    ,

     such

     as

     image

     recognition

    



```python
llm.shutdown()
```
