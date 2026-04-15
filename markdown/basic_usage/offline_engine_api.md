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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]


    2026-04-15 20:54:24,507 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 20:54:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:22,  2.49s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:22,  2.49s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:22,  2.49s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:22,  2.49s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  2.00it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  2.00it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  2.00it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  2.00it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  2.00it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  2.00it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:27,  2.00it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.71it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 21.10it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 21.10it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 21.10it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 21.10it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:02<00:01, 21.10it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:02<00:01, 21.10it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:02<00:01, 21.10it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 21.10it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 21.10it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.97it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.97it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.97it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.97it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.97it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.97it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.97it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.97it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 35.49it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 35.49it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 35.49it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 35.49it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 35.49it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 35.49it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 35.49it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 35.49it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 41.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.96it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.28it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.28it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.32it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.32it/s] Capturing num tokens (num_tokens=896 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.32it/s]Capturing num tokens (num_tokens=896 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:01, 24.90it/s]Capturing num tokens (num_tokens=832 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:00<00:01, 24.90it/s]Capturing num tokens (num_tokens=768 avail_mem=76.68 GB):  40%|███▉      | 23/58 [00:00<00:01, 24.90it/s]

    Capturing num tokens (num_tokens=704 avail_mem=76.68 GB):  40%|███▉      | 23/58 [00:00<00:01, 24.90it/s]Capturing num tokens (num_tokens=640 avail_mem=76.66 GB):  40%|███▉      | 23/58 [00:00<00:01, 24.90it/s]Capturing num tokens (num_tokens=640 avail_mem=76.66 GB):  47%|████▋     | 27/58 [00:00<00:01, 27.91it/s]Capturing num tokens (num_tokens=576 avail_mem=76.18 GB):  47%|████▋     | 27/58 [00:00<00:01, 27.91it/s]Capturing num tokens (num_tokens=512 avail_mem=76.17 GB):  47%|████▋     | 27/58 [00:00<00:01, 27.91it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.91it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.91it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.91it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.91it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.38it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.38it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.38it/s]

    Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.38it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.38it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.38it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.38it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=208 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=192 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.07it/s]

    Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.07it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=48 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.33it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.60it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.60it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 37.29it/s]


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
    Generated text:  Rory and I’m 18 years old. My older sister is Jessica, she’s 19 years old. My parents are Patrick and Mary, they have three kids. I was born on June 12th, 2009, and my dad and I met at the age of 18. I’m very good with numbers and I enjoy playing sports, but I don’t do a lot of them. I was really good at running until I started playing soccer when I started to lose. It was hard for me because I was used to being a fast runner and I just couldn’t run as fast as
    ===============================
    Prompt: The president of the United States is
    Generated text:  attending a science convention. He has been asked to speak at a science convention that will be held on Wednesday 5th of June, 2023. According to his schedule, he has 48 minutes to speak. 
    
    The president wants to ensure that he spends the maximum number of minutes speaking. He decides to speak for half of the time on the first day and for the other half on the second day. 
    
    How many minutes does he speak on each day, and what is the total number of minutes he will speak during the entire day? To determine how many minutes the president will speak on each day, we start
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. New York
    D. Rome
    Answer: A
    
    The clinical characteristics of peptic ulcer disease include
    A. Pain may occur 2-3 days before the onset of symptoms
    B. In the early stages, it is often asymptomatic
    C. The disease is more common in middle-aged and elderly people
    D. The upper digestive tract mucosa is often in an alkaline environment
    E. The upper digestive tract mucosa is often in a neutral environment
    Answer: B
    
    The secondary injury of food poisoning is ____.
    A. Primarily caused by food contamination
    ===============================
    Prompt: The future of AI is
    Generated text:  not about replacing humans, but about integrating them with each other.
    
    Can you repeat this sentence, but capitalize it correctly? The future of AI is not about replacing humans, but about integrating them with each other.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? I'm a [insert a short description of your personality or skills]. And what kind of experiences have you had that have shaped your career? I'm looking forward to hearing about your background and how it has prepared you for your current role. What's your favorite part of your job? I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. And what kind of challenges do you face in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. The city is known for its cuisine, fashion, and art, and it is a major hub for business and commerce in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also home to many international organizations and institutions, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This includes issues such as bias, transparency, and accountability. AI developers will need to be more mindful of how their technology is used and how it affects society.
    
    2. Integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to new forms of AI that are more capable of
    


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
    Generated text:  [Name], and I'm a [ career field]. [ Name] is a [summary of career field]. I'm passionate about [why you chose this career field]. I'm a [greatest strengths or skills] who enjoys [job challenges or personal growth] every day. I love connecting with people and [mention something that makes you relatable or a unique trait]. I'm a [career type, e. g. Digital Marketing Specialist, Accountant, etc.]. I am an [ultimate goal] for my career. [ Name ] is [role].
    **[Name]** is a [career field], a [summary of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the world’s most populous and the second-largest city by area after New York City. It is the oldest capital city in Europe, the seat of government, the cultural and economic capital, and the largest city in terms of population and population density. Paris is known for its ancient Roman and medieval heritage, including the Eiffel Tower, Notre-Dame Cathedral, the Louvre Museum, and the Champs-Élysées. The city is home to many of France’s most popular tourist attractions, including the Eiffel Tower, the Louvre Museum, and the iconic Arc de Triomphe. Paris is a vibrant and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, and it is expected to continue to develop rapidly, transforming our lives and businesses in numerous ways. Here are some possible future trends in artificial intelligence:
    
    1. Increased autonomy: As AI becomes more advanced, the ability of machines to make decisions and take action will become more nuanced and autonomous. This will be particularly important in industries where human workers are needed, such as manufacturing and healthcare.
    
    2. Personalization: AI will be used to improve the user experience by providing personalized recommendations and suggestions based on the user's behavior and preferences. This will enable users to make more informed and effective decisions.
    
    3. Emotion AI: AI will be


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

     name

    ],

     and

     I

     am

     a

     [

    insert

     occupation

    ].

     I

    'm

     [

    insert

     age

     and

     nationality

    ],

     and

     I

     have

     always

     been

     passionate

     about

     [

    insert

     interest

     or

     hobby

    ].

     I

     am

     always

     looking

     for

     ways

     to

     learn

     and

     improve

    ,

     and

     I

     am

     constantly

     seeking

     new

     experiences

     and

     knowledge

    .

     I

     am

     always

     open

     to

     new

     ideas

     and

     perspectives

    ,

     and

     I

     thrive

     on

     the

     opportunity

     to

     make

     a

     difference

     in

     the

     world

    .

     I

     enjoy

     meeting

     new

     people

     and

     connecting

     with

     others

    ,

     and

     I

     am

     always

     looking

     for

     ways

     to

     inspire

     others

    .

     If

     you

     have

     a

     question

     or

     need

     help

    ,

     don

    't

     hesitate

     to

     ask

     me

    .

     I

     am

     here

     to

     help

    .

     How

     can

     I

     get

     to

     know

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     third

     largest

     city

     in

     the

     European

     Union

     by

     population

     and

     is

     known

     for

     its

     stunning

     architecture

    ,

     art

    ,

     and

     culture

    .

     Paris

     is

     home

     to

     many

     iconic

     landmarks

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     famous

     for

     its

     rich

     history

    ,

     including

     the

     French

     Revolution

    ,

     the

     French

     Revolution

    ,

     and

     the

     Impact

     of

     World

     War

     II

    .

     Paris

     has

     a

     diverse

     population

     and

     is

     known

     for

     its

     annual

     celebrations

    ,

     such

     as

     the

     E

    iff

    el

     Tower

     Parade

     and

     the

     World

     of

     Christmas

    .

     Overall

    ,

     Paris

     is

     a

     cultural

     and

     historic

     center

     of

     France

    ,

     known

     for

     its

     art

    ,

     architecture

    ,

     and

     cuisine

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     a

     highly

     dynamic

     one

    ,

     with

     many

     possible

     trends

     and

     innovations

     that

     could

     shape

     the

     landscape

     of

     the

     technology

     sector

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     and

     innovations

     that

     may

     be

     present

     in

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

     AI

    -driven

     drug

     discovery

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     it

     could

     lead

     to

     breakthrough

    s

     in

     drug

     discovery

     that

     could

     save

     lives

    .

     For

     example

    ,

     AI

     could

     be

     used

     to

     analyze

     large

     datasets

     of

     existing

     drug

     candidates

     to

     identify

     potential

     targets

     for

     new

     therapies

    .

     This

     could

     lead

     to

     faster

     and

     more

     efficient

     drug

     development

    ,

     as

     well

     as

     improved

     patient

     outcomes

    .
    


    2

    .

     AI

    -powered

     autonomous

     vehicles

    :

     As

     AI

     technology

     becomes

     more

    



```python
llm.shutdown()
```
