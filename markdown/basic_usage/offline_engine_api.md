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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.06it/s]


    2026-04-30 22:48:46,716 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 22:48:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  3.99it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:04,  9.00it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 20.75it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 17.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 17.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 17.96it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 17.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.94it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.66it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.66it/s] Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.66it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.66it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=640 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  50%|█████     | 29/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  50%|█████     | 29/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  50%|█████     | 29/58 [00:00<00:00, 41.56it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  50%|█████     | 29/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.06it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.21it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=48 avail_mem=71.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.21it/s]

    Capturing num tokens (num_tokens=48 avail_mem=71.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=32 avail_mem=71.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=28 avail_mem=71.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=24 avail_mem=71.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=12 avail_mem=71.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.73it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 39.69it/s]


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
    Generated text:  Lisa and I’m 24 years old. I’m a dedicated mom, teacher, and activist for social justice.
    I’m an advocate for the rights of all people, and I believe that everyone deserves the right to be free from oppression and discrimination, and to be able to live in a society where everyone is treated with respect and dignity. I am committed to using my platform as a teacher to educate people on the importance of social justice and how we can make a positive change in our communities. I believe in the power of collective action and working together to create a better world for all people.
    I'm currently a teacher at a community
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking official of the central government of the People's Republic of China. What is the function of the president of the United States?
    
    A. To oversee the entire country
    B. To serve as the head of state of the United States
    C. To represent the United States in the United Nations
    D. To represent the United States in the United Nations Security Council
    To determine the correct answer, let's analyze each option step by step:
    
    A. To oversee the entire country
    - This is not a function of the president of the United States. The president is responsible for running the country, not overseeing it.
    
    B.
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of France, and that is Paris, the city of love. Paris is a big city, and it has big buildings, big roads, and big parks. When you walk down the main street of Paris, you can see the city from the sky. When you look out of your car window, you can see the city from the window. The city is like a big castle. It has lots of interesting things to see and hear. A big band of singers plays jazz music in the city. There are a lot of gardens in the city. The city has a lot of museums, and there is a big zoo. The
    ===============================
    Prompt: The future of AI is
    Generated text:  in an area that’s just as intriguing as it is daunting: manufacturing. One of the most promising areas to explore in this space is the creation of new materials that will allow for the production of electronics at a level that is several orders of magnitude faster than the current state.
    The technologies are still very much in the early stages of development, and will require a lot of testing and modification to be put into production. But as technology becomes more and more advanced, and as the demand for faster, more efficient electronics grows, the world will be in a much better position to solve some of the world’s biggest problems.
    One of the most exciting


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


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its rich history, art, and culture, and is a popular tourist destination for visitors from around the world. The city is also home to many important institutions, including the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage, and continues to be a major center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be a growing emphasis on ensuring that they are used ethically and responsibly. This may include considerations of bias, privacy, and transparency.
    
    2. Greater integration with human decision-making: AI systems are likely to become more integrated with human decision-making
    


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
    Generated text:  [Your Name], and I am [Your Profession]. I am a [Your Profession] who has always been fascinated by the idea of [Your Passion], and I have dedicated my life to pursuing it. I believe that the best way to achieve my goals is to work hard and stay true to myself. I am always up for a challenge, and I love taking risks and pushing the boundaries. What is your passion and what inspired you to pursue it? [Your Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is the largest city in the European Union and one of the world's most populous urban areas. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Marne-la-Vallée. The city is known for its vibrant culture, beautiful architecture, and annual festivals. Paris has a rich history dating back to ancient times and is often referred to as the "City of a Hundred Faces" due to its diverse population and unique architecture. It is also home to many international organizations such as the United Nations and the European Parliament
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be dominated by the development of more advanced and complex systems. AI is expected to continue to develop at a rapid pace and incorporate more sophisticated algorithms, machine learning, and computer vision. In terms of applications, we can expect more advanced autonomous driving technology, self-driving cars, and virtual and augmented reality experiences. AI will also be used for more precise medical diagnosis, personalized medicine, and personalized education.
    In terms of ethics, we can expect more consideration of AI's potential impact on employment and inequality. We may see more AI systems that are biased or unfairly discriminatory, or systems that exacerbate existing social inequalities. However, we can also


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

    'm

     a

     [

    job

     title

    ]

     in

     [

    company

     name

    ].

     I

     love

     [

    reason

     why

     I

    'm

     passionate

     about

     the

     job

    ].

     I

    'm

     a

     dedicated

     [

    job

     title

    ]

     who

     enjoys

     [

    reason

     why

    ]

     and

     [

    reason

     why

    ]

     and

     is

     always

     striving

     to

     be

     the

     best

    .

     [

    Name

    ]

     is

     passionate

     about

     [

    reason

     why

    ]

     and

     [

    reason

     why

    ]

     and

     is

     always

     eager

     to

     learn

     and

     grow

    .

     I

     enjoy

     [

    reason

     why

    ]

     and

     [

    reason

     why

    ]

     and

     am

     always

     looking

     for

     opportunities

     to

     improve

     my

     skills

     and

     knowledge

    .

     [

    Name

    ]

     is

     a

     dedicated

     [

    job

     title

    ]

     who

     is

     always

     eager

     to

     learn

     and

     grow

    .

     I

    'm

     passionate

     about

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     and

     the

     capital

     of

     France

     and

     the

     second

    -largest

     city

     in

     Europe

     by

     population

     after

     Rome

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     known

     for

     its

     rich

     culture

    ,

     vibrant

     nightlife

    ,

     and

     historic

     architecture

    .

     Paris

     is

     also

     home

     to

     many

     iconic

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

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

     and

     is

     known

     for

     its

     artistic

    ,

     literary

    ,

     and

     intellectual

     heritage

    .

     Paris

     is

     an

     important

     center

     for

     business

    ,

     politics

    ,

     and

     culture

     in

     France

     and

     the

     world

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

     Light

    "

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

    ,

     but

     some

     trends

     that

     are

     likely

     to

     be

     prominent

     include

    :
    


     

     

    1

    .

     Increased

     automation

    :

     AI

     is

     likely

     to

     become

     more

     prevalent

     in

     the

     workplace

     as

     machines

     and

     algorithms

     are

     used

     to

     automate

     tasks

     and

     processes

     that

     are

     normally

     done

     by

     humans

    .


     

     

    2

    .

     Personal

    ization

    :

     AI

     is

     likely

     to

     become

     more

     personalized

    ,

     as

     machines

     are

     able

     to

     learn

     from

     data

     and

     adjust

     their

     behavior

     based

     on

     user

     preferences

     and

     behavior

    .


     

     

    3

    .

     Integration

     with

     other

     technologies

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

     and

     machine

     learning

    ,

     to

     enable

     more

     sophisticated

     and

     sophisticated

     applications

    .


     

     

    4

    .

     Increased

     ethical

     concerns

    :

     As

     AI

     becomes

    



```python
llm.shutdown()
```
