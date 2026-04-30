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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.42it/s]


    2026-04-30 09:20:35,890 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 09:20:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.01it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 19.90it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 28.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.58 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.58 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.58 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.57 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.57 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.56 GB):   9%|▊         | 5/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.56 GB):   9%|▊         | 5/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.55 GB):   9%|▊         | 5/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.55 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.55 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.55 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.55it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=56.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.53 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.53 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.53 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=960 avail_mem=56.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s] Capturing num tokens (num_tokens=896 avail_mem=56.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]

    Capturing num tokens (num_tokens=832 avail_mem=56.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=832 avail_mem=56.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=768 avail_mem=56.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=704 avail_mem=56.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=640 avail_mem=56.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=576 avail_mem=56.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=512 avail_mem=56.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=512 avail_mem=56.48 GB):  50%|█████     | 29/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=480 avail_mem=56.49 GB):  50%|█████     | 29/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=448 avail_mem=56.49 GB):  50%|█████     | 29/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=416 avail_mem=56.49 GB):  50%|█████     | 29/58 [00:00<00:00, 40.67it/s]

    Capturing num tokens (num_tokens=384 avail_mem=56.49 GB):  50%|█████     | 29/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=352 avail_mem=56.48 GB):  50%|█████     | 29/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=352 avail_mem=56.48 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=320 avail_mem=56.48 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=288 avail_mem=56.48 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=256 avail_mem=56.47 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=240 avail_mem=56.47 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=224 avail_mem=56.47 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=224 avail_mem=56.47 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=208 avail_mem=56.46 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=192 avail_mem=56.46 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.68it/s]

    Capturing num tokens (num_tokens=176 avail_mem=56.46 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=160 avail_mem=56.46 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=144 avail_mem=56.45 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=144 avail_mem=56.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=128 avail_mem=56.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=112 avail_mem=56.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=96 avail_mem=56.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.31it/s] Capturing num tokens (num_tokens=80 avail_mem=56.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=64 avail_mem=56.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.31it/s]

    Capturing num tokens (num_tokens=64 avail_mem=56.43 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=48 avail_mem=56.43 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=32 avail_mem=56.43 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=28 avail_mem=56.42 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=24 avail_mem=56.42 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=20 avail_mem=56.42 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=20 avail_mem=56.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=16 avail_mem=56.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=12 avail_mem=56.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=8 avail_mem=56.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.76it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=56.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=4 avail_mem=56.40 GB): 100%|██████████| 58/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=4 avail_mem=56.40 GB): 100%|██████████| 58/58 [00:01<00:00, 37.92it/s]


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
    Generated text:  Tamer and I am a dedicated writer and blogger in the creative writing field. I've been in the literary world since 2009 when I started working as an editor at a literary magazine, "Dance of Words." I've since published my first short story collection, "The Trumpet of the Headless Horseman," which was published by No Starch Press in 2014. I'm also a regular contributor to "Midwest Writers," a blog for Midwest writers, and a regular contributor to "The Creative Hour," a blog for writers and writers' communities. Tamer has also worked with "The
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ________ ( ). A. official B. senator C. elected official D. public figure
    Answer:
    C
    
    According to the passage, what does "the more" mean in the sentence "The price of the tickets is more than the price of the tickets"? 
    A. More expensive
    B. More affordable
    C. More important
    D. More convenient
    Answer:
    A
    
    What does "The more" mean in the sentence "The price of the tickets is more than the price of the tickets"? A. More expensive B. More affordable C. More important D. More convenient
    Answer:
    A
    
    In general,
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    
    A: Paris  
    B: Paris (abbreviation)  
    C: Montreuil  
    D: Montreuil (abbreviation) To determine the capital of France, we need to recall the official capital cities of France. The capital of France is Paris. The capital city is always capitalized in titles and is the official name given to it. Here are the steps to identify the correct answer:
    
    1. Identify the capital city of France.
    2. Capitalize the name of the capital city if it is being used as the title.
    
    The capital of France is Paris. The capital city is always capitalized in titles.
    
    Therefore, the correct
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s changing our world in ways we can’t even imagine.
    The future of AI is here, and it’s changing our world in ways we can’t even imagine.
    With the rise of quantum computing, the next step towards AI and machine learning is near. The field of quantum computing is known to be a major technological breakthrough, with the potential to significantly improve the performance and efficiency of AI systems.
    In this article, we’ll explore the current state of quantum computing and how it’s transforming the world of AI. We’ll also discuss the implications of this technology, including its potential to revolutionize industries such as healthcare, transportation


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few key points about you, such as your age, gender, occupation, and any other relevant information]. And what can you tell me about your work at [company name]? I'm looking forward to hearing about your experiences and accomplishments at [company name]. And what can you tell me about your hobbies or interests? I'm always looking for new experiences and adventures, so I'm excited to hear about your hobbies and interests as well
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. It is located on the Seine River and is the seat of government, administration, and culture for the French Republic. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a popular tourist destination and is known for its fashion, art, and cuisine. It is also a major economic center and a major transportation hub. The city is home to many international organizations and is a hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of devices and systems, from smart homes to self-driving cars. As more devices and systems become connected to the internet, we can expect to see even more integration of AI into our daily lives.
    
    2. Greater use of AI in healthcare: AI is already being used to improve the accuracy of medical diagnoses and treatments, and we can expect to see even more use of AI in healthcare in the future. AI-powered tools are already being used to analyze medical images, predict disease outbreaks, and
    


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
    Generated text:  [Name], and I come from [Location]. I've always been fascinated by [Industry] and have spent my entire life exploring its challenges and rewards. With [Number] years of experience in [Industry], I'm here to help you navigate through any obstacles or challenges that may come your way. Let's get to know each other, and I'll help you achieve your goals! #Experience #IndustryExpert #StartsSmall #TalkAboutIt #HelloWorld #NoseToIt #NeverGiveUp #SelfEffort #Relief #Success #ChallengeSolved #Growth #Support #Empowerment #SelfAwareness
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    The answer is 3. Paris. The capital of France is Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several trends and developments that will shape its development in the years to come. Some of the key trends and developments include:
    
    1. Increased reliance on AI for automation: As more industries automate or digitize processes, AI will become increasingly important for tasks such as data analysis, predictive analytics, and natural language processing.
    
    2. Integration of AI with other technologies: AI is increasingly being integrated with other technologies such as blockchain, quantum computing, and cloud computing, to create new and innovative applications.
    
    3. Advancements in deep learning: Advances in deep learning, the subset of machine learning that involves artificial neural networks, will enable AI to


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

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    _

    !

     I

    'm

     from

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    _

     at

     __

    ________

    .

     I

    'm

     currently

     in

     my

     __

    ________

    _

     year

     and

     am

     currently

     in

     the

     __

    ________

    _

     grade

    .

     I

     like

     to

     __

    ________

    _

     and

     I

     enjoy

     __

    ________

    .

     My

     favorite

     hobby

     is

     __

    ________

    .

     I

    'm

     going

     to

     start

     a

     new

     __

    ________

    _

     this

     year

     and

     hope

     to

     __

    ________

    _

     it

    .

     As

     a

    /an

     __

    ________

    _

     I

    'm

     going

     to

     __

    ________

    _

     new

     friends

     and

     __

    ________

    _

     to

     make

     new

     memories

    .

     I

    'm

     ready

     to

     __

    ________

    _

     and

     have

     a

    /an

     __

    ________

    _

    !

     I

    'm

     excited

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     fact

    ually

     accurate

     and

     provides

     a

     brief

     overview

     of

     the

     capital

     city

    's

     location

     and

     significance

    .

     However

    ,

     it

     could

     also

     be

     ph

    r

    ased

     differently

     or

     expanded

     upon

     to

     offer

     more

     context

     or

     additional

     information

    ,

     depending

     on

     the

     audience

     and

     purpose

     of

     the

     statement

    .

     For

     example

    ,

     if

     the

     statement

     is

     being

     used

     to

     create

     an

     educational

     resource

     about

     France

    ,

     it

     could

     be

     more

     concise

     and

     focus

     on

     the

     capital

    's

     unique

     characteristics

     or

     history

    .

     On

     the

     other

     hand

    ,

     if

     the

     statement

     is

     being

     used

     in

     a

     broader

     context

    ,

     it

     could

     be

     expanded

     to

     include

     additional

     information

     about

     the

     city

    's

     landmarks

    ,

     cultural

     heritage

    ,

     or

     significant

     events

    .

     The

     purpose

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     influenced

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     more

     people

     become

     aware

     of

     the

     potential

     risks

     of

     AI

    ,

     there

     will

     be

     increasing

     pressure

     to

     develop

     AI

     that

     is

     more

     transparent

    ,

     accountable

    ,

     and

     aligned

     with

     ethical

     principles

    .
    


    2

    .

     AI

     with

     emotional

     intelligence

    :

     AI

     will

     be

     designed

     to

     understand

     and

     respond

     to

     the

     emotions

     of

     people

    ,

     rather

     than

     just

     completing

     tasks

    .

     This

     could

     lead

     to

     more

     compassionate

     and

     empath

    etic

     AI

     that

     can

     handle

     complex

     emotional

     situations

    .
    


    3

    .

     Personal

    ization

    :

     AI

     will

     be

     able

     to

     learn

     and

     adapt

     to

     individual

     users

    ,

     allowing

     for

     more

     personalized

     experiences

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     use

     of

    



```python
llm.shutdown()
```
