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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.01it/s]


    2026-04-28 18:50:15,345 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 18:50:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 20.46it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   2%|▏         | 1/58 [00:00<00:16,  3.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   2%|▏         | 1/58 [00:00<00:16,  3.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   2%|▏         | 1/58 [00:00<00:16,  3.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   5%|▌         | 3/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   5%|▌         | 3/58 [00:00<00:07,  7.47it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   5%|▌         | 3/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   5%|▌         | 3/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):  10%|█         | 6/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  10%|█         | 6/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.50it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.42it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.03it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.03it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.03it/s] Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.03it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.03it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:01<00:01, 32.03it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  43%|████▎     | 25/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  43%|████▎     | 25/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  43%|████▎     | 25/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  43%|████▎     | 25/58 [00:01<00:00, 36.25it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  50%|█████     | 29/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  50%|█████     | 29/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  50%|█████     | 29/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=416 avail_mem=72.25 GB):  50%|█████     | 29/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  50%|█████     | 29/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  50%|█████     | 29/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.41it/s]

    Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.94it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.94it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.94it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.94it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.94it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.94it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=32 avail_mem=71.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=28 avail_mem=71.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=24 avail_mem=71.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=20 avail_mem=71.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.46it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=16 avail_mem=71.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=12 avail_mem=71.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=8 avail_mem=71.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.83it/s] Capturing num tokens (num_tokens=4 avail_mem=71.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB): 100%|██████████| 58/58 [00:01<00:00, 31.56it/s]


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
    Generated text:  Sven Thiele and I am an Associate Professor of Mathematics at the University of Connecticut. I have been teaching at the college level since 2003, with my Ph.D. in mathematics from the University of Vienna in 2004.\nMy research focuses on the geometry of metric spaces and, in particular, the structure and properties of the unit ball in metric spaces. I have been developing new tools to understand and attack problems in discrete geometry, and am interested in developing new problems and tools in this area.\nI am also the director of the Mathematics Undergraduate Research Program. My own research students have won numerous
    ===============================
    Prompt: The president of the United States is
    Generated text:  in Washington, D. C., for a three-day trip. From Monday to Friday, he spends 1/7 of his time at the White House and the rest of his time at various executive offices in the Executive Office Building. If he spends 3 hours at each executive office, how much total time does he spend at the Executive Office Building over the three-day trip?
    
    To determine the total time the president spends at the Executive Office Building over the three-day trip, we need to follow these steps:
    
    1. Calculate the total time spent at the White House.
    2. Determine the total time spent at the Executive Office Building.
    3
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) London
    C) Rome
    D) Berlin
    
    To determine the capital of France, we need to recall the historical and political context of France. The capital of France is typically the largest city and is the seat of government, administration, and culture for the country.
    
    1. **Paris**: Paris is the capital of France, the largest city in Europe by population. It is known for its iconic Eiffel Tower, Notre-Dame Cathedral, and the annual Eiffel Tower Tour.
    2. **London**: London is the capital of the United Kingdom and is the largest city in the European Union.
    ===============================
    Prompt: The future of AI is
    Generated text:  in human hands
    
    Updated: Jun 15, 2020
    
    The path of AI is to be like a man on a river.
    
    If a river flows to a point, the most effective way to describe the future of AI is as a man on a river. How will he get there? Will he ride on a horse or a bullet train, or by foot? Will he follow the flow of the river or will he follow the revolution in technology that is ahead? Will he leap over the barriers that stand in his way, or will he break free and move in a new direction?
    
    Here is my attempt to give


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


    Generated text:  Paris, also known as "La Ville Flottante" (floating city) due to its floating population of people. It is the largest city in Europe and the second largest city in the world by population. Paris is known for its rich history, art, and cuisine, and is a major tourist destination. It is also home to the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a cultural and political center of France and a major economic hub. It is also known for its fashion industry and its role in the French Revolution. The city is home to many famous landmarks and museums, including the Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: As AI becomes more advanced, it is likely to automate more tasks, freeing up human workers to focus on more complex and creative work. This could lead to a shift in the job market, with many jobs being replaced by AI.
    
    2. Improved privacy and security: As AI becomes more sophisticated, it is likely to require more data to function effectively. This could lead to increased privacy concerns, as AI systems may be able to learn about individuals' behavior and
    


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
    Generated text:  [Name], and I am a/an [Character] who has been a/an [Objective] for [Number of Years]. I have always loved to [Describe One Interesting Experience or Activity] and am always eager to [Describe Something New or Unique]. I am [Age], [Gender], and I live in [Location]. What is your favorite hobby or activity? How do you stay healthy and what are some things you enjoy doing in your free time? Lastly, I am [Describe Any Special Talents or Attributes]. I hope you find me interesting and would like to meet you! 😊👍✨
    
    [Your Name]
    [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement can be expanded to include the following facts: Paris is the capital of France, which is the largest country in Europe, and it is home to numerous museums and art galleries, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Additionally, Paris is known for its world-renowned fashion industry, which has inspired countless fashion designers and trends. The city is also famous for its iconic Eiffel Tower, which stands as a symbol of Paris and has become a major tourist destination. Lastly, Paris is a cosmopolitan city with a rich cultural scene, including theaters, galleries,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see a proliferation of new technologies and applications, driven by advances in machine learning, computer science, and hardware. Here are some possible future trends in AI:
    
    1. Increased AI-powered autonomous vehicles: As AI technology continues to improve, we can expect autonomous vehicles to become more prevalent in our daily lives. These vehicles will be equipped with sensors and cameras that can detect and respond to a wide range of situations, including lane keeping, avoiding pedestrians and obstacles, and emergency braking.
    
    2. AI-powered healthcare: AI is already being used in various healthcare applications, including personalized medicine, drug discovery, and image recognition. In the future, we


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

    major

    ity

     of

     the

     time

    ]

     AI

     assistant

     that

     I

     maintain

     and

     does

     not

     have

     emotions

    .

     I

     am

     programmed

     to

     provide

     helpful

     and

     informative

     responses

     to

     the

     best

     of

     my

     abilities

    ,

     but

     I

     don

    't

     have

     personal

     feelings

     or

     opinions

    .

     My

     purpose

     is

     to

     assist

     users

     in

     achieving

     their

     goals

     and

     provide

     them

     with

     the

     information

     they

     need

    ,

     without

     imposing

     any

     personal

     bias

     or

     agenda

    .

     If

     there

     is

     anything

     I

     can

     do

     to

     assist

     you

     today

    ,

     just

     let

     me

     know

    .

     [

    Your

    self

    ].

     


    Is

     it

     possible

     to

     make

     this

     question

     more

     specific

     to

     the

     AI

     assistant

     character

    ?

     I

     have

     a

     question

     and

     I

     would

     like

     to

     have

     a

     more

     specific

     introduction

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     historic

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     ancient

     Roman

     era

    .

     Paris

     is

     famous

     for

     its

     beautiful

     architecture

    ,

     world

    -ren

    owned

     museums

    ,

     and

     diverse

     cuisine

    .

     It

     is

     also

     known

     for

     its

     annual

     art

     festival

     and

     annual

     E

    iff

    el

     Tower

     celebration

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     has

     become

     a

     cultural

     hub

     for

     France

    .

     Its

     iconic

     landmarks

     include

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

     is

     also

     home

     to

     many

     museums

    ,

     theaters

    ,

     and

     parks

    ,

     and

     is

     a

     major

     transportation

     hub

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Love

    "

     and

     is

     known

     for

     its

     romantic

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     anticipated

     to

     evolve

     rapidly

    ,

     and

     there

     are

     several

     trends

     that

     are

     expected

     to

     shape

     its

     development

    .

     Some

     of

     the

     most

     promising

     trends

     include

    :
    


    1

    .

     Improved

     accuracy

     and

     reliability

    :

     AI

     is

     becoming

     more

     accurate

     and

     reliable

     as

     researchers

     continue

     to

     develop

     new

     algorithms

     and

     models

    .

     AI

     systems

     can

     now

     achieve

     higher

     levels

     of

     precision

     and

     accuracy

     in

     tasks

     such

     as

     image

     recognition

    ,

     speech

     recognition

    ,

     and

     natural

     language

     processing

    .
    


    2

    .

     Personal

    ization

     and

     customization

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     will

     be

     able

     to

     learn

     and

     adapt

     to

     the

     individual

     needs

     and

     preferences

     of

     users

    .

     Personal

    ized

     AI

     will

     be

     able

     to

     offer

     tailored

     recommendations

     and

     solutions

    ,

     resulting

     in

     better

     experiences

     and

     more

     efficient

     operations

    



```python
llm.shutdown()
```
