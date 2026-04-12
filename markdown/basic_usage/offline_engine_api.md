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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.55it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]


    2026-04-12 23:33:09,657 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 23:33:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.32it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.32it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.32it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.32it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.32it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.32it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.32it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.32it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.32it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.32it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.32it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.32it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.32it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 35.32it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 39.44it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 39.44it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 39.44it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 39.44it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 39.44it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 39.44it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 39.44it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 39.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  50%|█████     | 29/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.09it/s]

    Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.42it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.42it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.42it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.42it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.42it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.19it/s]

    Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.10it/s]

    Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=20 avail_mem=137.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=20 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=16 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=12 avail_mem=137.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=8 avail_mem=137.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.52it/s] Capturing num tokens (num_tokens=4 avail_mem=137.21 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.52it/s]

    Capturing num tokens (num_tokens=4 avail_mem=137.21 GB): 100%|██████████| 58/58 [00:01<00:00, 38.62it/s]


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
    Generated text:  Amy. I'm a very kind and helpful person. I have many friends, including Lucy, Jack, and Dick. My favorite color is green, and I like to play computer games. My favorite food is ice cream. I always make lots of ice cream. I love helping people. I often visit my friend Sally to have lunch. What do you think Amy is like? Amy is a kind and helpful person. She is very patient and gives me lots of ice cream! ( ) 1. What's Amy like? A. She is a kind and helpful person. B. She is a hard-working person. C. She
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. He is a ____. 
    A. Leader of the United States 
    B. Leader of the Congress 
    C. Leader of the President's Administration 
    D. Leader of the Federal Government
    Answer:
    
    A
    
    A uniformly distributed number of copper spheres of radius r is suspended in a uniform electric field. The potential difference between any two adjacent spheres is ____. 
    A. proportional to the square of the radius r 
    B. proportional to r 
    C. proportional to the product of the radius r and the potential difference V 
    D. independent of the radius r
    Answer:
    
    D
    
    The most important and basic function of the
    ===============================
    Prompt: The capital of France is
    Generated text:  [closed]
    
    closed as not a real question by Américo Tavares, Joe, Vítor Riahi, Fabio Frati, user278927 Oct 23 '18 at 13:33
    
    It's [closed] but not exactly a question with an answer.
    If this question can be reworded to fit the rules in the help center, please edit the question.
    
    Which capital city of France is located on the Atlantic coast?
    A. Brussels
    B. Nice
    C. Bordeaux
    D. Nice
    The answer is "Nice". The answer was written by
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting. At one time it was considered a grim possibility and a threat to humanity. But thanks to advances in computing power and machine learning, AI is now a very powerful tool for solving complex problems. It is capable of doing things that were previously thought impossible, and can do so faster than ever before.
    
    One of the key benefits of AI is that it can improve efficiency and productivity. By automating tasks, it can help reduce the workload on human resources and improve the overall quality of work. AI can also be used to analyze data and identify patterns that would otherwise be difficult or impossible to identify.
    
    Another benefit of AI is that it


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich history dating back to the Roman Empire and the French Revolution. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. It is also home to many famous landmarks, including the Notre-Dame Cathedral and the Palace of Versailles. Paris is a vibrant and dynamic city that continues to evolve and change over time. The city is a major center of culture, politics
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, we may see even more widespread use of AI in healthcare, with the potential to improve diagnosis
    


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
    Generated text:  __________ and I am a/an __________. (Example: "Hello, my name is Jane Smith and I am a/an Research Scientist at NASA.")
    
    As you can see, I am a skilled scientist, passionate about exploring the mysteries of the universe. I am excited to share my expertise with you and learn from your questions. What's your name, and what can you tell me about yourself? Let's connect and learn more! 😊✨✨✨
    
    Make it more engaging with relevant words and phrases to paint a more vivid picture of your character. Also, add a touch of humor to make your introduction more memorable. Lastly,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an international metropolis renowned for its rich cultural heritage, picturesque old town, and cosmopolitan street life.
    That's a great start, but can you please add more details about the historical significance of Paris? I would love to know more about its cultural and artistic heritage.
    Certainly! Paris is a city with a rich cultural heritage that dates back to ancient times. The city has been an important center of trade and culture throughout history, with its most famous landmarks being the Eiffel Tower and Notre Dame Cathedral. Paris is also home to many famous museums, art galleries, and theaters, including the Louvre Museum, the Musée
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be rapidly evolving, and several key trends are shaping the field:
    
    1. Increased Use of AI in Health Care: AI is already being used in healthcare to improve patient outcomes, such as by analyzing medical images and identifying potential cancer diagnoses. Future trends could see more AI-based tools being integrated into healthcare systems, allowing for more personalized treatments and improved patient outcomes.
    
    2. Development of AI in the Financial Sector: AI is already being used to analyze financial data and identify fraud patterns, and future trends could see more widespread use in the financial sector. AI-powered chatbots and virtual assistants could become more common, providing valuable support and assistance to


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

     am

     an

     [

    occupation

    ]

     who

     has

     always

     been

     [

    background

     information

    ,

     such

     as

     having

     a

     particular

     skill

    ,

     experience

    ,

     or

     interest

    .

     What

     is

     your

     background

     and

     where

     did

     you

     learn

     or

     develop

     your

     expertise

    ?

     I

    'm

     always

     looking

     for

     opportunities

     to

     expand

     my

     knowledge

     and

     skill

    set

    ,

     and

     I

    'm

     excited

     to

     learn

     more

     about

     [

    subject

     or

     field

    ].

     How

     do

     you

     see

     yourself

     growing

     in

     the

     future

    ?

     Please

     feel

     free

     to

     add

     any

     interesting

     facts

     or

     anecdotes

     that

     come

     to

     mind

     that

     may

     help

     me

     understand

     you

     better

    .

     
    


    Remember

    ,

     your

     self

    -int

    roduction

     should

     be

     neutral

     and

     straightforward

    .

     It

     should

     capture

     your

     personality

     and

     interests

    ,

     while

     also

     indicating

     your

     interest

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     and

     its

     rich

     history

     dating

     back

     to

     the

     

    7

    th

     century

    .
    


    The

     capital

     city

     of

     France

     is

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

    ,

     with

     a

     population

     of

     over

     

    7

     million

     people

     and

     a

     bustling

     metropolitan

     area

    .

     The

     city

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     and

     its

     rich

     history

     dating

     back

     to

     the

     

    7

    th

     century

    .

     Paris

     is

     a

     major

     international

     center

     of

     culture

     and

     commerce

    ,

     with

     numerous

     museums

    ,

     museums

    ,

     and

     theaters

    ,

     as

     well

     as

     numerous

     art

     galleries

     and

     historic

     sites

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     promising

    ,

     with

     potential

     developments

     that

     could

     shape

     our

     lives

    ,

     industries

    ,

     and

     technologies

     in

     profound

     ways

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     are

     likely

     to

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

     use

     of

     AI

     for

     medical

     diagnosis

     and

     treatment

    :

     With

     the

     rise

     of

     AI

    -powered

     medical

     imaging

     and

     diagnostic

     tools

    ,

     we

     are

     likely

     to

     see

     greater

     use

     of

     AI

     for

     medical

     diagnosis

     and

     treatment

    .

     This

     could

     lead

     to

     more

     accurate

     and

     personalized

     diagnoses

    ,

     as

     well

     as

     more

     effective

     and

     efficient

     treatments

    .
    


    2

    .

     More

     autonomous

     and

     flexible

     robots

    :

     The

     development

     of

     robotic

     systems

     that

     can

     operate

     autonom

    ously

     and

     be

     flexible

     in

     their

     capabilities

     is

     a

     key

     area

     of

    



```python
llm.shutdown()
```
