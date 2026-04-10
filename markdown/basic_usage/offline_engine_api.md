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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.44it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.43it/s]


    2026-04-10 11:03:16,987 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 11:03:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:09,  5.18it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:02<00:03, 11.78it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:02<00:03, 11.78it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:02<00:03, 11.78it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:02<00:03, 11.78it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:02<00:03, 11.78it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:03<00:03, 11.78it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:03<00:03, 11.78it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:03<00:03, 11.78it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 25.32it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 25.32it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 25.32it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 25.32it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 25.32it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 25.32it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 25.32it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:01, 25.32it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 39.16it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 39.16it/s]

    Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 45.52it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 45.52it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 45.52it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 45.52it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 45.52it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 45.52it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 45.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.56it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=60.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.31 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=960 avail_mem=60.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s] Capturing num tokens (num_tokens=896 avail_mem=60.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]

    Capturing num tokens (num_tokens=832 avail_mem=60.31 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=768 avail_mem=60.31 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=768 avail_mem=60.31 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.44it/s]Capturing num tokens (num_tokens=704 avail_mem=60.31 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.44it/s]Capturing num tokens (num_tokens=640 avail_mem=60.30 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.44it/s]Capturing num tokens (num_tokens=576 avail_mem=60.30 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.44it/s]Capturing num tokens (num_tokens=512 avail_mem=60.29 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.44it/s]Capturing num tokens (num_tokens=480 avail_mem=60.31 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.44it/s]Capturing num tokens (num_tokens=448 avail_mem=60.31 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.44it/s]Capturing num tokens (num_tokens=448 avail_mem=60.31 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=416 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=384 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=352 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.59it/s]

    Capturing num tokens (num_tokens=320 avail_mem=60.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=288 avail_mem=60.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=256 avail_mem=60.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=256 avail_mem=60.29 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.57it/s]Capturing num tokens (num_tokens=240 avail_mem=60.28 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.57it/s]Capturing num tokens (num_tokens=224 avail_mem=60.28 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.57it/s]Capturing num tokens (num_tokens=208 avail_mem=60.28 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.57it/s]Capturing num tokens (num_tokens=192 avail_mem=60.28 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.57it/s]Capturing num tokens (num_tokens=176 avail_mem=60.27 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.57it/s]Capturing num tokens (num_tokens=160 avail_mem=60.27 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.57it/s]Capturing num tokens (num_tokens=160 avail_mem=60.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.25it/s]Capturing num tokens (num_tokens=144 avail_mem=60.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.25it/s]Capturing num tokens (num_tokens=128 avail_mem=60.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.25it/s]

    Capturing num tokens (num_tokens=112 avail_mem=60.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.25it/s]Capturing num tokens (num_tokens=96 avail_mem=60.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.25it/s] Capturing num tokens (num_tokens=80 avail_mem=60.25 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.25it/s]Capturing num tokens (num_tokens=64 avail_mem=60.25 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.25it/s]Capturing num tokens (num_tokens=64 avail_mem=60.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.93it/s]Capturing num tokens (num_tokens=48 avail_mem=60.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.93it/s]Capturing num tokens (num_tokens=32 avail_mem=60.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.93it/s]Capturing num tokens (num_tokens=28 avail_mem=60.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.93it/s]Capturing num tokens (num_tokens=24 avail_mem=60.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.93it/s]Capturing num tokens (num_tokens=20 avail_mem=60.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.93it/s]Capturing num tokens (num_tokens=16 avail_mem=60.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.93it/s]Capturing num tokens (num_tokens=16 avail_mem=60.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 51.50it/s]Capturing num tokens (num_tokens=12 avail_mem=60.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 51.50it/s]

    Capturing num tokens (num_tokens=8 avail_mem=60.22 GB):  95%|█████████▍| 55/58 [00:01<00:00, 51.50it/s] Capturing num tokens (num_tokens=4 avail_mem=60.22 GB):  95%|█████████▍| 55/58 [00:01<00:00, 51.50it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:01<00:00, 44.52it/s]


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
    Generated text:  John. I like to write and share stories about my personal life and experiences. I'm currently looking for a new job and need some advice on how to apply for it. I'm relatively new to the job market and I'm trying to decide on the best way to apply for a position. Can you please share some advice on the best way to apply for a job?
    John
    Hello John, I'm glad to hear that you're looking for a new job! Here are some tips on how to apply for a job:
    
    1. Research the company: Before applying, research the company and its mission and values to make sure it align
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the vice president. If John is the vice president and Mary is the president, how many candidates are there in total for the vice president position?
    To determine the total number of candidates for the vice president position, we need to consider the roles of the vice president and the president. According to the problem, John is the vice president and Mary is the president. Since these are the only two positions that can exist, the total number of candidates is simply the number of positions, which is 2.
    
    Therefore, the total number of candidates for the vice president position is \(\boxed{2}\).
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and the population is 2.6 million. 
    
    Let's call a town a "cultural capital" if it is populated by more people than Paris and has more than 100,000 inhabitants.
    
    In how many towns will the population be larger than 2.6 million and still be a "cultural capital"?
    
    To determine how many towns will have a population larger than 2.6 million and still be a "cultural capital," we need to identify the towns that meet both criteria: being a "cultural capital" and having a population larger than 100,000
    ===============================
    Prompt: The future of AI is
    Generated text:  poised to be profound and transformative. Here are five trends that are likely to shape the future of artificial intelligence (AI) in the coming years:
    1. Integration with IoT: With the increasing integration of IoT into our everyday lives, there is a growing need for AI to be integrated seamlessly with IoT. This is likely to lead to a more efficient and cost-effective use of AI in various industries.
    2. Data privacy and security: As AI systems become more complex and sophisticated, data privacy and security become increasingly important. With the increasing amount of data being generated and analyzed, there is a need for robust data protection measures to ensure the privacy and


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love About My Profession]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Personality Trait] who is [What I Do Well]. I'm always ready to learn and improve. I'm excited to meet you and see what you have to offer. How can I get to know you better? I'd love to hear more about your background and how you got into the industry. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a popular tourist destination, with its rich history, art, and cuisine attracting millions of visitors each year. The city is also home to numerous museums, theaters, and other cultural institutions. Paris is a vibrant and dynamic city with a rich history and a strong sense of community. Its status as the capital of France is a testament to its importance and significance as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the potential trends that are likely to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased Use of AI in Finance: AI is already being used in
    


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
    Generated text:  [insert name] and I'm [insert profession and age]. I'm passionate about [insert personal passion or area of interest]. I enjoy [insert hobbies or activities]. I am a [insert occupation] with a [insert role, such as student, staff member, or entrepreneur]. I have [insert accomplishments, such as completing [insert project or task]], and I strive to be [insert personal goal or ambition]. I am [insert nationality, ethnicity, or culture], and I value [insert personal values or beliefs]. I am [insert any other relevant traits or qualities]. I am [insert any personal statement or short biography]. [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a sprawling city with a rich history, including the 12th-century Eiffel Tower, a major landmark of the city.
    
    Can you paraphrase this statement to make it easier to understand for someone who is not familiar with French language? The capital of France is Paris. It is a huge city with a history that dates back to the 12th century. It also has a famous landmark called the Eiffel Tower, which is one of the most recognizable symbols of the city. The capital of France is a very big place with many buildings and interesting things to see. 
    
    This is a simplified version
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and there are many potential areas where it could continue to grow and evolve. Here are some possible trends that may shape AI in the coming years:
    
    1. Increased accuracy and efficiency: As AI technologies advance, we can expect to see even more accurate predictions and more efficient processes. This will likely lead to a wider range of applications for AI in various industries.
    
    2. Enhanced privacy and security: With the increasing amount of data being collected and processed, it's essential to ensure that AI systems are secure and protect personal information. This will likely lead to stricter regulations and better privacy protections for AI-based technologies.
    
    3. AI becoming more collaborative:


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

     am

     a

     [

    insert

     your

     profession

     or

     field

     of

     expertise

    ].

     I

     have

     always

     been

     passionate

     about

     [

    insert

     something

     specific

     that

     interests

     you

    ].

     I

     have

     also

     been

     involved

     in

     various

     [

    insert

     activities

     in

     your

     field

    ]

     and

     have

     been

     recognized

     for

     my

     dedication

     to

     [

    insert

     something

     specific

    ,

     such

     as

     a

     competition

    ,

     event

    ,

     or

     award

     you

    've

     won

    ].

     I

     believe

     that

     being

     open

    -minded

    ,

     patient

    ,

     and

     always

     willing

     to

     learn

     is

     what

     sets

     me

     apart

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     improve

    .

     I

     would

     love

     to

     hear

     about

     my

     experiences

     and

     what

     makes

     me

     unique

    .

     How

     can

     I

     get

     in

     touch

     with

     you

     if

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     in

     the

     heart

     of

     the

     Paris

     region

    ,

     on

     the

     River

     Se

    ine

    ,

     and

     is

     the

     largest

     city

     in

     metropolitan

     France

    ,

     the

     second

     largest

     in

     the

     United

     Kingdom

    ,

     and

     the

     third

     largest

     in

     the

     world

     by

     population

    .

     Paris

     is

     a

     major

     financial

     center

    ,

     and

     has

     a

     rich

     cultural

     heritage

     that

     is

     celebrated

     in

     its

     museums

     and

     other

     cultural

     institutions

    .

     The

     city

     is

     also

     home

     to

     important

     government

     institutions

    ,

     including

     the

     Council

     of

     State

    ,

     the

     Senate

    ,

     and

     the

     Chamber

     of

     De

    puties

    .

     It

     is

     also

     the

     country

    ’s

     second

    -largest

     city

     by

     size

     and

     one

     of

     the

     country

    ’s

     most

     important

     economic

     and

     cultural

     centers

    .

     Paris

     is

     known

     for

     its

     diverse

     and

     historic

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     a

     number

     of

     trends

     that

     are

     set

     to

     shape

     how

     we

     interact

     with

     technology

     and

     work

     in

     the

     workplace

    .

     Some

     of

     the

     most

     notable

     trends

     include

    :
    


    1

    .

     Increased

     sophistication

     of

     AI

    :

     As

     AI

     becomes

     more

     powerful

     and

     capable

    ,

     it

     will

     continue

     to

     learn

     and

     evolve

    ,

     becoming

     more

     capable

     of

     performing

     tasks

     that

     were

     previously

     impossible

    .

     This

     will

     lead

     to

     even

     more

     complex

     applications

     of

     AI

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     AI

     will

     become

     more

     accessible

    :

     As

     AI

     becomes

     more

     advanced

     and

     efficient

    ,

     it

     will

     become

     more

     accessible

     to

     a

     wider

     range

     of

     people

    .

     This

     will

     likely

     lead

     to

     the

     development

     of

     more

     personalized

     and

     user

    -friendly

     AI

    



```python
llm.shutdown()
```
