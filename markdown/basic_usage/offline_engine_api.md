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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.61it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.61it/s]


    2026-05-02 06:14:59,119 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 06:14:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:50,  1.06it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:50,  1.06it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:50,  1.06it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:50,  1.06it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:50,  1.06it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:19,  2.59it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:19,  2.59it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:19,  2.59it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:19,  2.59it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:19,  2.59it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:19,  2.59it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:19,  2.59it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:05<00:19,  2.59it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:06,  6.16it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03, 11.32it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 25.70it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 34.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.44 GB):   3%|▎         | 2/58 [00:00<00:04, 13.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.44 GB):   3%|▎         | 2/58 [00:00<00:04, 13.48it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.44 GB):   3%|▎         | 2/58 [00:00<00:04, 13.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.44 GB):   7%|▋         | 4/58 [00:00<00:03, 14.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.44 GB):   7%|▋         | 4/58 [00:00<00:03, 14.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.43 GB):   7%|▋         | 4/58 [00:00<00:03, 14.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.43 GB):  10%|█         | 6/58 [00:00<00:03, 16.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.42 GB):  10%|█         | 6/58 [00:00<00:03, 16.71it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.42 GB):  10%|█         | 6/58 [00:00<00:03, 16.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.42 GB):  10%|█         | 6/58 [00:00<00:03, 16.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.42 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.41 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.41 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.37 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.79it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=53.37 GB):  21%|██        | 12/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.37 GB):  21%|██        | 12/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.37 GB):  21%|██        | 12/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.36 GB):  21%|██        | 12/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.81it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=53.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.33 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.34it/s]Capturing num tokens (num_tokens=960 avail_mem=53.34 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.34it/s] Capturing num tokens (num_tokens=896 avail_mem=53.34 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.34it/s]Capturing num tokens (num_tokens=832 avail_mem=53.33 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.34it/s]Capturing num tokens (num_tokens=768 avail_mem=53.33 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.34it/s]Capturing num tokens (num_tokens=768 avail_mem=53.33 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.92it/s]Capturing num tokens (num_tokens=704 avail_mem=53.33 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.92it/s]Capturing num tokens (num_tokens=640 avail_mem=53.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.92it/s]Capturing num tokens (num_tokens=576 avail_mem=53.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.92it/s]Capturing num tokens (num_tokens=512 avail_mem=53.31 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.92it/s]Capturing num tokens (num_tokens=480 avail_mem=53.32 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.92it/s]

    Capturing num tokens (num_tokens=448 avail_mem=53.32 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.92it/s]Capturing num tokens (num_tokens=448 avail_mem=53.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=416 avail_mem=53.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=384 avail_mem=53.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=352 avail_mem=53.31 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=320 avail_mem=53.31 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=288 avail_mem=53.31 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=256 avail_mem=53.30 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=256 avail_mem=53.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=240 avail_mem=53.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=224 avail_mem=53.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=208 avail_mem=53.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=192 avail_mem=53.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]

    Capturing num tokens (num_tokens=176 avail_mem=53.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=160 avail_mem=53.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=160 avail_mem=53.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=144 avail_mem=53.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=128 avail_mem=53.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=112 avail_mem=53.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=96 avail_mem=53.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.76it/s] Capturing num tokens (num_tokens=80 avail_mem=53.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=64 avail_mem=53.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=64 avail_mem=53.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=48 avail_mem=53.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=32 avail_mem=53.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.01it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=24 avail_mem=53.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=20 avail_mem=53.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=20 avail_mem=53.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=16 avail_mem=53.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=12 avail_mem=53.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=8 avail_mem=53.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s] Capturing num tokens (num_tokens=4 avail_mem=53.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=4 avail_mem=53.24 GB): 100%|██████████| 58/58 [00:01<00:00, 35.94it/s]


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
    Generated text:  Frank. I'm a student in our school. I can play computer games well. I have many friends. They are all my good friends. My friend Billy is a bit shy. He likes to sit on the sofa and talk to me. My friend Tony likes to play tennis. He is good at tennis. He can play with a friend. I like to play tennis with him. My friend Sam is a bit shy. He doesn't like to talk to me. I always tell him to talk to me. My friend Lucy likes to draw pictures. She is very good at drawing. She always draws pictures for me. What do
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to become more environmentally conscious. On Earth Day, 2000 people signed a petition to increase the use of renewable energy sources. The president estimates that the renewable energy usage will increase by 50% compared to Earth Day to come. If the current usage is 10% of the total population, how many people would use renewable energy on Earth Day, 2020, assuming the percentage usage continues?
    To determine how many people would use renewable energy on Earth Day, 2020, we need to follow these steps:
    
    1. **Determine the current population on Earth Day:**
      
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    Answer this question: what country is the capital of france? The capital of France is Paris. Paris is the capital and largest city of France, and it is located in the northwestern part of the country. The country's largest city is Lyon, which is also the administrative center of the Lyon region. The cities in the northwestern part of France are Nice and Marseille, which are known for their wine and food industries. Additionally, Bordeaux is the largest city and the most populous city in the Bordeaux region. France is a unitary parliamentary republic, and Paris is the capital of France.
    Answer this question: what is the capital
    ===============================
    Prompt: The future of AI is
    Generated text:  about how much you want to trust it. We are only a few years away from a world where the vast majority of our lives will be spent living within the AI of a virtual environment. This is the world that Generation Z is currently living in, and the majority of that generation is likely to be the first to see the impact of AI on their everyday lives.
    There are several reasons why this world will be different.
    The first one is about the way the world is structured. Because the world will be full of virtual reality devices, it will be very hard to find people who have actually used them. There are going to be too many


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As a [job title], I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn new things and try new things. I'm also a great communicator and enjoy working with people from all walks of life. What do you do for a living? As a [job title], I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn new things and try new things. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also famous for its rich history, art, and culture, including the Louvre Museum, the Palace of Versailles, and the Notre-Dame Cathedral. Paris is a major transportation hub, with many major airports and train stations. It is also a popular tourist destination, with millions of visitors each year. The city is known for its fashion, food, and wine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and adaptive AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be an increased need for privacy and security measures to protect user data and prevent misuse. This could lead to the development of new privacy-preserving algorithms and techniques to ensure that AI systems are used responsibly and ethically.
    
    3. Increased focus
    


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
    Generated text:  [insert character's name] and I'm a [insert character's occupation or role]. I'm [insert character's age] years old and I'm a [insert character's personality trait or characteristic]. I'm [insert character's notable achievement or accomplishment]. I've always been [insert character's characteristic or trait] and I enjoy [insert character's hobbies, interests, or passions]. I'm [insert character's gender] and I have a [insert character's nationality or origin] background. I'm [insert character's occupation or role] and I'm passionate about [insert character's main interest or hobby]. I'm [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Paris is the largest and most populous city in France and has a rich history dating back over 5,000 years. It is located on the left bank of the Seine River, near the River Seine and the Palace of Versailles. 
    
    Paris has a diverse and diverse cultural scene, and is home to iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre Dame Cathedral, and Montmartre. The city is also famous for its fashion industry, where many of Paris' fashion houses are based. 
    
    Paris has a long-standing tradition of hospitality and has numerous museums, including the Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic, and it will continue to evolve and adapt to new challenges and opportunities. Some potential trends that could shape the future of AI include:
    
    1. Enhanced Personalization: AI will continue to improve its ability to understand and respond to individual user needs. This will enable personalized and more efficient use of resources, which could lead to increased productivity and lower costs for businesses.
    
    2. Autonomous Vehicles: AI will become more integrated into our daily lives, with autonomous vehicles becoming more common. This could lead to a significant reduction in traffic accidents and a reduction in carbon emissions, as well as increased mobility for people with disabilities.
    
    3. Better Medical Care


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

    ]

     with

     [

    Your

     Education

     Level

    ]

     degree

    .

     I

     am

     passionate

     about

     [

    Your

     Passion

    ].

     I

     am

     an

     [

    Your

     Mot

    ivation

    ]

     person

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

     knowledge

     in

     my

     field

    .

     I

     believe

     in

     the

     power

     of

     [

    Your

     Method

    ].

     I

     am

     a

     [

    Your

     Core

     Values

    ]

     person

     who

     is

     committed

     to

     making

     a

     positive

     impact

     in

     the

     world

    .

     I

     strive

     to

     be

     [

    Your

     Character

     Traits

    ]

     and

     I

     am

     always

     willing

     to

     learn

     and

     grow

    .

     I

     am

     eager

     to

     help

     others

     in

     my

     journey

    .

     Thank

     you

     for

     the

     opportunity

     to

     meet

     you

    .

     Let

    's

     create

     a

     great

     connection

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     known

     for

     its

     rich history

    ,

     vibrant

     culture

    ,

     and

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     famous

     for

     its

     annual

     celebrations

    ,

     including

     the

     Saint

    -E

    t

    ienne

     Fair

     and

     the

     Carnival

    ,

     as

     well

     as

     its

     status

     as

     the

     world

    's

     most

     populous

     city

    .

     The

     city

     is

     home

     to

     a

     diverse

     population

     and

     is

     a

     major

     economic

     and

     political

     center

     of

     France

    .

     According

     to

     the

     United

     Nations

    ,

     Paris

     is

     the

     world

    's

     

    1

    0

    th

     largest

     city

     by

     population

    .

     Additionally

    ,

     the

     city

     is

     home

     to

     many

     important

     institutions

    ,

     including

     the

     French

     Academy

    ,

     the

     French

     National

     Library

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     several

     trends

    ,

     some

     of

     which

     are

     emerging

    ,

     and

     some

     that

     are

     expected

     to

     be

     long

    -term

    :
    


    1

    .

     Increased

     automation

    :

     As

     AI

     becomes

     more

     capable

    ,

     it

     is

     expected

     to

     automate

     many

     of

     the

     tasks

     that

     humans

     currently

     perform

    ,

     such

     as

     data

     analysis

    ,

     predictive

     maintenance

    ,

     and

     customer

     service

    .

     This

     will

     increase

     efficiency

     and

     reduce

     the

     need

     for

     human

     intervention

    .
    


    2

    .

     Personal

    ization

    :

     AI

     will

     enable

     AI

    -powered

     systems

     to

     learn

     and

     adapt

     to

     users

    '

     preferences

    ,

     providing

     personalized

     experiences

     and

     recommendations

    .

     This

     will

     lead

     to

     more

     seamless

     and

     efficient

     interactions

     with

     technology

    .
    


    3

    .

     Autonomous

     vehicles

    :

     With

     advancements

     in

     AI

    ,

     autonomous

     vehicles

     (

    AV

    s

    )

     are

     expected

     to

    



```python
llm.shutdown()
```
