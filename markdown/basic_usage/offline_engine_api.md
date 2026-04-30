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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]


    2026-04-30 23:37:26,031 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 23:37:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:41,  4.94s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:41,  4.94s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:05<00:04,  7.86it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.09it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.79 GB):  21%|██        | 12/58 [00:00<00:01, 28.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.77 GB):  21%|██        | 12/58 [00:00<00:01, 28.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.77 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.07 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.79it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=75.07 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.07 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=960 avail_mem=75.07 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.51it/s] Capturing num tokens (num_tokens=896 avail_mem=75.06 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=832 avail_mem=75.06 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=768 avail_mem=75.06 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=704 avail_mem=75.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=704 avail_mem=75.05 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.13it/s]Capturing num tokens (num_tokens=640 avail_mem=75.05 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.13it/s]Capturing num tokens (num_tokens=576 avail_mem=75.05 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.13it/s]Capturing num tokens (num_tokens=512 avail_mem=75.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.13it/s]Capturing num tokens (num_tokens=480 avail_mem=75.05 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.13it/s]

    Capturing num tokens (num_tokens=448 avail_mem=75.05 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.13it/s]Capturing num tokens (num_tokens=416 avail_mem=75.05 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.13it/s]Capturing num tokens (num_tokens=416 avail_mem=75.05 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=384 avail_mem=75.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=352 avail_mem=75.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=320 avail_mem=75.03 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.51it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.95it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.95it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.95it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.95it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.95it/s]Capturing num tokens (num_tokens=20 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.95it/s]Capturing num tokens (num_tokens=16 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.95it/s]Capturing num tokens (num_tokens=16 avail_mem=74.55 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.91it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.91it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.91it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.91it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 40.45it/s]


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
    Generated text:  Angela and I have been working as a teaching assistant for the past 15 years. I have been teaching and tutoring all around the world and I have been working as a tutor at a local tutoring company, as well as working as a school teacher, but I have never tutored a college level class. I have been a teacher for 33 years, but I am currently a tutor for the last 15 years.
    As a student I remember being overwhelmed by my own academic goals and the amount of work I was doing. I would feel like I was constantly falling behind and not achieving what I wanted to do. I know
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ____ position.
    A. ceremonial
    B. political
    C. symbolic
    D. ceremonial and symbolic
    Answer:
    B
    
    When the teacher introduces the theme of 'passing down traditional culture through the Internet' to the students, which of the following is an appropriate instructional strategy? 
    A. Implement a multimedia teaching plan
    B. Organize discussion activities
    C. Use case studies
    D. Share video clips
    Answer:
    A
    
    The type of data type that can hold integers with decimal places is ______. ( )
    A. Integer
    B. Long
    C. Double
    D. Float
    Answer:
    D
    
    The following
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. According to the direction of the river, the city is in the south. From the other side, it is on the north.
    What are the name of the city? Paris. The capital of France is Paris, France. It is the largest city in the country and the most visited tourist destination in Europe. The city was founded by the Ancient Romans in 509 BCE and is located on the Seine River, which flows through the city. Paris is home to many iconic landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower, which was originally built as a flying station for
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is also rapidly changing. We are now in a “new normal” of rapid technological change. The impact of such change on the landscape of work is profound and multifaceted. Here are a few points to consider:
    
      1. The speed and scale of change is unprecedented. New technologies are evolving at an exponential rate, and the pace of change is accelerating. The need for adaptability and agility is growing.
      2. AI and automation are changing the way we work, share information, and solve problems. AI is revolutionizing decision-making and problem-solving, while automation is making the job market more efficient


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is the largest city in France by population. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its cuisine, fashion, and art scene. The city is home to many world-renowned museums, including the Louvre and the Musée d'Orsay. It is also known for its annual festivals and events, such as the Eiffel Tower Parade and the Carnaval. Paris is a vibrant and diverse city with a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI systems in the public eye.
    
    3. Increased use of AI in healthcare
    


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
    Generated text:  [insert character's name], and I'm here to share my journey of discovery and growth. I'm a curious individual who loves to learn and challenge myself. I'm never afraid to ask questions and seek knowledge. My passion for knowledge and education is infectious, and I love sharing my insights with others. I have a strong work ethic, and I'm always striving to improve myself. I'm also passionate about volunteering and contributing to my community in various ways. I'm a self-reliant and resilient person who thrives in a fast-paced, dynamic environment. I'm constantly learning and growing, and I'm excited to continue doing so
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the North Eastern region of the country.
    
    That's correct! Paris is the capital of France, located in the heart of the Paris region, just east of the Seine river, and served as the capital of France from 1804 to 1969. It is the largest city in the European Union and is the 11th most populous city in the world. The city is known for its rich history, culture, and modern architecture. Paris is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, Notre Dame Cathedral, and Marais district. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and diverse, with many potential developments shaping its direction. Here are some possible trends to watch:
    
    1. Greater integration of AI into human decision-making: AI is already being used in complex decision-making processes, but more integration with human decision-making is likely to occur. AI-powered tools will become more intuitive, allowing humans to more easily understand complex data and make informed decisions.
    
    2. Enhanced autonomy: Autonomous AI will continue to evolve, with AI systems becoming more capable of performing tasks that were previously done by humans. This could lead to more autonomous vehicles, robots, and other autonomous systems becoming increasingly common.
    
    3. AI in healthcare: AI


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

    职业

    ]

     with

     a

     passion

     for

     [

    field

     of

     interest

    ].

     I

     love

     [

    reason

     for

     passion

    ].

     Whether

     it

    's

     writing

    ,

     photography

    ,

     or

     whatever

     else

    ,

     I

    'm

     always

     excited

     to

     share

     my

     creativity

     with

     you

    .

     I

    'm

     currently

     [

    age

    ]

     years

     old

    ,

     and

     I

     enjoy

     [

    what

     I

     enjoy

     doing

    ]

     with

     my

     family

     and

     friends

    .

     I

    've

     always

     been

     a

     [

    what

     brings

     me

     happiness

    ]

     person

    ,

     and

     I

     try

     to

     make

     the

     most

     of

     every

     moment

    .

     I

    'm

     looking

     forward

     to

     the

     day

     when

     I

     can

     bring

     my

     own

     [

    dream

    ]

     to

     life

    .

     What

     a

     wonderful

     introduction

    !

     What

    's

     your

     favorite

     hobby

     or

     activity

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     also

     known

     as

     the

     City

     of

     Light

    .

     
    


    -

     **

    Paris

     City

     Hall

    **:

     Paris

    's

     main

     city

     hall

    ,

     known

     for

     its

     elegant

     architecture

     and

     rich

     history

    .


    -

     **

    E

    iff

    el

     Tower

    **:

     The

     iconic

     monument

     that

     rises

     above

     the

     city

     and

     is

     one

     of

     the

     most

     famous

     landmarks

     in

     the

     world

    .


    -

     **

    T

    ours

    **:

     The

     most

     visited

     tourist

     attraction

     in

     France

    ,

     featuring

     famous

     sites

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

     and

     Notre

    -D

    ame

     Cathedral

    .


    -

     **

    Tour

    ism

    **:

     Paris

     is

     a

     major

     tourist

     destination

    ,

     known

     for

     its

     vibrant

     nightlife

    ,

     museums

    ,

     and

     shopping

    .

     It

     is

     home

     to

     numerous

     world

    -ren

    owned

     attractions

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

    ,

     with

     potential

     to

     revolution

    ize

     many

     different

     fields

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     into

     a

     variety

     of

     fields

    ,

     including

     health

     care

    ,

     transportation

    ,

     finance

    ,

     and

     entertainment

    .

     We

     can

     expect

     to

     see

     even

     more

     integration

     in

     the

     coming

     years

    ,

     with

     AI

     playing

     a

     more

     significant

     role

     in

     these

     areas

    .
    


    2

    .

     Growth

     of

     the

     AI

     workforce

    :

     As

     AI

     technology

     becomes

     more

     complex

     and

     advanced

    ,

     the

     demand

     for

     AI

     professionals

     will

     increase

    .

     This

     could

     result

     in

     an

     increase

     in

     the

     number

     of

     AI

     workers

    ,

     who

     could

     work

     alongside

     humans

     in

     their

     jobs

    .
    


    3

    .

    



```python
llm.shutdown()
```
