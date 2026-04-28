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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.33it/s]


    2026-04-28 21:12:40,881 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 21:12:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]

    Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:04,  9.42it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:04,  9.42it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:04,  9.42it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:02, 13.23it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:02, 13.23it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:02, 13.23it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:02, 13.23it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:02, 13.23it/s]

    Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:02, 13.23it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:02, 13.23it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 18.40it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 18.40it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 18.40it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 18.40it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 18.40it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 18.40it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 18.40it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 18.40it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 25.31it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 25.31it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 25.31it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 25.31it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 25.31it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 25.31it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 25.31it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 25.31it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 32.64it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 32.64it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 32.64it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 32.64it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 32.64it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 32.64it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 32.64it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 32.64it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 39.21it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 39.21it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 39.21it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 39.21it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 39.21it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 39.21it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 39.21it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 39.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.18 GB):   3%|▎         | 2/58 [00:00<00:08,  6.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.20 GB):   3%|▎         | 2/58 [00:00<00:08,  6.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.19 GB):   3%|▎         | 2/58 [00:00<00:08,  6.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.19 GB):   7%|▋         | 4/58 [00:00<00:05, 10.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.19 GB):   7%|▋         | 4/58 [00:00<00:05, 10.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.19 GB):   7%|▋         | 4/58 [00:00<00:05, 10.22it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=57.19 GB):  10%|█         | 6/58 [00:00<00:03, 13.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.17 GB):  10%|█         | 6/58 [00:00<00:03, 13.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.16 GB):  10%|█         | 6/58 [00:00<00:03, 13.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.16 GB):  10%|█         | 6/58 [00:00<00:03, 13.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.16 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.12 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.79it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.12 GB):  21%|██        | 12/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.13 GB):  21%|██        | 12/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.12 GB):  21%|██        | 12/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.12 GB):  21%|██        | 12/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.11 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.11 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.10 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.09 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.94it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=57.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.06 GB):  33%|███▎      | 19/58 [00:01<00:01, 26.48it/s]Capturing num tokens (num_tokens=960 avail_mem=57.07 GB):  33%|███▎      | 19/58 [00:01<00:01, 26.48it/s] Capturing num tokens (num_tokens=896 avail_mem=57.07 GB):  33%|███▎      | 19/58 [00:01<00:01, 26.48it/s]Capturing num tokens (num_tokens=832 avail_mem=57.07 GB):  33%|███▎      | 19/58 [00:01<00:01, 26.48it/s]Capturing num tokens (num_tokens=832 avail_mem=57.07 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.40it/s]Capturing num tokens (num_tokens=768 avail_mem=57.06 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.40it/s]Capturing num tokens (num_tokens=704 avail_mem=57.06 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.40it/s]Capturing num tokens (num_tokens=640 avail_mem=57.06 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.40it/s]Capturing num tokens (num_tokens=576 avail_mem=57.06 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.40it/s]Capturing num tokens (num_tokens=512 avail_mem=57.04 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.40it/s]

    Capturing num tokens (num_tokens=480 avail_mem=57.06 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.40it/s]Capturing num tokens (num_tokens=480 avail_mem=57.06 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=448 avail_mem=57.06 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=416 avail_mem=57.05 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=384 avail_mem=57.05 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=352 avail_mem=57.05 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=320 avail_mem=57.04 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=320 avail_mem=57.04 GB):  60%|██████    | 35/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=288 avail_mem=57.04 GB):  60%|██████    | 35/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=256 avail_mem=57.04 GB):  60%|██████    | 35/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=240 avail_mem=57.03 GB):  60%|██████    | 35/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=224 avail_mem=57.03 GB):  60%|██████    | 35/58 [00:01<00:00, 40.69it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.02 GB):  60%|██████    | 35/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=192 avail_mem=57.02 GB):  60%|██████    | 35/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=192 avail_mem=57.02 GB):  71%|███████   | 41/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=176 avail_mem=57.02 GB):  71%|███████   | 41/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=160 avail_mem=57.02 GB):  71%|███████   | 41/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=144 avail_mem=57.01 GB):  71%|███████   | 41/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=128 avail_mem=57.01 GB):  71%|███████   | 41/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=112 avail_mem=57.01 GB):  71%|███████   | 41/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=96 avail_mem=57.01 GB):  71%|███████   | 41/58 [00:01<00:00, 44.64it/s] Capturing num tokens (num_tokens=96 avail_mem=57.01 GB):  81%|████████  | 47/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=80 avail_mem=57.00 GB):  81%|████████  | 47/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=64 avail_mem=57.00 GB):  81%|████████  | 47/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=48 avail_mem=57.00 GB):  81%|████████  | 47/58 [00:01<00:00, 46.91it/s]

    Capturing num tokens (num_tokens=32 avail_mem=56.99 GB):  81%|████████  | 47/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=28 avail_mem=56.99 GB):  81%|████████  | 47/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=24 avail_mem=56.99 GB):  81%|████████  | 47/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=24 avail_mem=56.99 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=20 avail_mem=56.98 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=16 avail_mem=56.98 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=12 avail_mem=56.98 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=8 avail_mem=56.97 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.35it/s] Capturing num tokens (num_tokens=4 avail_mem=56.97 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=4 avail_mem=56.97 GB): 100%|██████████| 58/58 [00:01<00:00, 33.13it/s]


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
    Generated text:  Sophie and I'm currently a student at the University of Sydney. I studied mathematics and neuroscience at the University of Sydney. I love to travel, read books and watch movies. In my spare time, I like to explore the city and try new foods.
    
    ## What are my majors and minors at the University of Sydney?
    
    • Mathematics
    • Neuroscience
    
    ## What's your first online course? What subject was it?
    
    I took the course Introduction to Statistics. This course is based on the open-source text by David Freedman. It was the first time I really took an online course. I had problems with the course and I took one more
    ===============================
    Prompt: The president of the United States is
    Generated text:  considered the most powerful person in the world. The country's central bank is considered the world's most powerful organization. Many important governmental agencies have been established in the United States to provide oversight of the country's economy and politics. To make things worse, many corporations have been formed in the United States in the past 50 years, which have expanded the scope of government. One of the main ways in which the U.S. government has become more powerful is through the use of the Federal Reserve. The Federal Reserve is the U.S. central bank, and it has been responsible for many of the decisions that have been made in the past
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. Lille C. Calais D. Lyon
    A. Paris
    答案: A
    
    自体移植的指征不包括
    A. 无明显的死骨、死腔或死髓
    B. 皮肤有感染，但无感染性脓肿
    C. 心脏、肺等大血管器官手术切口
    D. 可以发现穿刺物
    E. 皮肤、粘膜、浅筋膜等部位的浅、深裂隙性损伤
    答案: C
    
    以下关于证券市场线的描述正确的是____。
    A. 证券
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of humans, but there is one piece of technology that will be central to the future of AI. The future of AI is in the hands of humans, but there is one piece of technology that will be central to the future of AI. What is it?
    The answer to this question is likely to be a machine learning algorithm, such as a deep neural network, that has been trained on vast amounts of data to recognize patterns and make predictions. These algorithms are designed to learn from a vast amount of data, and they are able to adapt to new data and improve their performance over time. They are also capable of handling complex


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many of the world's most famous museums and attractions. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. Its status as the world's most populous city is due in part to its status as the capital of France. The city is also known for its cuisine, fashion, and music, and is a popular tourist destination. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This could lead to the development of new AI systems that are designed to be more transparent, accountable, and responsible.
    
    2. Integration with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. As these technologies continue to evolve, it is likely that AI will be integrated with other technologies to create
    


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
    Generated text:  [Name], and I am a [age] year old [character]. I am an artist, but I have never been a professional. However, I have been the subject of many projects, and I love to make people feel happy and inspired by my art. I have a love for music and I like to create music videos. I am passionate about my art and I believe that it is my job to make people feel good. I have a love for technology and I use it to create my art. I am an artist, but I have never been a professional. However, I have been the subject of many projects, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and the second-largest city in the world by population. Paris is known for its rich history, art, and food scene. The city is home to landmarks such as the Eiffel Tower and the Louvre Museum, as well as numerous museums and cultural institutions. Paris is also famous for its fashion industry and the iconic Eiffel Tower, which stands as a symbol of the city's city spirit. The city is also home to the French Parliament and many other government buildings. As of 2021, Paris had a population of approximately 2.3 million people. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, and there are many potential areas where it will play a key role in shaping society. Here are some possible future trends in AI:
    
    1. Increased accuracy and efficiency: As AI technology continues to improve, we can expect to see more accurate and efficient applications of AI in various fields. This could include self-driving cars, chatbots, virtual assistants, and more.
    
    2. Enhanced personalization: AI will also become more personal, allowing for more tailored experiences and recommendations based on individual needs and preferences. This could help address issues of social inequality and provide more personalized support and services.
    
    3. Integration with humans: AI will continue to gain


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

     Sarah

    .

     I

     am

     a

     talented

     musician

     who

     enjoys

     playing

     the

     piano

     and

     performing

     in

     groups

    .

     I

     have

     a

     strong

     work

     ethic

     and

     I

     strive

     to

     improve

     my

     skills

     continuously

    .

     I

     love

     to

     travel

     and

     explore

     new

     places

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     and

     expand

     my

     hor

    izons

    .

     I

     love

     supporting

     musicians

     and

     helping

     them

     achieve

     their

     goals

    .

     I

     also

     enjoy

     reading

     and

     writing

    ,

     and

     I

    'm

     always

     looking

     for

     new

     writing

     projects

    .

     Thank

     you

     for

     considering

     my

     profile

    .

     Is

     there

     anything

     else

     I

     should

     know

     about

     myself

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     a

     physical

     presence

    ,

     so

     I

    'm

     not

     capable

     of

     interacting

     with

     humans

    .

     However

    ,

     I

     can

     assist

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historical

     monuments

    ,

     gastr

    onomy

    ,

     and

     cultural

     heritage

    .

     Paris

     is

     also

     a

     major

     transportation

     hub

     and

     is

     home

     to

     the

     French

     Parliament

     and

     the

     E

    iff

    el

     Tower

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     the

     fifth

    -largest

     city

     in

     the

     world

     by

     population

    .

     
    


    Please

     provide

     the

     French

     word

     for

     "

    E

    iff

    el

     Tower

    ".

     
    


    "

    La

     Tour

     E

    iff

    el

    "

     or

     "

    La

     Tour

     E

    iff

    el

     de

     Paris

    "

     
    


    Paris

     is

     a

     city

     with

     a

     rich

     and

     diverse

     history

    ,

     including

     the

     presence

     of

     numerous

     UNESCO

     World

     Heritage

     Sites

    .

     It

     is

     also

     known

     for

     its

     international

     cuisine

    ,

     nightlife

    ,

     and

     cultural

     activities

    .

     Paris

     is

     a

     popular

     tourist

     destination

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

    .

     With

     continued

     advancements

     in

     hardware

     and

     software

    ,

     AI

     is

     expected

     to

     continue

     to

     advance

     and

     become

     more

     integrated

     into

     our

     daily

     lives

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Integration

     of

     AI

     into

     everyday

     technology

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     is

     likely

     to

     be

     integrated

     into

     more

     and

     more

     everyday

     technologies

    ,

     such

     as

     smart

     homes

    ,

     self

    -driving

     cars

    ,

     and

     voice

     assistants

     like

     Siri

     and

     Alexa

    .
    


    2

    .

     AI

     will

     be

     used

     for

     personal

    izing

     healthcare

    :

     As

     AI

     can

     analyze

     large

     amounts

     of

     data

    ,

     it

     will

     be

     able

     to

     provide

     personalized

     healthcare

     recommendations

     to

     patients

    .

     This

     will

     be

     especially

     useful

     in

     treating

     diseases

     that

     are

     currently

     difficult

     to

     diagnose

    ,

     such

    



```python
llm.shutdown()
```
