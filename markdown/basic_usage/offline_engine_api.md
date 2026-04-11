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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.04it/s]


    2026-04-11 07:20:40,809 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 07:20:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 11.32it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 11.32it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 23.23it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 28.04it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 28.04it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 28.04it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 28.04it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 28.04it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 28.04it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 28.04it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 32.79it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 32.79it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 37.21it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 37.21it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 37.21it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 37.21it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 37.21it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 37.21it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 37.21it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 37.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=135.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=126.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=126.39 GB):   3%|▎         | 2/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:03, 15.78it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.02 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.39it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=121.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.98 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.98 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.97 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.95 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]Capturing num tokens (num_tokens=960 avail_mem=120.96 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s] Capturing num tokens (num_tokens=896 avail_mem=120.96 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]Capturing num tokens (num_tokens=896 avail_mem=120.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=832 avail_mem=120.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=768 avail_mem=120.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=704 avail_mem=120.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=640 avail_mem=120.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=576 avail_mem=120.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=576 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=512 avail_mem=120.94 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=480 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.46it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=416 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=384 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=384 avail_mem=120.95 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=352 avail_mem=120.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=320 avail_mem=120.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=288 avail_mem=120.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=256 avail_mem=120.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=240 avail_mem=120.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=240 avail_mem=120.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=224 avail_mem=120.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=208 avail_mem=120.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.97it/s]

    Capturing num tokens (num_tokens=192 avail_mem=120.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=176 avail_mem=120.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=160 avail_mem=120.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=160 avail_mem=120.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=144 avail_mem=120.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=128 avail_mem=120.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=112 avail_mem=120.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=96 avail_mem=120.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.82it/s] Capturing num tokens (num_tokens=80 avail_mem=120.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=80 avail_mem=120.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=64 avail_mem=120.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=48 avail_mem=120.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.10it/s]

    Capturing num tokens (num_tokens=32 avail_mem=120.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=28 avail_mem=120.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=24 avail_mem=120.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=24 avail_mem=120.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=20 avail_mem=120.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=16 avail_mem=120.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=12 avail_mem=120.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=8 avail_mem=120.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.40it/s] Capturing num tokens (num_tokens=4 avail_mem=120.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=4 avail_mem=120.87 GB): 100%|██████████| 58/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=4 avail_mem=120.87 GB): 100%|██████████| 58/58 [00:01<00:00, 39.21it/s]


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
    Generated text:  Sam. I was born in England in 1987. Now I am living in a country called China. I love reading books and watching movies in English. I also like to listen to the music and I'm good at singing. I have been to many places in China. I'm interested in Chinese food and I'm good at cooking. I have lots of friends. I have a big house, my favorite pet is a cat, and I have a good relationship with my parents. The people of China are very friendly and polite. They always say "hello" to each other. The weather in China is beautiful every day
    ===============================
    Prompt: The president of the United States is
    Generated text:  a well-known leader and often takes public statements, but when a new president is sworn in, the outgoing president often gives a speech as a form of farewell. In 1989, John F. Kennedy gave a speech that was meant to be a farewell to a new president, but it was in fact a farewell to a different political party. He gave his speech in Washington, D. C. In 1991, he was re-elected, and Kennedy was not re-elected as the winner of the Republican Party primary. In 2004, he lost his re-election bid to George W. Bush,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and its largest urban area is the Greater Paris Region, also known as the Paris metropolitan area. The Paris metropolitan area includes the following cities::
    
    - 60% of the city population live in the 12 boroughs of Paris, which are on the outskirts of the city centre.
    - 15% of the city population live in the 18% of the city which is contained in the Grand Paris region.
    - The remaining 30% of the city population live in the 10% of the city that is contained in the Île-de-France.
    - The Île-de-France
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and we have to be more cautious. It is also inevitable that many people will choose to use AI to do things which they normally wouldn't do. For example, these include using voice assistants for banking or running car programs. Many people will choose to use AI to help them in other aspects of their life such as making better decisions, improving the quality of life, reducing the cost of healthcare, and reducing the risk of accidents. AI has the potential to help us do many things that we normally wouldn't be able to do. However, there are also concerns about the ethical implications and potential negative effects of AI. The key to addressing


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I'm [gender]. I have [number] years of experience in [industry], and I'm always looking for ways to [describe your professional goal or passion]. I'm a [type of person] and I enjoy [describe your hobbies or interests]. I'm [positive] and [positive]. I'm [positive] and [positive]. I'm [positive] and [positive]. I'm [positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. It is also the birthplace of the French Revolution and the home of the French language. Paris is a bustling metropolis with a rich history and a vibrant culture that attracts millions of visitors each year. Its status as the world's most populous city is due to its large population and the city's importance in French culture and politics. The city is also home to many famous landmarks and museums, including the Louvre and the Musée d'Orsay. Paris is a city that has played a significant role in shaping French culture and identity
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Enhanced natural language processing: AI is already capable of understanding and generating human language, but as the technology advances, it is likely to become even more sophisticated, allowing for more natural and intuitive interactions with humans.
    
    3. Improved privacy and security: As AI systems become more complex and sophisticated, there is a risk of privacy and security breaches. As a result, there is likely to be increased focus
    


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
    Generated text:  [insert character's name], and I am [insert character's profession or role in the story]. In my spare time, I enjoy [insert an activity that interests me]. I believe in the importance of [insert something, such as giving back, kindness, or a cause that is important to you], and I strive to make a positive impact in the world. I am constantly learning, seeking out new experiences and ideas, and always looking for ways to grow and improve. I am a true advocate for [insert something, such as making a difference in the community, being an equal player in a game, or simply doing the right thing
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historical and cultural center, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre Dame Cathedral. The city is also known for its beautiful gardens, including the Seine River and the Palais des Nations. Paris is home to many world-renowned museums, including the Louvre, the Musée d'Orsay, and the Musée d'Art Moderne. The city is also known for its cuisine, which includes French, Italian, and German dishes. Paris is a major transportation hub, with a well-developed public transportation system and many famous landmarks and attractions within
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  complex and uncertain, and it's hard to predict exactly where it will lead. However, here are some possible trends that some experts believe will shape the future of AI:
    
    1. Increased emphasis on ethical considerations: As more AI systems become increasingly complex and autonomous, it's becoming increasingly important to consider how these systems will impact society. This could lead to a shift towards more ethical and responsible design, as well as greater focus on the potential negative consequences of AI.
    
    2. More integration with human decision-making: As AI systems become more sophisticated, it's possible that they will be more integrated with human decision-making processes. This could lead to a


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

    First

     Name

    ]

     and

     I

     am

     [

    Last

     Name

    ].

     I

     am

     a

     [

    insert

     occupation

    ]

     who

     has

     always

     been

     [

    insert

     something

     about

     your

     personality

     or

     interests

    ].

     I

     love

     [

    insert

     something

     about

     your

     hobbies

     or

     passions

    ].

     What

    's

     your

     name

    ,

     and

     what

     do

     you

     do

    ?

     -

     I

    'm

     [

    insert

     your

     name

    ]

     and

     I

    'm

     a

     [

    insert

     your

     profession

    /

    occupation

    ].

     I

     love [

    insert

     something

     that

    's

     a

     reason

     you

     enjoy

     what

     you

     do

    ].

     What

    's

     your

     name

    ,

     and

     what

     do

     you

     do

    ?

     -

     My

     name

     is

     [

    insert

     your

     name

    ],

     and

     I

    'm

     a

     [

    insert

     your

     profession

    /

    occupation

    ].

     I

     love

     [

    insert

     something

     about

     your

     hobbies

     or

     interests

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

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

     Its

     historical

     and

     cultural

     significance

     is

     recognized

     throughout

     the

     world

    .

     A

     visit

     to

     Paris

     offers

     a

     glimpse

     into

     French

     culture

    ,

     cuisine

    ,

     and

     a

     unique

     blend

     of

     modern

    ity

     and

     ancient

     history

    .

     French

     culture

     can

     be

     found

     in

     its

     language

    ,

     art

    ,

     music

    ,

     and

     cuisine

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     its vibrant

     streets

    ,

     colorful

     architecture

    ,

     and

     diverse

     neighborhoods

    .

     The

     city

     offers

     a

     range

     of

     attractions

    ,

     from

     historical

     sites

     to

     modern

     art

     galleries

    ,

     and

     visitors

     can

     immer

    se

     themselves

     in

     French

     culture

     and

     history

    .

     The

     city

     is

     also

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     expand

     and

     evolve

    , driven

     by

     a

     variety

     of

     trends

     and

     changes

     in society

    ,

     technology

    ,

     and

     the

     economy

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     Human

    -A

    I

     Interaction

    :

     AI

     is

     likely

     to

     become

     more

     closely

     integrated

     with

     human

     culture

     and

     society

    ,

     and

     humans

     may

     interact

     with

     AI

     systems

     on

     a

     more

     regular

     basis

    .

     This

     could

     lead

     to

     new

     forms

     of

     AI

    -driven

     human

     interaction

    ,

     such

     as

     virtual

     assistants

     or

     personalized

     assistants

     that can

     better

     understand

     and

     respond

     to

     human

     needs.
    


    2

    .

     More

     Privacy

    -

    Res

    pecting

     AI

    :

     As

     AI

     systems become

     more powerful

     and

     capable

    ,

     there

     may

     be increasing

     concerns

     about their

     impact

    



```python
llm.shutdown()
```
