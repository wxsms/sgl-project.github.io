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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.20it/s]


    2026-04-06 21:23:51,247 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 21:23:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<02:53,  3.04s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:25,  2.09it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 11.23it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 11.23it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 16.84it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 16.84it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 16.84it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 16.84it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 16.84it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 16.84it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 16.84it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 16.84it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 23.07it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 23.07it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 23.07it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 23.07it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 23.07it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 23.07it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 23.07it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 27.98it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 27.98it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 27.98it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 27.98it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 27.98it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 27.98it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 27.98it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 32.83it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 32.83it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 32.83it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 32.83it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 32.83it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 32.83it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 32.83it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 36.21it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 36.21it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 36.21it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 36.21it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 36.21it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 36.21it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 36.21it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 36.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  31%|███       | 18/58 [00:00<00:01, 35.49it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 35.49it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  31%|███       | 18/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.25it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.96it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.94it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.94it/s]

    Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.94it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.94it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.94it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.94it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.78it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.78it/s]

    Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.63it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 38.61it/s]


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
    Generated text:  Riki. I am an introvert who enjoys spending time in nature and trying to escape from my everyday routine. I like to take small walks in the park or go for a walk in the woods, and I enjoy the peacefulness and beauty of the outdoors. I also enjoy reading books, listening to music, and spending time with my pet, Riki. My favorite type of music is ambient and introspective, and I like to incorporate it into my daily routine.
    I have been working on creating a meditation app that provides guided meditations and mindfulness exercises. I am working on developing a deep listening app that will allow users to listen
    ===============================
    Prompt: The president of the United States is
    Generated text:  now in the middle of a very long and difficult campaign. The campaign, which lasted for a year, has a total of 1,313,523,945,232,583,310,984,309,142,549,796,596,062,289,284,230,460,845,962,597,612,041,369,962,448 votes
    ===============================
    Prompt: The capital of France is
    Generated text:  in which province?
    A. Normandy
    B. Brittany
    C. Île-de-France
    D. Paris
    Answer:
    A
    
    As the capital of France, where is the headquarters of the French Parliament located?
    A. In the capital city
    B. In the provincial capital
    C. In the city center
    D. In the central business district
    Answer:
    A
    
    Which of the following cities is the capital of France?
    A. Paris
    B. London
    C. New York
    D. Moscow
    Answer:
    A
    
    In the capital city of France, where does the headquarters of the national court for the first
    ===============================
    Prompt: The future of AI is
    Generated text:  more interconnected than ever. And that means you and me are already connected with AI in some way. Whether it’s in the cloud or in the ground, we all have access to AI in some way, shape, or form. This has led to a new buzzword: the “AI nano-connection,” and this new connection is beginning to expand exponentially.
    In this article, we will explore the implications of this new “AI nano-connection.” We will also discuss the challenges and opportunities that lie ahead for the AI nano-connection, as well as the potential consequences of this new connectivity.
    What is the AI nano-connection?
    The


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


    Generated text:  [Name] and I'm a [occupation] who has been working in the [industry] for [number] years. I'm passionate about [reason for interest] and have always been inspired by [person or thing that inspires me]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [type of person] who is always willing to take risks and push boundaries. I'm always eager to learn and grow, and I'm always looking for ways to make a positive impact in the world. I'm a [character trait] who is always ready to help others and make a difference in the world.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich cultural heritage and a vibrant nightlife. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also home to many museums, theaters, and other cultural institutions, making it a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the urban landscape. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, there is a growing emphasis on ethical considerations and the development of AI that is transparent, accountable, and responsible. As AI becomes more integrated into our daily lives, we can expect to see a greater emphasis on privacy, security, and data protection. Overall, the future of AI is likely to be one of continued innovation, integration, and ethical considerations. 
    
    Some possible future
    


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
    Generated text:  [Your Name] and I am a [job title] with [number of years of experience] years of experience. I was born and raised in [city or country] and moved to [new city or country] when [date]. I have always loved [job title] and have always wanted to pursue a career in [specific field or industry]. I believe in the value of [profession], and have always worked hard to achieve my goals. I am passionate about [profession], and have learned a lot from my mistakes. I am excited to continue my journey and work towards my dreams. Thank you for considering me for a job!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la首都" and is the largest city in Europe by population and area. It is located on the Seine River, which flows through the city, and is home to the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral. The city is also known for its rich cultural history, including a well-preserved medieval city and its role in the French Revolution. With its modern city center and stunning views of the Seine River, Paris is a popular tourist destination and the seat of the French government.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be driven by a multitude of trends, technologies, and innovations that will shape the capabilities, ethical considerations, and social implications of AI systems. Here are some of the key trends and technologies that are likely to shape the future of AI:
    
    1. Increased Integration with Human Interaction: As AI continues to become more advanced, there will be an increase in the integration of AI with human interaction. This could involve developing more conversational AI that learns from the interactions between humans and AI systems, or developing more intelligent systems that can understand and respond to human emotions and desires.
    
    2. Enhanced Human-AI Collaboration: As AI continues to evolve,


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

    ],

     and

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    ].

     I

    've

     always

     had

     a

     passion

     for

     [

    Their

     Passion

    ],

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

    .

     I

     have

     a

     broad

     knowledge

     of

     [

    Their

     Field

    ]

     and

     enjoy

     being

     able

     to

     share

     it

     with

     others

    .

     I

    'm

     a

     [

    My

     Personality

    ],

     and

     I

    'm

     always

     up

     for

     a

     challenge

    .

     What

     is

     your

     name

    ?

     What is

     your

     occupation

    ?

     What

     is

     your

     passion

    ?

     What

     is

     your

     personality

    ?

     How

     did

     you

     get

     to

     where

     you

     are

     today

    ?

     What

     is

     your

     goal

     for

     the

     future

    ?

     I

     can

    't

     wait

     to

     meet

     you

    !

     [

    Name

    ]

     [

    What

    's

     Your

     Name

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     and

     country

    ’s

     largest

     city

     and

     the

     country

    's

     largest

     metropolitan

     area

    .


    Paris

     has

     a

     population

     of

     over

     

    2

     million

     and

     is

     the

     oldest

     capital

     city

     in

     the

     world

    .

     The

     city

     is

     home

     to

     the

     E

    iff

    el

     Tower

     and

     numerous

     museums

     and

     galleries

    .

     Paris

     is

     a

     cultural

    ,

     political

    ,

     and

     economic

     hub

     that

     attracts

     millions

     of

     visitors

     every

     year

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

     and

     wine

    .

     It

     is

     famous

     for

     its

     fashion

     industry

     and

     has

     hosted

     many

     major

     fashion

     shows

     and

     fashion

     designers

     over

     the

     years

    .

     Paris

     is

     known

     as

     "

    La

     Ville

     Fl

    ene

    "

     and

     is

     a

     popular

     tourist

     destination

    .

     It

     has

     a

     rich

     history

     dating

     back

     over

     

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     can

     be

     seen

     in

     several

     key

     trends

    :
    


    1

    .

     Increased

     automation

    :

     AI

     is

     becoming

     more

     capable

     of

     performing

     repetitive

     and

     mundane

     tasks

    ,

     and

     it

     is

     likely

     to

     automate

     more

     of

     these

     tasks

     in

     the

     future

    .

     This

     will

     lead

     to

     a

     significant

     increase

     in

     efficiency

     and

     productivity

    ,

     as

     well

     as

     a

     reduction

     in

     human

     errors

    .
    


    2

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

     big

     data

    ,

     machine

     learning

    ,

     and

     blockchain

    .

     This

     integration

     will

     allow

     AI

     to

     work

     more

     closely

     with

     other

     systems

     and

     to

     provide

     a

     more

     accurate

     and

     reliable

     output

    .
    
    3

    .

     AI

     ethics

     and

     safety

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

    



```python
llm.shutdown()
```
