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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.96it/s]


    2026-04-14 00:15:14,543 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 00:15:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:03<00:03, 12.67it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:03<00:01, 20.86it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:03<00:00, 29.72it/s]

    Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 47.38it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 47.38it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 47.38it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 47.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 35.67it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 35.67it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=512 avail_mem=120.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.68it/s]

    Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.10it/s]

    Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.81it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.99it/s]

    Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.27it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 44.86it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 39.69it/s]


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
    Generated text:  Amelie, I am a French software engineer. I have a background in computer science and have a passion for exploring the applications of programming in real-world situations. I am also fluent in English and I have a strong understanding of natural language processing and natural language generation. 
    Can you please provide me with more information on how to use Python in the field of natural language processing? 
    Sure, I would be happy to provide you with more information on how to use Python in the field of natural language processing. Can you give me an overview of the main tools and libraries that are commonly used in this field, and how they are used in
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected by _____ of the citizens of the United States. ____
    A. one
    B. two
    C. three
    D. four
    Answer:
    C
    
    According to the standards of the National Security Law, in the event of a state of national emergency, the National People's Congress may decide to exercise the power of _____ on behalf of the state. ____
    A. Arrest
    B. Seize the property of state officials
    C. Seize state assets
    D. Seize the assets of state officials
    Answer:
    D
    
    According to the "Regulations on the Reporting, Investigation, and Handling of Production Safety Accidents
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located in the central region of the country. It is the cultural and intellectual center of the world. Paris is one of the most important cities in the world.
    Paris is a big city, a place of many beautiful streets, beautiful buildings, delicious food, and of course, the city of love.
    The City of Light
    Paris was founded as a Roman town in 701 AD. It is located on a hilltop, overlooking the Seine River. When it was founded, it was called "Paris". The French word "Paris" means "the city of light". The name was chosen as the city
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. What does the future of AI look like for the future of humanity? A 36th birthday celebration event was held recently in the Bochum campus of the University of Applied Sciences Bochum. The celebration featured a symphony of 1000 musicians, including a soloist by the composer Wolfgang Amadeus Mozart, and a 1000 year old timepiece, made by a local blacksmith. This event was organized by the Bochum University of Applied Sciences and was in cooperation with the Bauhaus Institute of Europe, the International Organised Music Association, the Academy of Art and Design of


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


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France, and it is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a major cultural and economic center in Europe, and it is home to many famous museums, theaters, and restaurants. The city is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. Overall, Paris is a vibrant and exciting city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, healthcare, finance, and transportation. Automation will likely lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for AI-driven innovation.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. There will be a need for
    


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
    Generated text:  [Name], and I'm a/an [occupation] who has lived [number] years in [place]. I'm a/an [job title] and [job title] and [job title], respectively. I'm passionate about [reason for being passionate about it], and I've always been passionate about [reason for being passionate about it]. I love [reason for being passionate about it] because [explanation]. I've always been driven by [reason for being driven by it] and I believe that [reason for being passionate about it] is my greatest strength. I also have a good sense of humor and love to laugh,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "The City of Light" and "The Garden of the Sun." It is located on the banks of the Seine River in the south-central region of France, and is the third-largest city in the European Union. Paris is a cosmopolitan city with a rich history, and is home to many of the world's most famous art, music, and cultural institutions. The city is also a significant economic and political center, and is one of the most popular tourist destinations in the world. Paris is home to many cultural landmarks, including the Louvre Museum, the Notre-Dame Cathedral, and the Eiffel
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  one of rapid evolution, with new trends emerging almost every day. Here are some of the potential areas of AI development we can expect to see in the near and long term:
    
    1. Improved Natural Language Processing: One of the most promising areas of AI development is natural language processing, which is the ability of AI systems to understand and generate human language. This includes areas like machine translation, language translation, sentiment analysis, and more. With advances in NLP, we can expect to see more interactive and conversational AI systems that can understand and respond to human language in natural ways.
    
    2. Enhanced Image and Video Recognition: Another area of AI


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

     am

     a

    /an

     [

    job

     title

    ].

     I

     have

     always

     been

     a

     hard

    working

    ,

     determined

     individual

     who

     is

     always

     ready

     to

     take

     on

     new

     challenges

    .

     I

     am

     passionate

     about

     learning

     new

     things

     and

     always

     seeking

     knowledge

    ,

     and

     I

     enjoy

     helping

     others

     to

     achieve

     their

     goals

    .

     I

     am

     a

     natural

     leader

     and

     am

     always

     ready

     to

     lead

    ,

     inspire

     and

     mentor

     others

    .

     I

     am

     a

     team

     player

     and

     always

     put

     others

     before

     myself

    .

     I

     am

     a

     true

     friend

     and

     always

     ready

     to

     connect

     with

     others

    .

     I

     am

     very

     compassionate

     and

     empath

    etic

    ,

     and

     I

     always

     try

     to

     treat

     others

     the

     way

     I

     would

     want

     to

     be

     treated

    .

     I

     am

     confident

     in

     my

     abilities

     and

     always

    
    
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

     and

     beautiful

     gardens

    .

     The

     city

     is

     home

     to

     many

     notable

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     The

     French

     language

     is

     spoken

     by

     millions

     of

     people

     in

     Paris

    ,

     while

     the

     capital

     is

     also

     known

     for

     its

     rich

     cultural

     history

     and

     diverse

     population

    .

     The

     city

     has

     a

     vibrant

     nightlife

     scene

    ,

     with

     many

     bars

    ,

     clubs

    ,

     and

     restaurants

    ,

     and

     is

     a

     popular

     tourist

     destination

     for

     many

     visitors

    .

     Paris

     has

     a

     rich

     history

    ,

     including

     the

     influence

     of

     the

     French

     Revolution

     and

     Napoleon

     Bon

    ap

    arte

    ,

     and

     is

     a

     popular

     destination

     for

     art

     enthusiasts

     and

     history

     buffs

    .

     Overall

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     wide

     range

     of

     trends

     and

     developments

    ,

     including

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     continues

     to

     become

     more

     advanced

    ,

     it

     is

     likely

     to

     be

     integrated

     with

     other

     technologies

     such

     as

     blockchain

    ,

     quantum

     computing

    ,

     and

     cyber

    security

     to

     create

     even

     more

     sophisticated

     and

     powerful

     AI

     systems

    .
    


    2

    .

     Expansion

     of

     the

     scope

     of

     AI

     applications

    :

     With

     the

     increasing

     availability

     of

     data

     and

     the

     ability

     to

     analyze

     it

     in

     new

     ways

    ,

     AI

     is

     likely

     to

     be

     used

     in

     a

     wider

     range

     of

     applications

    ,

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     security

    ,

     to

     create

     even

     more

     intelligent

     and

     effective

     solutions

    .
    


    3

    .

     Greater

     transparency

     and

     accountability

     in

     AI

     systems

    :

     As

     AI

     becomes

    



```python
llm.shutdown()
```
