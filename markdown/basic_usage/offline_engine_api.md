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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.49it/s]


    2026-04-13 14:09:33,514 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 14:09:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.96it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.96it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.96it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.96it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.96it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.96it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.30it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.30it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.30it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.30it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.30it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.30it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.30it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:09,  5.30it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s] 

    Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:02<00:03, 11.09it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:02<00:01, 18.74it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:02<00:01, 18.74it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:02<00:01, 18.74it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 18.74it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 18.74it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 18.74it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 18.74it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:03<00:01, 18.74it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 25.70it/s]

    Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s]

    Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 40.67it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 48.59it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 48.59it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 48.59it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 48.59it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 48.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.96 GB):   3%|▎         | 2/58 [00:00<00:03, 18.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.96 GB):   3%|▎         | 2/58 [00:00<00:03, 18.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.95 GB):   3%|▎         | 2/58 [00:00<00:03, 18.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.96 GB):   3%|▎         | 2/58 [00:00<00:03, 18.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.96 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.95 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.95 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.95 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.95 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.95 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.94 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.94 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.94 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.93 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=56.93 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.93 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.93 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.92 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.92 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.92 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.91 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.91 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.91 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.89 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=960 avail_mem=56.90 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.20it/s] Capturing num tokens (num_tokens=896 avail_mem=56.90 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=832 avail_mem=56.90 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.20it/s]

    Capturing num tokens (num_tokens=768 avail_mem=56.89 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=704 avail_mem=56.89 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=704 avail_mem=56.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.65it/s]Capturing num tokens (num_tokens=640 avail_mem=56.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.65it/s]Capturing num tokens (num_tokens=576 avail_mem=56.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.65it/s]Capturing num tokens (num_tokens=512 avail_mem=56.88 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.65it/s]Capturing num tokens (num_tokens=480 avail_mem=56.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.65it/s]Capturing num tokens (num_tokens=448 avail_mem=56.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.65it/s]Capturing num tokens (num_tokens=416 avail_mem=56.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.65it/s]Capturing num tokens (num_tokens=416 avail_mem=56.89 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.54it/s]Capturing num tokens (num_tokens=384 avail_mem=56.89 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.54it/s]

    Capturing num tokens (num_tokens=352 avail_mem=56.88 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.54it/s]Capturing num tokens (num_tokens=320 avail_mem=56.88 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.54it/s]Capturing num tokens (num_tokens=288 avail_mem=56.87 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.54it/s]Capturing num tokens (num_tokens=256 avail_mem=56.87 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.54it/s]Capturing num tokens (num_tokens=256 avail_mem=56.87 GB):  64%|██████▍   | 37/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=240 avail_mem=56.87 GB):  64%|██████▍   | 37/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=224 avail_mem=56.86 GB):  64%|██████▍   | 37/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=208 avail_mem=56.86 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.83it/s]Capturing num tokens (num_tokens=192 avail_mem=56.86 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.83it/s]

    Capturing num tokens (num_tokens=176 avail_mem=56.86 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.83it/s]Capturing num tokens (num_tokens=176 avail_mem=56.86 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=160 avail_mem=56.85 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=144 avail_mem=56.85 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=128 avail_mem=56.85 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=112 avail_mem=56.84 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=96 avail_mem=56.84 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.72it/s] Capturing num tokens (num_tokens=96 avail_mem=56.84 GB):  81%|████████  | 47/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=80 avail_mem=56.84 GB):  81%|████████  | 47/58 [00:01<00:00, 36.97it/s]

    Capturing num tokens (num_tokens=64 avail_mem=56.83 GB):  81%|████████  | 47/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=48 avail_mem=56.83 GB):  81%|████████  | 47/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=32 avail_mem=56.83 GB):  81%|████████  | 47/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=32 avail_mem=56.83 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=28 avail_mem=56.82 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=24 avail_mem=56.82 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=20 avail_mem=56.82 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=16 avail_mem=56.82 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=16 avail_mem=56.82 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.60it/s]Capturing num tokens (num_tokens=12 avail_mem=56.81 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.60it/s]

    Capturing num tokens (num_tokens=8 avail_mem=56.81 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.60it/s] Capturing num tokens (num_tokens=4 avail_mem=56.80 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.60it/s]Capturing num tokens (num_tokens=4 avail_mem=56.80 GB): 100%|██████████| 58/58 [00:01<00:00, 37.59it/s]


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
    Generated text:  Mikaela and I'm 28 years old. I'm a 10th grade senior at Marshall High School in South Africa. I have a passion for history, and I work in a museum. I love watching the exhibits, and I also have a love for the outdoors. My sister has a passion for the outdoors, and we work together to help her with her outdoor activities.
    What is your favorite hobby or activity, and how do you get started with it? As a 10th grade senior at Marshall High School in South Africa, I enjoy going to the local historical museum to learn about important events in history
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to build in different countries. He knows that each military base requires 50 military personnel and each soldier requires 20 units of supplies. If the president wants to build 10 bases in each country, how many units of supplies would he need for all these bases? To determine the total units of supplies required for all the bases, we need to follow these steps:
    
    1. Calculate the number of units of supplies needed for one military base.
    2. Calculate the number of units of supplies needed for 10 bases in one country.
    3. Calculate the total units of supplies needed for 
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located on the western bank of the River Seine, a major river in the north of France. The river rises in the mountains of the Massif Central, where it is joined by the Rhone River, which is a tributary of the Seine. The Seine is a long and narrow river, rising from the valleys of the Alps. Its width is 2.7 km, and its length is 135 km. The Seine is the longest river of France. The current of the Seine is very strong, reaching its greatest speed at the point where it meets the Rhone.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of individual users, not the corporate sector. In today’s fast-paced world, individuals are constantly bombarded with digital information, often at the expense of their time. The challenge for businesses is to find ways to make digital information more accessible and meaningful for individuals. One approach that has been gaining popularity is the use of virtual assistants and chatbots.
    Virtual assistants and chatbots are digital assistants that can interact with users via text, voice, or chat. They can provide information, answer questions, and even help with tasks. These assistants and chatbots can be used in a variety of ways, including virtual tutoring, customer service,


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


    Generated text:  [Name] and I am a [occupation] who has been [number of years] in the industry. I am passionate about [reason for passion], and I am always looking for ways to [action or goal]. I am a [character trait] and I am always [character trait]. I am [character trait] and I am always [character trait]. I am [character trait] and I am always [character trait]. I am [character trait] and I am always [character trait]. I am [character trait] and I am always [character trait]. I am [character trait] and I am always [character trait
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to many world-renowned museums, including the Musée d'Orsay and the Musée Rodin. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is also known for its cuisine, including French cuisine, which is famous for its rich flavors and complex ingredients. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also known for its fashion industry, with many famous
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we can expect to see more automation and AI-driven technologies becoming more prevalent in various industries. This could lead to job displacement, but also create new opportunities for people to work in areas like data analysis, machine learning, and software development.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security measures. This could lead to the development
    


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
    Generated text:  [Name], I'm an [age] year old, [gender] girl, and I'm from [city]. I'm in my first year of high school and I enjoy [interest or hobby]. I love making friends and I'm always looking for new experiences to try. I'm an [job title] and I enjoy [job title]. I have a sense of humor and I'm always on the lookout for new challenges and opportunities to grow and improve. I'm ready for the next adventure and I look forward to meeting you! [Your name]. Your message has arrived and I'm interested in learning more about you. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and the country's most populous city, with an estimated population of 10 million. The city is located in the south of the country, on the banks of the Seine River, and is home to many of France's oldest and most prestigious museums, including the Louvre. The city's rich history and cultural heritage are celebrated through its iconic landmarks, such as the Eiffel Tower, Notre-Dame Cathedral, and the Arc de Triomphe. Paris is also known for its romantic and artistic atmosphere, with its many cafes, theaters, and theaters, as well as its vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be one of rapid and continuous innovation, with a focus on improving the quality of life and creating new opportunities for humanity. Here are some possible future trends in AI:
    
    1. Personalized AI: As we collect more and more data on individuals, we will be able to create AI that is highly personalized and tailored to each person's needs. This could mean that AI could predict an individual's preferences, behavior, and activities, and adjust its responses accordingly.
    
    2. Autonomous vehicles: Self-driving cars are likely to become a reality in the near future, with AI playing a crucial role in their development. This could mean that AI is


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

     an

     artist

     and

     illustrator

     with

     a

     passion

     for

     creating

     beautiful

     and

     expressive

     works

     of

     art

    .

     I

    'm

     known

     for

     my

     ability

     to

     transform

     ordinary

     objects

     into

     breathtaking

    ly

     beautiful

     pieces

     of

     art

    .

     My

     work

     often

     explores

     themes

     of

     creativity

    ,

     imagination

    ,

     and

     the

     beauty

     of

     the

     world

     around

     us

    .

     I

    'm

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

     evolve

     as

     an

     artist

    .

     Thank

     you

     for

     asking

    !

     #

    Art

     |

     #

    Illustr

    ator

     |

     #

    Creative

     #

    Imp

    ress

    ive

     #

    Expl

    oring

     #

    Challenge

     #

    Ev

    olution

     #

    New

    Start

    s

     #

    Art

    In

    Motion

     #

    Artist

    Born

     #

    Creative

    Life

     #

    Express

    ive

    Works

     #

    Beautiful

    Art

     #

    Im

    agination

     #

    World

    Of

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     heart

     of

     the

     French

     countryside

     and

     known

     for

     its

     historic

     landmarks

    ,

     fashion

     industry

    ,

     and

     world

    -class

     museums

    .

     Its

     status

     as

     the

     largest

     city

     in

     the

     European

     Union

     is

     reflected

     in

     its

     population

     of

     over

     

    1

    0

     million

     residents

    ,

     which

     makes

     it

     the

     most

     populous

     city

     in

     France

     by

     a

     wide

     margin

    .

     Paris

     is

     also

     home

     to

     several

     of

     the

     world

    's

     most

     renowned

     art

     galleries

    ,

     museums

    ,

     and

     cultural

     institutions

    ,

     including

     the

     Lou

    vre

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Mus

    ée

     national

     d

    '

    Art

     moderne

    .

     Its

     status

     as

     a

     major

     economic

     hub

     for

     the

     region

     and

     beyond

     is

     reflected

     in

     its

     high

     standards

     of

     living

    ,

     world

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     full

     of

     possibility

    .

     Here

     are

     some

     potential

     trends

     that

     may

     emerge

     in

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     is

     becoming

     more

     integrated

     with

     other

     technologies

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     robotics

    .

     This

     integration

     could

     lead

     to

     more

     efficient

     and

     effective

     use

     of

     AI

     in

     various

     industries

    .
    


    2

    .

     Enhanced

     personal

    ization

    :

     AI

    -powered

     chat

    bots

     and

     virtual

     assistants

     are

     likely

     to

     become

     more

     personalized

     in

     the

     future

    .

     This

     could

     lead

     to

     better

     customer

     service

    ,

     more

     accurate

     recommendations

    ,

     and

     more

     efficient

     use

     of

     resources

    .
    


    3

    .

     AI

    -powered

     healthcare

    :

     AI

     could

     revolution

    ize

     the

     healthcare

     industry

    ,

     with

     applications

     in

     diagnosis

    ,

     treatment

    ,

     and

    



```python
llm.shutdown()
```
