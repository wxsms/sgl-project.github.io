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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.82it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.80it/s]


    2026-04-09 21:22:23,723 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 21:22:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:02<00:07,  6.28it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:03, 12.66it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:03, 12.66it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:03, 12.66it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.66it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.66it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.66it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.66it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.66it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.73it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.73it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.73it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.73it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.73it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.73it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.73it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.73it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 39.14it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 39.14it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 39.14it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 39.14it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 39.14it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 39.14it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 39.14it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 39.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   9%|▊         | 5/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.29it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.24it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=352 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=256 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]

    Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.70it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=48 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.09it/s]

    Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.16it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 39.17it/s]


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
    Generated text:  Tony. I'm a student in Grade 10. I like swimming and playing sports. My favorite sport is basketball, and I like playing with my friends. I also like to listen to music and watch TV. My favorite movie is "BoJack Horseman", I think it is very funny. What do you think I like to do? 1. What do you think I like to do? 2. What do you think I like to eat? 3. What do you think I like to wear? 4. What do you think I like to read? 5. What do you think I like to
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 4 inches tall. The vice president of the United States is 5 feet 10 inches tall. If the president's height is 1 inch shorter than the vice president's height, how tall are both people in feet and inches? To determine the height of the president and the vice president in inches, we start by converting their heights from feet and inches to just inches. We know that 1 foot is equal to 12 inches, so the height of the president in inches is:
    
    \[5 \text{ feet} \times 12 \text{ inches/foot} + 4 \text
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is one of the world's most famous cities and is a major commercial and cultural center. It has a rich and complex history. Founded in the 7th century, the city was founded by the Gauls and was the capital of the Gauls before being sacked by the Romans and later by the English. The city was originally called "Paris", which is why it is called Paris.
    
    The earliest known map of Paris shows the site of the castle of Saint-Rémy, which was later known as the Château de Chambord, also known as the Château of the Chambord, in
    ===============================
    Prompt: The future of AI is
    Generated text:  still uncertain, but the path forward is clear. Predicting the future is hard, and predicting the future is easy. A simple example of predicting the future is creating a basic model that predicts the outcome of a political campaign based on the campaign’s underlying variables. Without knowing the underlying variables, you could potentially have a great understanding of how a campaign would perform but not being able to predict the exact outcome. However, by analyzing the data collected from the campaign and incorporating the data into the model, you can create a model that more accurately predicts the outcome.
    In the same way, AI is expected to revolutionize many sectors, including healthcare,


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


    Generated text:  Paris. It is the largest city in France and the third-largest city in the European Union. It is also the oldest capital city in Europe, having been the seat of the French monarchy since the 13th century. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its fashion industry, with many famous fashion houses and boutiques located in the city. Paris is a popular tourist destination, with millions of visitors each year. It is also a major financial
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that could be expected in the future of AI:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, privacy, and transparency. This could
    


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
    Generated text:  [insert name], and I'm a [insert age range or profession] with [insert relevant experience or skills]. I have a passion for [insert a compelling reason why you're interested in this topic], and I'm always eager to learn new things. My goal is to become [insert a personal goal or aspiration], and I'm always willing to take on challenges and work hard to achieve it. I'm a [insert a neutral personality trait or trait you find admirable], and I'm always ready to be a good friend, mentor, or partner. I enjoy [insert a hobby or activity] and find it incredibly satisfying to work towards
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city in the country and the seat of government for the country's capital region and its largest city.
    The answer is: Paris is the capital city of France. It is the most populous city in the country and the seat of government for the country's capital region and its largest city.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be a rapidly evolving landscape, characterized by ever-changing technologies and new applications. Some potential future trends in artificial intelligence include:
    
    1. Advancements in machine learning: AI is set to continue its growth as it continues to learn from more data and improve its ability to understand and solve complex problems.
    
    2. Increased integration with physical and biological systems: AI will continue to be integrated with various physical and biological systems, from healthcare to transportation to manufacturing.
    
    3. AI in healthcare: AI will play an increasing role in healthcare, with the ability to analyze vast amounts of patient data to predict disease progression and develop personalized treatment plans.
    
    4. Autonomous


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

    ]

     and

     I

    'm

     a

     [

    Your

     Age

    ]

     year

     old

     aspiring

     [

    Your

     Job

     Title

    ]

     with

     a

     passion

     for

     [

    Your

     Hobby

    /

    Interest

    /

    Prof

    ession

    ]

    !

     I

     enjoy

     [

    Your

     Passion

    ,

     such

     as

     cooking

    ,

     photography

    ,

     or

     art

    ],

     and

     I

     believe

     that

     my

     unique

     combination

     of

     skills

    ,

     experience

    ,

     and

     creativity

     has

     led

     me

     to

     this

     position

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

     succeed

     in

     my

     career

    .

     Looking

     forward

     to

     meeting

     you

    !

     #

    h

    obby

     #

    interest

     #

    career

     #

    self

    int

    roduction

    


    #

    t

    ire

    less

     #

    amb

    itious

     #

    entre

    preneur

     #

    journal

    ist

     #

    v

    log

    ger

     #

    phot

    ographer

     #

    art

    ista

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

     and

     the

     largest

     city

     in

     the

     European

     Union

     by

     population

    .

     It

     is

     located

     on

     the

     Se

    ine

     river

     in

     the

     Paris

     region

    ,

     and

     is

     the

     seat

     of

     the

     French

     government

    ,

     capital

    ,

     and

     most

     populous

     city

     in

     France

    .

     The

     city

     is

     also

     the

     largest

     in

     terms

     of

     area

    ,

     at

     

    1

    ,

     

    0

    9

    7

     square

     kilometres

     (

    4

    3

    4

     sq

     mi

    ),

     and

     is

     home

     to

     many

     of

     France

    's

     cultural

     institutions

    ,

     including

     the

     Lou

    vre

     and

     the

     Notre

     Dame

     Cathedral

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     architecture

    ,

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     The

     city

     is

     also

     home

     to

     many

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

     and

     largely

     uncertain

    ,

     with

     potential

     advancements

     in

     areas

     such

     as

     machine

     learning

    ,

     robotics

    ,

     and

     natural

     language

     processing

    .

     However

    ,

     here

     are

     some

     possibilities

     for

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     Automation

    :

     One

     of

     the

     most

     promising

     trends

     in

     AI

     is

     automation

     of

     tasks

     that

     are

     currently

     performed

     by

     humans

    .

     This

     could

     lead

     to

     increased

     efficiency

     and

     productivity

    ,

     but

     could

     also

     create

     new

     job

     opportunities

     in

     areas

     like

     robotics

     and

     data

     analysis

    .
    


    2

    .

     AI

     Ethics

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    ,

     it

     is

     important

     to

     address

     ethical

     issues

     such

     as

     bias

    ,

     privacy

    ,

     and

     accountability

    .

     This

     could

     lead

     to

     changes

     in

     how

     AI

     is

     developed

    ,

     monitored

    ,

     and

     used

    ,

    



```python
llm.shutdown()
```
