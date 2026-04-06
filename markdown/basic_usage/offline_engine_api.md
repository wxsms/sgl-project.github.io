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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.28it/s]


    2026-04-06 23:27:00,290 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 23:27:00] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:13,  3.87it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:13,  3.87it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:13,  3.87it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:13,  3.87it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:03<00:13,  3.87it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:03<00:13,  3.87it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:03<00:13,  3.87it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:05,  8.58it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:05,  8.58it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:05,  8.58it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:05,  8.58it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:05,  8.58it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:05,  8.58it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:03<00:05,  8.58it/s]

    Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:03<00:05,  8.58it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 15.07it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 15.07it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 15.07it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 15.07it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 15.07it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 15.07it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 15.07it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 15.07it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 22.10it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 22.10it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 22.10it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 22.10it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 22.10it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 22.10it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 22.10it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.79it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.79it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.79it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.79it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.79it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.79it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.79it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.86it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.86it/s]

    Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.86it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.86it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.86it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.86it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.86it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.01it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.01it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.01it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.01it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.01it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.01it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.41it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.41it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.41it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.41it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.41it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.41it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.41it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.41it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 44.41it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 44.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.67 GB):   3%|▎         | 2/58 [00:00<00:03, 15.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.66 GB):   3%|▎         | 2/58 [00:00<00:03, 15.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.66 GB):   3%|▎         | 2/58 [00:00<00:03, 15.93it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=131.66 GB):   7%|▋         | 4/58 [00:00<00:03, 17.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.65 GB):   7%|▋         | 4/58 [00:00<00:03, 17.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.65 GB):   7%|▋         | 4/58 [00:00<00:03, 17.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.65 GB):   7%|▋         | 4/58 [00:00<00:03, 17.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.28it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.41it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=960 avail_mem=131.59 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.73it/s] Capturing num tokens (num_tokens=896 avail_mem=131.59 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=768 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=704 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=704 avail_mem=131.58 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.79it/s]

    Capturing num tokens (num_tokens=480 avail_mem=131.58 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=288 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=288 avail_mem=131.56 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=224 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.14it/s]

    Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=160 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.87it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=64 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.87it/s]

    Capturing num tokens (num_tokens=48 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=28 avail_mem=131.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.82it/s] Capturing num tokens (num_tokens=4 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=4 avail_mem=131.49 GB): 100%|██████████| 58/58 [00:01<00:00, 38.87it/s]


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
    Generated text:  Sarah and I am a senior at Reed College in Oregon. I’m interested in English and have been writing poetry and experimenting with prose for a long time. When I was a teenager I became interested in a new type of writing called “prose poetry” and have been experimenting with it since. I’ve even written poems that use the word “prose.” I’ve published two collections of poetry and I have just published a collection of prose poems. I’m now working on my next collection of prose poems called “Where I Should Have Been.” This collection is about my first time being the first person to walk my son, Alex, into
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what sport to promote for his country. He will choose the sport to be the theme of a campaign, and there are four sports that he considers: football, baseball, hockey, and soccer. For how many combinations of the four sports are the preferences of the four most populous countries of the world represented? To solve the problem, we need to determine the number of combinations of the four sports (football, baseball, hockey, and soccer) such that the preferences of the four most populous countries of the world are represented. The four most populous countries are China, India, Russia, and the United States.
    
    The four sports can
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. This city is the capital of France and the country’s most populous city. It was chosen as the capital because it was the birthplace of the French Revolution, and is the birthplace of the French Republic, which was the second country to recognize the French Revolution. The city is also home to many attractions including museums and monuments that showcase the history and culture of the French people.
    Paris has a long and rich history that dates back to the Roman Empire. The city was founded as the capital of the Roman Empire in the 1st century AD. However, it was not until the 5th century AD that the city was
    ===============================
    Prompt: The future of AI is
    Generated text:  a bright and hopeful one. The combination of advances in computer science, artificial intelligence, and deep learning has transformed the way we think about and interact with the world around us. AI is now capable of performing a vast array of tasks that would have been previously impossible. It can assist in areas such as healthcare, finance, transportation, and more. The potential applications of AI are vast, and there is no doubt that it will continue to grow and evolve in the coming years.
    One of the most exciting developments in the field of AI is the advancement of natural language processing (NLP). NLP is the ability of computers to understand, interpret


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [What I like to do]. I'm [What I enjoy doing]. I'm [What I'm passionate about]. I'm [What I'm good at]. I'm [What I'm good at]. I'm [What I'm good at]. I'm [What I'm good at]. I'm [What I'm good at]. I'm [What I'm good at]. I'm [What I'm good at]. I'm [What I'm good at]. I'm [What I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic center with a rich history dating back to the Middle Ages. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its vibrant nightlife, including the famous Montmartre district. Paris is a popular tourist destination and a major economic hub in Europe. It is home to many international organizations and is a major transportation hub for Europe. The city is also known for its cuisine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to improve, we can expect to see even more widespread adoption of AI in healthcare, with more sophisticated algorithms and machine learning models being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in
    


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
    Generated text:  [Name] and I'm [Age] years old. I'm a [occupation] and I enjoy [activity/s]. I'm a [personal trait or value] and I believe [why, if relevant]. I'm [interest or hobby] and I enjoy [time with friends] and [time with family].
    
    When I'm [medium of interaction]. I'm an [agent or trainer] and I'm always [positive or optimistic]. I'm [relationship status] and I'm always [friendly or cooperative]. I'm [where I work]. I'm [your potential job title] and I'm [your superpower]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and one of the most visited cities in the world. It is located in the center of France and is known for its rich history, beautiful architecture, and annual cultural events. Paris is also a popular tourist destination, attracting millions of visitors each year. Its iconic landmarks such as the Eiffel Tower and Louvre Museum are some of the world's most recognizable structures. Paris is also a major center for business and finance, with the French parliament, French President, and French CEOs all residing there. Despite being a world-renowned city, Paris remains relatively quiet and has a low population density,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be marked by a number of significant developments and trends. Some of the most notable trends include:
    
    1. Increased precision and accuracy: One of the most significant trends in AI is the increasing focus on improving the precision and accuracy of AI models. As AI systems become more complex and involve a wider range of data, they are likely to become more accurate and reliable.
    
    2. Personalization and adaptability: With the growing reliance on AI for personalization and adaptability, there is an increasing focus on developing AI systems that can learn from user behavior and make personalized recommendations. This trend is particularly relevant in the areas of healthcare, education,


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

    name

    ],

     and

     I

     am

     [

    age

    ].

     I

     am

     a

     person

     of

     [

    occupation

    ],

     and

     I

     love

     [

    what

     I

     do

    /

    love

     to

     do

    ].

     I

     have

     a

     great

     sense

     of

     humor

    ,

     and

     I

     love

     [

    reason

     for

     humor

    ].

     I

     have

     always

     been

     passionate

     about

     [

    something

     that

     makes

     me

     happy

    ],

     and

     I

     love

     to

     [

    reason

     for

     passion

    ].

     I

     am

     always

     looking

     for

     new

     experiences

     to

     expand

     my

     hor

    izons

    ,

     and

     I

     love

     [

    reason

     for

     this

    ].

     I

    'm

     always

     trying

     to

     learn

     new

     things

     and

     challenge

     myself

    ,

     and

     I

     love

     [

    reason

     for

     this

    ].

     I

    'm

     a

     [

    open

    ness

     level

    ],

     and

     I

     like

     to

     [

    amb

    ition

     level

    ].

     I

     love

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     is

     accurate

     and

     directly

     addresses

     the

     capital

     city

     of

     France

    .

     Here

     is

     the

     complete

     statement

    :
    


    The

     capital

     of

     France

     is

     Paris

    .

     
    


    This

     concise

     statement

     provides

     a

     direct

     and

     factual

     answer

     to

     the

     question

     posed

    .

     It

     is

     a

     clear

     and

     un

    ambiguous

     statement

     that

     addresses

     the

     capital

     city

     of

     France

     in

     a

     clear

     and

     straightforward

     manner

    .

     
    


    To

     further

     elaborate

    ,

     Paris

    ,

     officially

     known

     as

     the

     "

    City

     of

     Light

    ,"

     is

     the

     capital

     and

     largest

     city

     of

     France

    .

     It

     is

     a

     historical

     and

     cultural

     center

    ,

     known

     for

     its

     art

    ,

     cuisine

    ,

     and

     unique

     architecture

    .

     Paris

     is

     home

     to

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     possible

     trends

     that

     are

     being

     predicted

    :
    


    1

    .

     Automation

    :

     The

     integration

     of

     machines

     into

     human

     work

     processes

     is

     becoming

     more

     prevalent

    ,

     with

     AI

     systems

     taking

     on

     a

     greater

     number

     of

     tasks

    .

     This

     will

     lead

     to

     significant

     job

     displacement

    ,

     but

     it

     could

     also

     create

     new

     jobs

     in

     areas

     such

     as

     data

     analysis

    ,

     machine

     learning

    ,

     and

     robotics

    .
    


    2

    .

     Personal

    ization

    :

     With

     AI

     systems

     becoming

     more

     sophisticated

    ,

     they

     will

     be

     able

     to

     provide

     more

     personalized

     experiences

     to

     users

    .

     This

     could

     lead

     to

     improvements

     in

     customer

     service

    ,

     healthcare

    ,

     and

     education

    ,

     but

     it

     could

     also

     result

     in

     the

     need

     for

     more

     data

     collection

     and

     storage

    .
    


    3

    .

     Ethics

     and

     governance

    



```python
llm.shutdown()
```
