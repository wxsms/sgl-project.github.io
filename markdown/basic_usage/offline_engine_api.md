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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.57it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.56it/s]


    2026-05-04 19:41:22,939 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 19:41:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.55it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 11.95it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]

    Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 27.90it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 37.74it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 37.74it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 37.74it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 37.74it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 37.74it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 37.74it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 37.74it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 37.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.35 GB):   3%|▎         | 2/58 [00:00<00:03, 14.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.34 GB):   3%|▎         | 2/58 [00:00<00:03, 14.53it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.34 GB):   3%|▎         | 2/58 [00:00<00:03, 14.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.34 GB):   7%|▋         | 4/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.33 GB):   7%|▋         | 4/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.32 GB):   7%|▋         | 4/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.32 GB):  10%|█         | 6/58 [00:00<00:03, 17.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.31 GB):  10%|█         | 6/58 [00:00<00:03, 17.24it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.25 GB):  10%|█         | 6/58 [00:00<00:03, 17.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.26 GB):  10%|█         | 6/58 [00:00<00:03, 17.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.26 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.25 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.26 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.26 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.26 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.26 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.25 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.25 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.24 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.24 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.25 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.23 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.22 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.22 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.22 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.20 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.07it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.21 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.07it/s] Capturing num tokens (num_tokens=896 avail_mem=74.20 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=832 avail_mem=74.19 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=832 avail_mem=74.19 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.88it/s]Capturing num tokens (num_tokens=768 avail_mem=74.19 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.88it/s]Capturing num tokens (num_tokens=704 avail_mem=74.18 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.88it/s]Capturing num tokens (num_tokens=640 avail_mem=74.18 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.88it/s]Capturing num tokens (num_tokens=576 avail_mem=74.17 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.88it/s]Capturing num tokens (num_tokens=512 avail_mem=74.17 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.88it/s]Capturing num tokens (num_tokens=512 avail_mem=74.17 GB):  50%|█████     | 29/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=480 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:01<00:00, 35.98it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.18 GB):  50%|█████     | 29/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=416 avail_mem=74.17 GB):  50%|█████     | 29/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=384 avail_mem=74.17 GB):  50%|█████     | 29/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=352 avail_mem=74.16 GB):  50%|█████     | 29/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=352 avail_mem=74.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=320 avail_mem=74.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=288 avail_mem=74.15 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=256 avail_mem=74.14 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=240 avail_mem=74.14 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]Capturing num tokens (num_tokens=224 avail_mem=74.13 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.32it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.13 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=208 avail_mem=74.12 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=192 avail_mem=74.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=176 avail_mem=74.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=160 avail_mem=74.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=144 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=144 avail_mem=74.10 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=128 avail_mem=74.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.59it/s] Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.59it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.90it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.90it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 34.19it/s]


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
    Generated text:  Harry Potter. I'm a young wizard with magical powers, and I've been on a journey to find a way to save the wizarding world from its enemies. Can you tell me more about your experiences in the wizarding world? The Dark Lord and his minions have been wreaking havoc, and the wizarding community is struggling to stay together. How did you come to be involved in this fight against the Dark Lord?
    
    As for my magical abilities, I can control the weather, heal wounds, and even make spells to defend myself and others. However, I'm not a fighter, and I believe that magic should be used for good
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing a new policy to improve the health of the nation. He wants to introduce a new tax on sugary drinks. He has collected data on the sales of sugary drinks in different regions of the United States over the last few years.
    
    He has determined that the number of sugary drinks sold in a given year is proportional to the number of people in that region. Specifically, the number of sugary drinks sold, \( S \), in a given year can be modeled by the formula \( S = 1000P \), where \( P \) is the number of people in the region.
    
    The president wants to calculate
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is a famous tourist destination. The population of Paris is 2.5 million. If 10% of the population immigrates to France and then 50% of the population emigrate, how many people will be left in Paris after these changes? 
    After the population changes, there will be 2.5 million x 0.10 = 250,000 people immigrating.
    After the population changes, there will be 2.5 million x 0.50 = 1.25 million emigrating.
    The total number of people left in Paris will
    ===============================
    Prompt: The future of AI is
    Generated text:  not a black and white proposition, but rather a spectrum of possibilities. However, when it comes to the role of AI in the future of manufacturing, a series of interesting possibilities exist.
    
    AI can help companies to improve their manufacturing processes, but it is not the only factor that plays a role in their success. Here are some reasons why AI is likely to play a significant role in the future of manufacturing, and what companies should consider when it comes to incorporating AI into their operations.
    
    AI can help companies to improve their manufacturing processes by automating and optimizing them. This can lead to increased efficiency, reduced costs, and better quality products. For


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. Let's chat! [Name] [Job Title] [Company Name] [Company Address] [City, State ZIP Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [Instagram Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [Instagram Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [Instagram Profile] [LinkedIn Profile
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a popular tourist destination and a major economic center. Paris is home to many famous French artists, writers, and musicians, and is a cultural and artistic hub. The city is also known for its rich history, including the Roman and French empires, and its role in the French Revolution and the French Revolution. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. It is a major transportation hub and a major economic center in Europe. The city is also known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency, reduced costs, and improved quality of life for many people.
    
    2. AI ethics and privacy concerns: As AI technology becomes more integrated into our daily lives, there will be increasing concerns about its ethical implications and potential privacy violations. This will likely lead to more stringent regulations and standards for
    


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
    Generated text:  [Name], and I am [Job Title]. I am a highly skilled [job title], with a passion for [job title], which I have honed through years of dedication and hard work. In my role, I am always trying to improve my skills and find new ways to solve complex problems. I am a team player and love working with others to achieve our goals, and I am always looking for ways to add value to the team. I have a friendly and approachable personality, and I am always ready to help and assist others. Please let me know if you would like to introduce me to anyone. [Name] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    [Code to display a paragraph about the facts about the capital city, such as its population, its status, or any unique aspects of the city.]
    Paris is the capital of France and the largest city in the country. It is also one of the most important cities in the world and is known for its rich history, art, and culture. The city is located on the left bank of the Seine River and is home to many of France's most famous landmarks and attractions, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also known for its numerous festivals and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting and has the potential to transform many industries and transform society in ways we have never experienced before. Some possible trends in AI include:
    
    1. Improved AI Ethics: There is a growing concern about the impact of AI on society and the environment. As AI becomes more advanced, there will be a need to ensure that it is used in a way that is ethical and responsible. AI developers will need to consider the impact of their creations on the environment and ensure that they do not harm the planet.
    
    2. Increased Use of AI in Healthcare: AI has the potential to revolutionize healthcare, from personalized treatment to the development of new medications.


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

    insert

     name

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ].

     I

     have

     a

     love

     for

     adventure

    ,

     travel

    ,

     and

     exploring

     new

     places

    .

     I

    'm

     always

     up

     for

     a

     challenge

     and

     always

     looking

     for

     new

     experiences

    .

     What

     kind

     of

     character

     would

     you

     be

     interested

     in

    ?

     In

     your

     experience

    ,

     how

     would

     you

     describe

     yourself

    ?

     What

     makes

     you

     unique

    ?

     And

     if

     you

     were

     to

     have

     a

     pet

    ,

     what

     would

     it

     be

    ?

     I

    'd

     love

     to

     hear

     your

     insights

     on

     me

     and

     why

     you

    'd

     want

     to

     know

     more

     about

     me

    .

     Sure

    ,

     I

    'd

     love

     to

     hear

     more

     about

     you

    !

     Can

     you

     give

     me

     some

     more

     details

     about

     your

     current

     interests

     and

     hobbies

    ?

     That

     way

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     "

    The

     capital

     of

     France

     is

     Paris

    "

     is

     fact

    ually

     accurate

     and

     includes

     all

     the

     key

     details

    :

     it

     is

     the

     capital

     city

     of

     France

    ,

     and

     it

     is

     the

     French

     capital

     city

    .

     
    


    To

     verify

     this

     statement

    ,

     one

     can

     easily

     search

     for

     the

     official

     website

     of

     the

     French

     government

     or

     the

     official

     information

     provided

     by

     the

     French

     government

     itself

    .

     They

     can

     then

     confirm

     that

     Paris

     is

     indeed

     the

     capital

     city

     of

     France

    .

     
    


    This

     statement

     also

     align

    s

     with

     the

     cultural

     and

     historical

     significance

     of

     Paris

     as

     the

     capital

     city

     of

     France

    .

     Paris

     is

     known

     for

     its

     historical

     and

     cultural

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     the

     Notre

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     there

     are

     a

     number

     of

     possible

     trends

     that

     we

     can

     expect

     to

     see

     over

     the

     next

     few

     decades

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     more

     data

     is

     collected

     and

     analyzed

     by

     AI

    ,

     we

     may

     see

     increased

     focus

     on

     ethical

     considerations

     and

     regulatory

     frameworks

     to

     ensure

     that

     AI

     is

     used

     in

     a

     responsible

     and

     beneficial

     way

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     into

     the

     fabric

     of

     our

     lives

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     a

     more

     seamless

     integration

     of

     AI

     into

     our

     daily

     lives

    ,

     from

     voice

     assistants

     to

     self

    -driving

     cars

    .
    


    3

    .

     AI

     will

     become

     more

     accessible

     to

     everyone

    :

     While

     AI

     has

     been

     used

     to

     perform

     tasks

     such

     as

     financial

     forecasting

    



```python
llm.shutdown()
```
