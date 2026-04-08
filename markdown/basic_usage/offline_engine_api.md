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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.28it/s]


    2026-04-08 08:52:25,929 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 08:52:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:26,  2.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:26,  2.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:26,  2.58s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:38,  1.42it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:38,  1.42it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:38,  1.42it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:38,  1.42it/s]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:38,  1.42it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:12,  4.06it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:12,  4.06it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:12,  4.06it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:12,  4.06it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:02<00:12,  4.06it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:02<00:12,  4.06it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:02<00:12,  4.06it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:02<00:05,  8.94it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:02<00:05,  8.94it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:02<00:05,  8.94it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:02<00:05,  8.94it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:02<00:05,  8.94it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:02<00:05,  8.94it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:02<00:05,  8.94it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:02<00:05,  8.94it/s]

    Compiling num tokens (num_tokens=1024):  22%|██▏       | 13/58 [00:03<00:05,  8.94it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 16.86it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]

    Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 25.26it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 33.20it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 47.40it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 47.40it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 47.40it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 47.40it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 47.40it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 47.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.43 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:04, 12.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:04, 12.29it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:04, 12.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.42 GB):  10%|█         | 6/58 [00:00<00:07,  6.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.42 GB):  10%|█         | 6/58 [00:00<00:07,  6.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.41 GB):  10%|█         | 6/58 [00:00<00:07,  6.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.36 GB):  10%|█         | 6/58 [00:00<00:07,  6.67it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=53.36 GB):  16%|█▌        | 9/58 [00:00<00:04,  9.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.36 GB):  16%|█▌        | 9/58 [00:00<00:04,  9.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.35 GB):  16%|█▌        | 9/58 [00:01<00:04,  9.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.35 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.35 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.01it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=53.34 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.34 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.34 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.02it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=53.34 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.34 GB):  26%|██▌       | 15/58 [00:01<00:05,  7.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.34 GB):  26%|██▌       | 15/58 [00:01<00:05,  7.54it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=53.33 GB):  26%|██▌       | 15/58 [00:01<00:05,  7.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.33 GB):  29%|██▉       | 17/58 [00:01<00:04,  9.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.33 GB):  29%|██▉       | 17/58 [00:01<00:04,  9.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  29%|██▉       | 17/58 [00:01<00:04,  9.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.05it/s]

    Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.05it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.05it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.05it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=512 avail_mem=70.94 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  52%|█████▏    | 30/58 [00:02<00:01, 25.12it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  52%|█████▏    | 30/58 [00:02<00:01, 25.12it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  52%|█████▏    | 30/58 [00:02<00:01, 25.12it/s]

    Capturing num tokens (num_tokens=384 avail_mem=70.95 GB):  52%|█████▏    | 30/58 [00:02<00:01, 25.12it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  52%|█████▏    | 30/58 [00:02<00:01, 25.12it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:02<00:00, 27.31it/s]Capturing num tokens (num_tokens=320 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:02<00:00, 27.31it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:02<00:00, 27.31it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:02<00:00, 27.31it/s]Capturing num tokens (num_tokens=240 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:02<00:00, 27.31it/s]

    Capturing num tokens (num_tokens=240 avail_mem=70.93 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.47it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.47it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.47it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.47it/s]

    Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  71%|███████   | 41/58 [00:02<00:00, 19.57it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  71%|███████   | 41/58 [00:02<00:00, 19.57it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:02<00:00, 19.57it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:02<00:00, 19.57it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:02<00:00, 19.57it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:02<00:00, 19.57it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.09it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.09it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.09it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.09it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.09it/s]

    Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  86%|████████▌ | 50/58 [00:03<00:00, 27.78it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  86%|████████▌ | 50/58 [00:03<00:00, 27.78it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:03<00:00, 27.78it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:03<00:00, 27.78it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:03<00:00, 27.78it/s]Capturing num tokens (num_tokens=16 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:03<00:00, 27.78it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:03<00:00, 27.78it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:03<00:00, 33.67it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:03<00:00, 33.67it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  97%|█████████▋| 56/58 [00:03<00:00, 33.67it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:03<00:00, 18.35it/s]


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
    Generated text:  Jay, a proud 17 year old male, I am a young adult with access to the internet, and I have over 1,000 friends on Facebook. I am not a terrorist, nor am I involved in any criminal activity. I just want to say thank you for the internet and the internet access I have. 
    
    What is the most common way you access the internet? 
    What is the most common way you learn about new things? 
    What is the most common way you get news? 
    What is the most common way you get information on the internet? 
    What is the most common way you find information
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy and decides to visit three different countries. The president starts in Washington, D.C., and then goes to Paris, then returns to Washington, D.C. If the president's schedule repeats every day, what is the total number of days it will take for him to make a round trip?
    To determine the total number of days it will take for the president to make a round trip, we need to calculate the total distance he travels and then figure out how many days it will take to cover that distance once a day.
    
    First, let's calculate the distance the president travels each day:
    1. The distance from Washington, D.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. True
    B. False
    Answer:
    A
    
    Which of the following statements about the implementation of the EU's conditional payment directive is incorrect?
    A. The implementation of the conditional payment directive will affect the price of pharmaceutical products.
    B. The implementation of the conditional payment directive will affect the reimbursement ratio for health insurance.
    C. The implementation of the conditional payment directive will affect the reimbursement ratio for public healthcare.
    D. The implementation of the conditional payment directive will affect the reimbursement ratio for public education.
    Answer:
    C
    
    Which of the following statements about the liquidity of the share capital of a listed company is incorrect?
    A.
    ===============================
    Prompt: The future of AI is
    Generated text:  still far away. But what is AI, really? It’s an umbrella term encompassing all artificial intelligence, from designing intelligent machines to understanding human intelligence. In fact, it’s a collection of all the different technologies that allow machines to learn from data and have the ability to do tasks, like reading text, playing chess, and writing code. Let’s get into the nitty-gritty of what AI actually is, what it can do, and what it can’t do.
    What is AI?
    AI, also known as artificial intelligence, is the development and application of intelligent machines that can perform tasks and solve problems based on data. It


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


    Generated text:  [Name] and I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as a vibrant arts and culture scene. Paris is a popular tourist destination and a major economic center in France. It is also known for its cuisine, fashion, and wine industry. The city is home to many important institutions, including the French Academy of Sciences and the Louvre Museum. Overall, Paris is a fascinating city with a rich history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for workers and businesses.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use
    


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
    Generated text:  [Name]. I am an avid reader of fiction books. I have been reading books since I was a child and have always been fascinated by the characters and plotlines of the stories. I enjoy discovering new worlds and characters, and I have a passion for storytelling. I love writing my own stories and sharing them with others. What's your favorite book or author that inspires you? As an AI language model, I don't have personal preferences or feelings, but I can tell you that my favorite books and authors are those that inspire me and my users to create stories and communicate with others. What's your favorite book or author that inspires you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That statement is accurate and includes the following key points:
    
    - Paris is the capital and largest city of France
    - It is located in the northwestern part of the country
    - Paris is known for its historical landmarks and architecture
    - It is the home of the French flag and official language of the French Republic
    - The city is the largest city in France by area and population, with a population of over 2. 3 million people
    
    Paris is also the seat of the French government and the head of state, as well as the largest city in the world by population. Its famous landmarks include Notre-Dame de Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and several trends are expected to shape its development. Here are some of the most promising trends:
    
    1. Increased automation and robotics: As automation and robotics continue to advance, they will become more integrated into the industrial and business processes. This will lead to increased efficiency, cost savings, and potential job displacement, but it could also create new job opportunities.
    
    2. Improved privacy and data protection: With the increasing amount of personal data being collected and used, there is a growing concern about the security and privacy of that data. AI systems will need to be designed with robust privacy protections to protect sensitive information.
    
    3. Enhanced intelligence and


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

    Job

     title

    ].

     I

    've

     always

     been

     passionate

     about

     [

    specific

     hobby

     or

     interest

    ].

     Let

    's

     chat

     about

     [

    something

     you

     both

     enjoy

     or

     are

     familiar

     with

    ].

     How

     are

     you

     today

    ?

     [

    Name

    ]:

     This

     is

     my

     first

     intro

    .

     It

    's

     nice

     to

     meet

     you

    .

     How

     are

     you

    ?

     I

    'm

     [

    Name

    ],

     and

     I

    'm

     a

     [

    Job

     title

    ].

     I

    'm

     always

     passionate

     about

     [

    specific

     hobby

     or

     interest

    ].

     And

     let

    's

     chat

     about

     [

    something

     you

     both

     enjoy

     or

     are

     familiar

     with

    ].

     How

     about

     you

    ?

     What

    's

     new

    ?

     What

    's

     been

     your

     favorite

     trip

     ever

    ?

     And

     what

    's

     your

     favorite

     book

    ?

     I

    'm

     your

     new

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     the

     Birth

    place

     of

     the

     Enlightenment

    ,

     and

     the

     City

     of

     Love

    .

     It

     is

     home

     to

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

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     iconic

     landmarks

    .

     Paris

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     cultural

     heritage

    ,

     including

     French

     cuisine

    ,

     wine

    ,

     and

     music

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     known

     for

     its

     history

    ,

     art

    ,

     and

     culture

    ,

     and

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     a

     thousand

     days

    ."

     Paris

     is

     a

     city

     of

     contrasts

    ,

     from

     its

     historic

     architecture

     to

     its

     lively

     nightlife

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     more

     sophisticated

     and

     diverse

     applications

    ,

     as

     well

     as

     greater

     integration

     with

     other

     technologies

    .

     Some

     potential

     trends

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     can

     be

     used

     to

     diagnose

     diseases

     more

     accurately

    ,

     provide

     personalized

     treatment

     plans

    ,

     and

     monitor

     patient

     progress

    .

     Additionally

    ,

     AI

     can

     be

     used

     to

     improve

     the

     accuracy

     of

     medical

     imaging

    ,

     such

     as

     MRI

     and

     X

    -rays

    ,

     leading

     to

     more

     accurate

     diagnoses

    .
    


    2

    .

     AI

     in

     manufacturing

    :

     AI

     can

     be

     used

     to

     optimize

     production

     processes

    ,

     reduce

     waste

    ,

     and

     improve

     quality

     control

    .

     Additionally

    ,

     AI

     can

     be

     used

     to

     develop

     new

     materials

     and

     technologies

    ,

     leading

     to

     the

     development

     of

     new

     products

     and

     services

    .
    


    3

    .

     AI

    



```python
llm.shutdown()
```
