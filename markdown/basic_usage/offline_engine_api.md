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


    2026-04-10 07:21:41,749 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 07:21:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.23it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.23it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.23it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.83it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.83it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.83it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.83it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.58it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.58it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.58it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.58it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.72it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.72it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.72it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.72it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 17.48it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 17.48it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 17.48it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 17.48it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 17.48it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 21.20it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 21.20it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 21.20it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 21.20it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 21.20it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 24.38it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 24.38it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 24.38it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 24.38it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 24.38it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 26.93it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 26.93it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 26.93it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 26.93it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 26.93it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 26.93it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 26.93it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 33.74it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s]

    Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 43.53it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 51.63it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 51.63it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 51.63it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 51.63it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 51.63it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 51.63it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.09 GB):   3%|▎         | 2/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.09 GB):   3%|▎         | 2/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.08 GB):   3%|▎         | 2/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.09 GB):   3%|▎         | 2/58 [00:00<00:02, 19.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.09 GB):   9%|▊         | 5/58 [00:00<00:02, 23.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.08 GB):   9%|▊         | 5/58 [00:00<00:02, 23.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.08 GB):   9%|▊         | 5/58 [00:00<00:02, 23.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.08 GB):   9%|▊         | 5/58 [00:00<00:02, 23.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.08 GB):   9%|▊         | 5/58 [00:00<00:02, 23.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.24it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=57.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.04 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=960 avail_mem=57.03 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.70it/s] Capturing num tokens (num_tokens=896 avail_mem=57.03 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=832 avail_mem=57.03 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.70it/s]

    Capturing num tokens (num_tokens=768 avail_mem=57.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=704 avail_mem=57.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=704 avail_mem=57.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=640 avail_mem=57.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=576 avail_mem=57.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=512 avail_mem=57.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=480 avail_mem=57.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=448 avail_mem=57.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=416 avail_mem=57.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=416 avail_mem=57.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.88it/s]Capturing num tokens (num_tokens=384 avail_mem=57.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.88it/s]Capturing num tokens (num_tokens=352 avail_mem=57.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.88it/s]Capturing num tokens (num_tokens=320 avail_mem=57.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.88it/s]

    Capturing num tokens (num_tokens=288 avail_mem=57.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.88it/s]Capturing num tokens (num_tokens=256 avail_mem=57.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.88it/s]Capturing num tokens (num_tokens=240 avail_mem=57.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.88it/s]Capturing num tokens (num_tokens=240 avail_mem=57.00 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.63it/s]Capturing num tokens (num_tokens=224 avail_mem=56.99 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.63it/s]Capturing num tokens (num_tokens=208 avail_mem=56.99 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.63it/s]Capturing num tokens (num_tokens=192 avail_mem=56.99 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.63it/s]Capturing num tokens (num_tokens=176 avail_mem=56.99 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.63it/s]Capturing num tokens (num_tokens=160 avail_mem=56.98 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.63it/s]Capturing num tokens (num_tokens=144 avail_mem=56.98 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.63it/s]Capturing num tokens (num_tokens=144 avail_mem=56.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=128 avail_mem=56.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=112 avail_mem=56.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.57it/s]

    Capturing num tokens (num_tokens=96 avail_mem=56.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.57it/s] Capturing num tokens (num_tokens=80 avail_mem=56.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=64 avail_mem=56.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=48 avail_mem=56.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=48 avail_mem=56.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.86it/s]Capturing num tokens (num_tokens=32 avail_mem=56.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.86it/s]Capturing num tokens (num_tokens=28 avail_mem=56.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.86it/s]Capturing num tokens (num_tokens=24 avail_mem=56.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.86it/s]Capturing num tokens (num_tokens=20 avail_mem=56.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.86it/s]Capturing num tokens (num_tokens=16 avail_mem=56.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.86it/s]Capturing num tokens (num_tokens=12 avail_mem=56.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.86it/s]Capturing num tokens (num_tokens=12 avail_mem=56.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 51.39it/s]Capturing num tokens (num_tokens=8 avail_mem=56.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 51.39it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=56.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 51.39it/s]Capturing num tokens (num_tokens=4 avail_mem=56.93 GB): 100%|██████████| 58/58 [00:01<00:00, 45.09it/s]


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
    Generated text:  Harry Potter, and I'm the boy with the famous wand that can call upon the powers of the Dark Arts. How do you like your family? Do you have any siblings? 
    
    If you were a character from a story or a book, who would you be? Would you be a wizard or a magic expert? How would you interact with your family? 
    
    As an AI language model, I don't have a family or a personal identity, so I can't provide an answer to your questions. However, I can generate a fictional response based on your request.
    
    As a wizard, I would have different siblings, each with a unique
    ===============================
    Prompt: The president of the United States is
    Generated text:  a rich man. He has many jobs and sometimes he becomes very busy. He has to work for many days and he does not have enough time to rest and enjoy his favorite things. In addition, the president has to deal with the many problems that the government is facing today. But you will be surprised to know that the president is also a good friend to people. He makes friends with people from all over the world. He often writes letters and sends messages to friends and family. He spends time with his family and friends. He enjoys doing so many things. The president is also a great leader. He makes important decisions. He is
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. Correct
    B. Incorrect
    Answer: A
    
    The correct period for car purchases is generally from ____. A. 3 years to 10 years B. 1 to 3 years C. 1 to 10 years D. 3 to 10 years
    Answer: D
    
    The concrete slump value of the test pieces is 35mm. Which of the following statements is correct? 
    A. The concrete was mixed using the manual method.
    B. The concrete was mixed using the air-dry method.
    C. The concrete was mixed using the rapid mix method.
    D. The
    ===============================
    Prompt: The future of AI is
    Generated text:  here. While we are still a long way from its full implementation, we already see the applications of AI in every field. We will see the Internet of Things, the development of 5G, and the development of robotics with self-driving cars. Not to mention, there are also more and more ways we can use AI to develop new industries and new markets. Can you imagine what the future of AI looks like? To answer this question, we must look at the results of the AI development in the past few decades, from the work of deep learning, the development of neural networks, to the appearance of large language models. The development


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am passionate about [reason for being at the company]. I am always looking for ways to [what I enjoy doing]. I am a [type of person] and I am always [what I strive for]. I am [what I am most proud of]. I am [what I am most proud of]. I am [what I am most proud of]. I am [what I am most proud of]. I am [what I am most proud of]. I am [what I am most proud of]. I am [what I am most proud of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. Its cuisine, including French cuisine, is also renowned worldwide. The city is known for its fashion industry, with many famous fashion houses and boutiques. Paris is also a center for art and culture, with many museums, galleries, and theaters. It is a city that is constantly evolving and changing,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to the needs of humans. This could lead to more sophisticated and personalized AI systems that can better understand and respond to the needs of their users.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use
    


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
    Generated text:  [Your Name]. I'm a [job title] and I love to [describe your role in your work]. In my free time, I enjoy [write what you do, including your hobbies or interests]. If you're ever in need of advice or assistance, I'm here to help. Thank you for taking the time to meet me. Let me know if you have any questions or if there's anything I can do to assist you! [Your Name] My name is [Your Name]. I'm a [job title] and I love to [describe your role in your work]. In my free time, I enjoy [write
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Roie," the "City of Light," and "La Rose," as it is a vibrant and lively city that is home to millions of people, including its famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
    
    Does this next sentence follow, given the preceding text?
    
    Paris is the largest city in France. 
    
    Options are: (A). yes. (B). no.
    A). yes.
    
    The statement "Paris is the largest city in France" is correct. The previous text states that Paris is "home to millions of people" and that it
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting and there are many possibilities for it to evolve. Here are a few possible trends that could play a role in shaping the future of AI:
    
    1. Increased integration of AI into common technologies: As we see more and more AI systems being used in everyday products, such as smart home devices and virtual assistants, we could see a more seamless integration of AI into the broader tech landscape. This might lead to even more advanced AI systems being used in our everyday lives.
    
    2. AI becomes more ethical and responsible: As we see more AI systems being developed and deployed, it's becoming more important for AI to be developed ethically and responsibly


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

     __

    __.

     My

     name

     is

     Alex

    .

     __

    __.

     


    A

    .

     My

     name

     is

     ______

    __,

     I

    'm

     ______

    __.

     My

     name

     is

     ______

    __

    _.

      


    B

    .

     My

     name

     is

     __

    ________

    ,

     I

    'm

     ______

    __.

      


    C

    .

     My

     name

     is

     __

    ________

    ,

     I

    'm

     ______

    __.

      


    D

    .

     My

     name

     is

     ______

    __,

     I

    'm

     ______

    __.

      


    E

    .

     My

     name

     is

     ______

    __,

     I

    'm

     ______

    __.

      


    F

    .

     My

     name

     is

     ______

    __,

     I

    'm

     ______

    __.

      


    G

    .

     My

     name

     is

     ______

    __,

     I

    'm

     ______

    __.

      


    H

    .

     My

     name

     is

     ______

    __,

     I

    'm

     ______

    __.

      


    I

    .

     My

     name

     is

     ______

    __,

     I

    'm

     ______

    __.

      


    J

    .

    
    
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

    .

     Its

     historical

     significance

     lies

     in

     its

     architecture

    ,

     arts

    ,

     and

     cultural

     institutions

    .

     It

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     and

     elegant

     atmosphere

    .

     Paris

     is

     the

     heart

     of

     France

    ,

     known

     for

     its

     various

     landmarks

    ,

     festivals

    ,

     and

     cultural

     events

    .

     It

     is

     also

     the

     birth

    place

     of

     some

     famous

     French

     literature

    ,

     music

    ,

     and

     art

    .

     France

    's

     capital

     city

     is

     located

     in

     the

     heart

     of

     the

     country

    ,

     and

     its

     legacy

     is

     widely

     recognized

     worldwide

    .

     It

     serves

     as

     the

     political

    ,

     cultural

    ,

     and

     economic

     center

     of

     France

    ,

     and

     has

     been

     a

     significant

     hub

     for

     the

     country

    's

     history

     and

     identity

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     combination

     of

     technological

     advancements

    ,

     changing

     societal

     needs

    ,

     and

     the

     continued

     advancement

     of

     research

     and

     development

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     could

     be

     expected

     to

     shape

     the

     future

     of

     the

     field

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

     likely

     to

     continue

     to

     be

     integrated

     with

     other

     technologies

    ,

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

     computer

     vision

    ,

     which

     will

     enable

     even

     more

     sophisticated

     and

     powerful

     applications

     of

     AI

    .

     This

     integration

     could

     lead

     to

     the

     development

     of

     new

     applications

     and

     technologies

     that

     will

     bring

     AI

     closer

     to

     the

     point

     where

     it

     can

     take

     over

     many

     aspects

     of

     human

     life

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

     concerns

    :

     As

    



```python
llm.shutdown()
```
