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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.95it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.13it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.13it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.13it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.13it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.13it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.13it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.13it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.13it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.18it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.18it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.18it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.18it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.18it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.18it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.18it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 22.70it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 17.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 17.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 17.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.37 GB):   9%|▊         | 5/58 [00:00<00:02, 17.91it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.37 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.21it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.10 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.12 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.12 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.35 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.34 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.12it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.32 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.20 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.31 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.14it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.31 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.20 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.28 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.29 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.29 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.29 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.27 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.74it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.28 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.74it/s] Capturing num tokens (num_tokens=960 avail_mem=74.28 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.34it/s]Capturing num tokens (num_tokens=896 avail_mem=74.27 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.34it/s]Capturing num tokens (num_tokens=832 avail_mem=74.27 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.34it/s]Capturing num tokens (num_tokens=768 avail_mem=74.26 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.34it/s]Capturing num tokens (num_tokens=768 avail_mem=74.26 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.91it/s]Capturing num tokens (num_tokens=704 avail_mem=74.25 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.91it/s]Capturing num tokens (num_tokens=640 avail_mem=74.25 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.91it/s]Capturing num tokens (num_tokens=576 avail_mem=74.24 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.91it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.21 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.91it/s]Capturing num tokens (num_tokens=512 avail_mem=74.21 GB):  50%|█████     | 29/58 [00:01<00:01, 25.57it/s]Capturing num tokens (num_tokens=480 avail_mem=74.24 GB):  50%|█████     | 29/58 [00:01<00:01, 25.57it/s]Capturing num tokens (num_tokens=448 avail_mem=74.24 GB):  50%|█████     | 29/58 [00:01<00:01, 25.57it/s]Capturing num tokens (num_tokens=416 avail_mem=74.23 GB):  50%|█████     | 29/58 [00:01<00:01, 25.57it/s]Capturing num tokens (num_tokens=384 avail_mem=74.22 GB):  50%|█████     | 29/58 [00:01<00:01, 25.57it/s]Capturing num tokens (num_tokens=384 avail_mem=74.22 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.54it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.54it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.54it/s]Capturing num tokens (num_tokens=288 avail_mem=74.20 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.54it/s]Capturing num tokens (num_tokens=288 avail_mem=74.20 GB):  62%|██████▏   | 36/58 [00:01<00:01, 17.49it/s]Capturing num tokens (num_tokens=256 avail_mem=74.20 GB):  62%|██████▏   | 36/58 [00:01<00:01, 17.49it/s]Capturing num tokens (num_tokens=240 avail_mem=74.18 GB):  62%|██████▏   | 36/58 [00:01<00:01, 17.49it/s]Capturing num tokens (num_tokens=224 avail_mem=74.19 GB):  62%|██████▏   | 36/58 [00:01<00:01, 17.49it/s]Capturing num tokens (num_tokens=208 avail_mem=74.19 GB):  62%|██████▏   | 36/58 [00:01<00:01, 17.49it/s]Capturing num tokens (num_tokens=192 avail_mem=74.18 GB):  62%|██████▏   | 36/58 [00:01<00:01, 17.49it/s]Capturing num tokens (num_tokens=192 avail_mem=74.18 GB):  71%|███████   | 41/58 [00:01<00:00, 22.51it/s]Capturing num tokens (num_tokens=176 avail_mem=74.18 GB):  71%|███████   | 41/58 [00:01<00:00, 22.51it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.17 GB):  71%|███████   | 41/58 [00:02<00:00, 22.51it/s]Capturing num tokens (num_tokens=144 avail_mem=74.16 GB):  71%|███████   | 41/58 [00:02<00:00, 22.51it/s]Capturing num tokens (num_tokens=128 avail_mem=74.16 GB):  71%|███████   | 41/58 [00:02<00:00, 22.51it/s]Capturing num tokens (num_tokens=128 avail_mem=74.16 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.98it/s]Capturing num tokens (num_tokens=112 avail_mem=74.15 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.98it/s]Capturing num tokens (num_tokens=96 avail_mem=74.15 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.98it/s] Capturing num tokens (num_tokens=80 avail_mem=74.14 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.98it/s]Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.98it/s]Capturing num tokens (num_tokens=48 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.98it/s]Capturing num tokens (num_tokens=48 avail_mem=74.13 GB):  86%|████████▌ | 50/58 [00:02<00:00, 29.84it/s]Capturing num tokens (num_tokens=32 avail_mem=74.10 GB):  86%|████████▌ | 50/58 [00:02<00:00, 29.84it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.11 GB):  86%|████████▌ | 50/58 [00:02<00:00, 29.84it/s]Capturing num tokens (num_tokens=24 avail_mem=74.11 GB):  86%|████████▌ | 50/58 [00:02<00:00, 29.84it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  86%|████████▌ | 50/58 [00:02<00:00, 29.84it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.23it/s]Capturing num tokens (num_tokens=16 avail_mem=74.10 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.23it/s]Capturing num tokens (num_tokens=12 avail_mem=74.09 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.23it/s]Capturing num tokens (num_tokens=8 avail_mem=74.08 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.23it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.08 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.23it/s]Capturing num tokens (num_tokens=4 avail_mem=74.08 GB): 100%|██████████| 58/58 [00:02<00:00, 29.88it/s]Capturing num tokens (num_tokens=4 avail_mem=74.08 GB): 100%|██████████| 58/58 [00:02<00:00, 23.32it/s]


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
    Generated text:  Jonathan. I am from Liverpool, England. I have been a member of the Royal Academy of Dramatic Art for 18 years and I have performed in more than 400 plays. I have received an Academy Award for Best Actor and have won numerous prizes for my work in the theatre. I have acted in many of Shakespeare's plays and have performed with numerous other world-class actors including Richard Harris, Ian McKellen and Laurence Olivier. I have acted in more than 20 plays for the Royal Shakespeare Company and have also performed in the West End at the Theatre Royal, Drury Lane. I have also performed in
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a small island nation, which has a budget of $100,000 for their annual visit. The president decides to spend 20% of the budget on food, 30% on transportation, and 10% on healthcare. What is the amount of money left in the budget for other purposes? To determine the amount of money left in the budget for other purposes, we first need to calculate the total percentage of the budget allocated to these three categories.
    
    The president's budget is $100,000, and we will calculate the amount allocated to each category as a percentage of the
    ===============================
    Prompt: The capital of France is
    Generated text: 
    
    A. Paris  
    B. Nice  
    C. Barcelona  
    D. Madrid
    
    To determine the capital of France, we need to recall the correct answer choices provided. Let's go through the options step by step:
    
    A. Paris - Paris is the capital of France.
    B. Nice - Nice is the capital of France.
    C. Barcelona - Barcelona is not the capital of France.
    D. Madrid - Madrid is the capital of Spain, not France.
    
    Since the correct answer is Paris, the correct choice is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  one of convergence.
    
    And while these technologies can have a wide range of consequences, they are inherently beneficial and can provide vast benefits to society as a whole.
    
    The world is now getting in sync with every device, service, and machine. It's a world where everything is connected, including your smartphone or your computer. Artificial intelligence, or AI, is one of the key technologies that are at the forefront of this convergence.
    
    This convergence of technology has already brought a number of benefits to society, but there are also a number of challenges that we need to be aware of as we move forward.
    
    The following are some of the ways that AI is


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also a major center for art, culture, and politics in Europe. Paris is home to many world-renowned museums, including the Louvre and the Musée d'Orsay, and is a popular tourist destination for its rich history, beautiful architecture, and vibrant culture. The city is also known for its cuisine, including its famous croissants and its traditional French dishes.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to automate many of the tasks that are currently performed by humans. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. Improved privacy and security: As AI becomes more advanced, it is likely to require more data to function effectively. This could lead to increased privacy concerns, as AI systems may be able to access and analyze
    


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
    Generated text:  [Name], and I'm a [job or occupation] with a passion for [thing]. What brings you to this world?
    Hello, my name is [Name], and I'm a [job or occupation] with a passion for [thing]. What brings you to this world? [Name] is a bright, curious, and always looking for knowledge person. What do you enjoy doing in your free time? [Name] enjoys reading, spending time outdoors, and exploring new cultures. What inspires you? [Name] is inspired by the stories of great leaders, the beauty of the natural world, and the importance of preserving the environment
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city and the second-most populous metropolitan area in the European Union and the most populous European city. It is also the world’s most populous city by both population and area, and is the largest metropolitan area in the world by land area. It is the heart of France’s cultural, artistic, and intellectual life, and is also the historical center of modern France. It is home to some of the world’s most famous landmarks, including the Eiffel Tower, the Louvre Museum, and Notre Dame Cathedral. It is also the home of the French language and its cultural institutions. Paris is an important economic and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some potential trends that are often discussed:
    
    1. Increased automation and digitalization: As AI becomes more widespread, it is likely to automate many tasks that are currently done manually, leading to increased automation and digitalization.
    
    2. AI in healthcare: AI is already being used in medical diagnosis and treatment, and it is likely to become more integrated into healthcare practices in the future.
    
    3. AI in finance: AI is already being used to analyze financial data, but it is likely to become even more integrated into finance and investment strategies in the future.
    
    4. AI in education: AI is already being used to personalize


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

    ],

     and

     I

     am

     a

     [

    job

     title

    ]

     at

     [

    Company

     Name

    ].

     I

     am

     a

     [

    number

    ]

     year

     old

     professional

     with

     [

    number

    ]

     years

     of

     experience

     in

     [

    industry

    ].

     I

     am

     passionate

     about

     [

    your

     profession

     or

     career

    ]

     and

     I

     am

     always

     eager

     to

     learn

     and

     improve

    .

     I

     am

     patient

    ,

     organized

    ,

     and

     have

     a

     strong

     work

     ethic

    .

     I

     enjoy

     helping

     others

     and

     I

     am

     always

     willing

     to

     lend

     a

     hand

     when

     needed

    .

     I

     am

     [

    number

    ]

     years

     of

     experience

     in

     [

    professional

     field

    ],

     and

     I

     am

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     knowledge

    .

     I

     am

     [

    number

    ]

     years

     old

    ,

     [

    number

    ]

     feet

     tall

    ,

     and

    
    
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

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .

     
    


    This

     statement

     is

     concise

     and

     provides

     a

     factual

     overview

     of

     the

     capital

     city

    's

     iconic

     landmarks

    .

     It

     highlights

     the

     significance

     of

     Paris

     to

     French

     culture

     and

     history

    ,

     and

     mentions

     a

     specific

     landmark

     that

     is

     a

     must

    -

    see

     when

     visiting

     the

     city

    .

     The

     statement

     is

     concise

     and

     to

     the

     point

    ,

     making

     it

     easy

     to

     understand

     and

     remember

    .

     
    


    However

    ,

     the

     statement

     could

     be

     expanded

     to

     include

     additional

     facts

     about

     Paris

    ,

     such

     as

     its

     language

    ,

     cuisine

    ,

     or

     fashion

     scene

    ,

     to

     make

     it

     more

     informative

     and

     engaging

     for

     readers

    .

     For

     example

    :

     
    


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     one

     of

     unprecedented

     growth

    ,

     innovation

    ,

     and

     integration

     across

     various

     domains

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     AI

    -driven

     automation

    :

     AI

     is

     expected

     to

     continue

     becoming

     more

     sophisticated

    ,

     and

     its

     ability

     to

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

     is

     expected

     to

     increase

    .

     This

     includes

     automation

     of

     repetitive

     and

     mundane

     tasks

    ,

     as

     well

     as

     personalized

     and

     context

    -aware

     customer

     service

    .
    


    2

    .

     Enhanced

     AI

     ethics

     and

     privacy

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

     expected

     to

     raise

     new

     ethical

     and

     privacy

     concerns

    .

     As

     such

    ,

     there

     is

     a

     need

     for

     continued

     development

     of

     AI

     ethics

     and

     privacy

     standards

    .
    


    3

    .

     AI

    -driven

     healthcare

     advancements

    



```python
llm.shutdown()
```
