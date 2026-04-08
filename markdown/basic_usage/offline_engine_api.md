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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]


    2026-04-08 20:00:23,818 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 20:00:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.30it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.30it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.30it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.28it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.28it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.28it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.28it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.28it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.28it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.28it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.28it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.58it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.58it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.58it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.58it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.58it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.58it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.58it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.70it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.70it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.70it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.70it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.70it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.70it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.70it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=132.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=132.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=132.39 GB):   3%|▎         | 2/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=7168 avail_mem=132.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=132.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=6144 avail_mem=132.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.71it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=132.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=132.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=132.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=132.37 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=132.37 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=132.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=132.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=132.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=132.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=132.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=132.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=132.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=132.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=132.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=132.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=132.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=132.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.24it/s]Capturing num tokens (num_tokens=1536 avail_mem=132.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=132.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=132.32 GB):  31%|███       | 18/58 [00:00<00:01, 35.24it/s]

    Capturing num tokens (num_tokens=960 avail_mem=132.33 GB):  31%|███       | 18/58 [00:00<00:01, 35.24it/s] Capturing num tokens (num_tokens=896 avail_mem=132.33 GB):  31%|███       | 18/58 [00:00<00:01, 35.24it/s]Capturing num tokens (num_tokens=896 avail_mem=132.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=832 avail_mem=132.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=768 avail_mem=132.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=704 avail_mem=132.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=640 avail_mem=132.31 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=576 avail_mem=132.31 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=576 avail_mem=132.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=512 avail_mem=132.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=480 avail_mem=132.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.34it/s]

    Capturing num tokens (num_tokens=448 avail_mem=132.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=416 avail_mem=132.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=384 avail_mem=132.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=384 avail_mem=132.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=352 avail_mem=132.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=320 avail_mem=132.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=288 avail_mem=132.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=256 avail_mem=132.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=240 avail_mem=132.29 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=240 avail_mem=132.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=224 avail_mem=132.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]

    Capturing num tokens (num_tokens=208 avail_mem=132.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=192 avail_mem=132.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=176 avail_mem=132.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=160 avail_mem=132.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=160 avail_mem=132.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=144 avail_mem=132.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=128 avail_mem=132.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=112 avail_mem=132.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=96 avail_mem=132.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.57it/s] Capturing num tokens (num_tokens=80 avail_mem=132.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.57it/s]

    Capturing num tokens (num_tokens=80 avail_mem=132.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=64 avail_mem=132.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=48 avail_mem=132.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=32 avail_mem=132.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=28 avail_mem=132.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=24 avail_mem=132.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=24 avail_mem=132.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=20 avail_mem=132.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=16 avail_mem=132.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=12 avail_mem=132.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=8 avail_mem=132.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.08it/s] Capturing num tokens (num_tokens=4 avail_mem=132.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.08it/s]

    Capturing num tokens (num_tokens=4 avail_mem=132.23 GB): 100%|██████████| 58/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=4 avail_mem=132.23 GB): 100%|██████████| 58/58 [00:01<00:00, 39.46it/s]


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
    Generated text:  John, a man who has a strong interest in history and archaeology. I am interested in learning more about different cultures and their unique stories, traditions, and behaviors. I believe that learning about these cultures can help me better understand myself and others. Can you provide me with some useful information on historical and cultural topics that I may find helpful? Certainly! There are many fascinating topics in history and archaeology that you can explore and learn more about. Here are some general topics that you may find helpful:
    
    1. Ancient Civilizations: This includes topics such as the rise and fall of civilizations like Egypt, Greece, and Rome, and the
    ===============================
    Prompt: The president of the United States is
    Generated text:  now trying to decide who will be the next leader of the United States. He has two candidates, Candidate A and Candidate B. He has been given a specific set of criteria that must be met for a candidate to be elected. The criteria are as follows:
    - Candidate A must be a citizen of the United States and born in the year 1990 or later.
    - Candidate A must have a degree in computer science, with a minimum GPA of 3.8.
    - Candidate B must be a citizen of the United States and born in the year 1990 or later.
    - Candidate B must have a degree
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. Lille
    C. Bordeaux
    D. Lyon
    Answer:
    
    A
    
    As the main body of the construction project, the construction entity is ____.
    A. Project Owner
    B. Supplier
    C. Designer
    D. Supervision Unit
    Answer:
    
    A
    
    Which of the following is NOT a correct criterion for classifying elderly care products?
    A. Product structure
    B. Product life cycle
    C. Product composition
    D. Product price
    Answer:
    
    D
    
    The entire process of dealing with issues concerning the establishment and implementation of construction project contracts, as well as the management and supervision of construction
    ===============================
    Prompt: The future of AI is
    Generated text:  huge, but in the past few years, we've seen some big setbacks. Some researchers have pointed out that this is largely due to overreliance on algorithms, which can be biased. Others have pointed out that AI is inherently flawed and can be very biased in terms of fairness. 
    
    However, according to a recent study, a group of researchers from the University of Technology, Sydney, has taken a more nuanced approach to address these issues. In their study, they used a new type of algorithm called "unconfoundable adversarial attack" to create a fair algorithm.
    
    Unconfoundable adversarial attacks are a type of


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, fashion, and cuisine, and is home to many world-renowned museums, theaters, and other cultural institutions. The city is also known for its diverse population, including many ethnic groups and a large immigrant community. Paris is a major hub for business, finance, and tourism, and is a popular destination for visitors from around the world. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, healthcare, transportation, and finance. Automation will likely lead to increased efficiency, cost savings, and productivity gains.
    
    2. AI ethics and privacy concerns: As AI becomes more integrated into our daily lives, there will be increasing concerns about its ethical implications and potential privacy risks. There will be a need for regulations and guidelines to ensure that AI is
    


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
    Generated text:  Sarah, and I'm a professional blogger. I'm a huge fan of travel and have been documenting my travels for years, sharing my adventures with others. I love learning about new places and cultures, and I'm always on the lookout for interesting experiences to share with my readers. Whether it's a new city, a local market, or a cultural event, I'm always excited to see what I can share with you. Thank you for having me. This short self-introduction is neutral and does not contain any personal information about the character. The intro is concise and to the point, allowing for a natural conversation to begin. The tone
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the "City of Light" and "The City of Light." It is a city of rich history and culture, known for its iconic architecture, world-famous museums, and vibrant nightlife. Paris is a global center of politics, art, and culture, with numerous cultural institutions, including the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral. The French government, led by President Emmanuel Macron, has recently focused on modernizing the city and improving its infrastructure. Paris is an important part of French identity and is known for its festivals, food, and entertainment. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  complex and multifaceted, with many potential trends and advancements to consider. Some possible trends and developments include:
    
    1. Increased focus on ethical AI: As concerns about AI become more widespread and ethical issues arise, there will be a growing push towards developing AI that is more transparent, accountable, and aligned with societal values.
    
    2. Integration of AI into other areas of technology: As AI becomes more integrated into other areas of technology, such as healthcare, finance, and transportation, we may see greater adoption and development of AI solutions.
    
    3. Expansion of AI in the workforce: AI is already having a significant impact on the workplace, with applications


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

    /an

     [

    occupation

    ].

     I

    'm

     in

     my

     [

    age

    ]

     years

     old

    ,

     and

     I

    'm

     [

    a

     skill

     or

     hobby

    ].

     I

     love

     [

    my

     profession

     or

     hobby

    ],

     and

     I

     enjoy

     [

    activities

     I

     enjoy

     doing

    ].

     I

     like

     to

     [

    d

    ile

    mma

     I

     want

     to

     address

    ].

     I

    'm

     a

    /an

     [

    occupation

    ]

     who

     is

     always

     [

    good

     at

     something

    ].

     I

     love

     to

     [

    ad

    venture

     we

    're

     currently

     on

    ]

     with

     [

    other

     person

    ],

     and

     I

     want

     to

     do

     something

     [

    project

     or

     mission

    ].

     I

    'm

     a

    /an

     [

    occupation

    ],

     and

     I

     am

     always

     [

    good

     at

     something

    ].

     I

     love

     to

     [

    ad

    venture

     we

    're

     currently

     on

    ],

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     historical

     and

     cultural

     center

     with

     a

     rich

     history

     of

     art

    ,

     architecture

    ,

     and

     literature

    .

     The

     city

     has

     been

     the

     seat

     of

     government

    ,

     government

    ,

     and

     legislature

     of

     France

     since

     

    1

    2

    5

    6

    ,

     and

     is

     known

     for

     its

     lively

     and

     diverse

     culture

    .

     Paris

     is

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

     the

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     art

     galleries

    ,

     and

     has

     a

     vibrant

     music

     and

     dance

     scene

    .

     Paris

     is

     a

     city

     that

     has

     a

     long

     and

     stor

    ied

     history

    ,

     and

     is

     a

     world

    -ren

    owned

     cultural

     and

     artistic

     center

    .

     It

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     the

     development

     of

     more

     advanced

     and

     versatile

     models

     that

     can

     better

     understand

     and

     interpret

     human

     language

    .

     This

     may

     include

     the

     development

     of

     more

     sophisticated

     algorithms

     that

     can

     identify

     and

     analyze

     complex

     patterns

     in

     large

     amounts

     of

     data

    ,

     as

     well

     as

     the

     integration

     of

     machine

     learning

     techniques

     with

     human

     expertise

     to

     improve

     the

     accuracy

     of

     predictions

     and

     recommendations

    .

     Additionally

    ,

     AI

     is

     likely

     to

     become

     increasingly

     integrated

     with

     other

     technologies

    ,

     such

     as

     the

     Internet

     of

     Things

     (

    Io

    T

    )

     and

     the

     Internet

     of

     Things

     (

    Io

    T

    )

     and

     the

     Internet

     of

     Things

     (

    Io

    T

    )

     and

     the

     Internet

     of

     Things

     (

    Io

    T

    )

     and

     the

     Internet

     of

     Things

     (

    Io

    T

    ),

     to

     create

     a

     more

    



```python
llm.shutdown()
```
