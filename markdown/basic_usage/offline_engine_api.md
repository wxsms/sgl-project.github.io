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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]


    2026-05-02 01:38:06,005 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 01:38:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.28it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:08<00:03,  9.62it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:08<00:07,  4.05it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:08<00:02,  6.86it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:08<00:00, 10.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.99it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.75it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.75it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.63it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.21it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.21it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.21it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.21it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.21it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.21it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.34it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.34it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.34it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.34it/s]

    Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.78it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.75it/s]

    Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.75it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.13it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.13it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.13it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.13it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.13it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.13it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 47.56it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 41.55it/s]


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
    Generated text:  Henry White and I'm a lawyer by profession. I'm also a writer and poet who has written many books on various subjects. My books include "The Rise and Fall of the American Empire," "The Perfect Plague," "Achilles and the Sun God," "The Secret Lives of the Dead," and "The Perfect Fantasy." I also write poetry, including "The Hidden Facts," "The Great Lover," and "The Secrets of the Perfect Planet." My writing has been published in many magazines and anthologies. I write frequently for various websites including "Farewell to the Endless," "The Secret Lives of the Dead
    ===============================
    Prompt: The president of the United States is
    Generated text:  a highly elected official who represents the federal government of the United States. The president is the head of the executive branch of the government. He is responsible for implementing the laws of the federal government and the acts of the executive branch of the federal government, and for carrying out the duties of the president of the United States. The president is also the commander-in-chief of the armed forces of the United States.
    
    The president is elected by the members of the United States Congress and is not accountable to any one person. However, he is accountable to the American people for the actions of the government of the United States.
    
    One of the major duties of
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lyon
    C. Brest
    D. Bordeaux
    Answer:
    A
    
    Which of the following phenomena is a macroscopic application of Brownian motion?
    A. Dust motility
    B. Seed dispersal
    C. Formation of fog
    D. Formation of a dew
    Answer:
    A
    
    Which of the following is NOT a primary factor affecting economic growth?
    A. Technological innovation
    B. Natural resources
    C. Political stability
    D. Exchange rate
    Answer:
    C
    
    The landmark event that marked the complete victory of the Anti-Japanese War in 1945 was ______.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and holds great promise for the following industries. A. Advanced Manufacturing B. Retail C. Finance D. Construction
    Answer:
    
    ABCD
    
    Which of the following are important signs of the death of a patient? A. Absence of spontaneous breathing B. Absence of spontaneous circulation C. Pale complexion D. Breathing difficulty E. Refusal to be moved
    Answer:
    
    ABE
    
    Patient, male, 28 years old, married, has been suffering from upper abdominal pain and discomfort for the past year. In the past month, he has experienced frequent episodes of upper abdominal pain. The pain starts during fasting, lasting from


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about yourself]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [insert a short, positive, enthusiastic statement about your job]. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. What do you enjoy doing? I enjoy [insert a short,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also a popular tourist destination, with millions of visitors each year. The city is known for its fashion industry, art scene, and food culture. It is a major economic and political center in Europe, and has a rich history dating back to the Roman Empire. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective AI systems that can perform tasks that are currently beyond the capabilities of humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency,
    


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
    Generated text:  [Name], I'm a [Title] at [Company]. I'm excited to be here and contribute to your team. Feel free to ask me anything and I'll do my best to answer. What can I expect from my first interaction with you? This is my first interaction with you today and I hope you're excited to meet me. Please let me know how you'd like to start our conversation. I'm available 7 days a week to schedule meetings. How can I assist you today? I look forward to working with you and helping you achieve your goals. It's always my pleasure to help anyone who needs my services.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an international and historic city located in the center of the country, and it is the largest city in Europe. It is also known as the "City of Light" and the "City of Love." Paris is home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is a hub of culture, art, and commerce and is known for its rich history, beautiful architecture, and vibrant nightlife. It is also a major transportation hub, with a well-developed public transportation system and numerous airports. As the capital of France, Paris plays an important role in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be very exciting and diverse, with many possibilities for technological advancements and applications. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI-powered diagnostic tools, virtual assistants for medical consultations, and robots in surgery are expected to greatly improve the quality of care and reduce errors.
    
    2. AI in manufacturing: AI-driven robots and autonomous vehicles will revolutionize the manufacturing industry, enabling more efficient and sustainable production processes.
    
    3. AI in transportation: AI-powered autonomous vehicles, self-driving cars, and better traffic management systems will reduce accidents and improve efficiency.
    
    4. AI in finance: AI algorithms for risk assessment


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

    ]

     and

     I

     am

     a

     [

    character

     type

    ].

     My

     favorite

     [

    character

     trait

    ]

     is

     [

    X

    ].


    What

     are

     your

     [

    character

     trait

    ]

     and

     why

     do

     you

     think

     it

     is

     important

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     feelings

     or

     personal

     experiences

    ,

     but

     I

    'm

     designed

     to

     help

     answer

     your

     questions

     and

     provide

     helpful

     responses

    .

     So

    ,

     if

     you

     have

     any

     questions

     or

     need

     information

     on

     a

     particular

     topic

    ,

     feel

     free

     to

     ask

     me

    ,

     and

     I

    'll

     do

     my

     best

     to

     provide

     the

     information

     you

    're

     looking

     for

    .

     #

    I

    Chat

     #

    Tech

    Talk

    


    [

    Name

    ]

     [

    Character

     type

    ]

     [

    Character

     trait

    ]

     #

    F

    amous

    Person

     #

    AI

     #

    Chat

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Instructions

    :

     Replace

     the

     key

     terms

     "

    France

    ",

     "

    capital

     city

    ",

     and

     "

    Paris

    "

     with

     proper

     nouns

     and

     use

     bullet

     points

     to

     organize

     the

     information

    .

     "

    Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     in

     the

     center

     of

     the

     country

     and

     known

     for

     its

     stunning

     architecture

    ,

     world

    -ren

    owned

     museums

    ,

     and

     rich

     history

    .

     It

     is

     the

     second

    -largest

     city

     in

     France

    ,

     with

     a

     population

     of

     around

     

    2

    .

    5

     million

     people

    .

     Paris

     was

     founded

     by

     the

     Romans

     in

     the

     

    6

    th

     century

     and

     was

     initially

     a

     small

    ,

     isolated

     village

    .

     It

     was

     later

     a

     major

     trade

     center

    ,

     and

     in

     the

     

    1

    9

    th

     century

    ,

     it

     became

     a

     significant

     center

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     number

     of

     complex

     and

     interconnected

     factors

    ,

     including

     developments

     in

     computing

     technology

    ,

     machine

     learning

     algorithms

    ,

     and

     the

     growth

     of

     big

     data

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

     emerge

     in

     the

     coming

     years

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

     is

     increasingly

     being

     used

     in

     healthcare

     to

     help

     diagnose

     and

     treat

     diseases

    ,

     predict

     patient

     outcomes

    ,

     and

     improve

     patient

     care

    .

     This

     could

     include

     the

     use

     of

     machine

     learning

     algorithms

     to

     analyze

     medical

     images

    ,

     detect

     patterns

     in

     patient

     data

    ,

     and

     assist

     in

     the

     diagnosis

     and

     treatment

     of

     complex

     conditions

     such

     as

     cancer

    ,

     heart

     disease

    ,

     and

     Alzheimer

    's

     disease

    .
    


    2

    .

     Enhanced

     automation

    :

     AI

     is

     likely

     to

    



```python
llm.shutdown()
```
