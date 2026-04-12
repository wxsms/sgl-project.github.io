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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.38it/s]


    2026-04-12 06:17:50,702 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 06:17:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.00it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.00it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.00it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.00it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.00it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.00it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.00it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.00it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.62it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.62it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.62it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.62it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.62it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.62it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.62it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.63it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.63it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.63it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.63it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.63it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.63it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.63it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.88it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.88it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.33 GB):   9%|▊         | 5/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.33 GB):   9%|▊         | 5/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.43 GB):   9%|▊         | 5/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.43 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.54it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.00 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.98 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.48it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.48it/s] Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  50%|█████     | 29/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 40.55it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  50%|█████     | 29/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=288 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=224 avail_mem=118.92 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.03it/s]

    Capturing num tokens (num_tokens=224 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s]

    Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 39.20it/s]


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
    Generated text:  Stéphane. I’m a 22-year-old French musician, writer and photographer. I’m currently writing my first novel, a piece of autobiographical fiction. I also write short stories, poetry and prose. I like to publish a little every week. I like to experiment with new styles, techniques and language, and am always looking for new ways to use language. I’m also in the process of studying photography and I want to create my own photography exhibitions, some of them in collaboration with writers. I’m passionate about photography and I’m starting to embrace this more seriously. I’m also a translator. I’m currently
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to become more environmentally conscious. He decides to implement a recycling program in his city. If he follows a certain pattern, in how many ways can he assign a recycling bin to each resident so that there is exactly one resident who receives no recycling bin, and exactly one resident who receives more than one recycling bin?
    
    To solve the problem, let's denote the residents of the city by \( R_1, R_2, \ldots, R_n \). The president's recycling program follows the pattern that each resident has a unique number of recycling bins. Specifically, each resident \( R_i \) is assigned a unique number of
    ===============================
    Prompt: The capital of France is
    Generated text:  in:
    A. Paris
    B. The Hague
    C. Brussels
    D. Vienna
    Answer:
    
    A
    
    During the process of handling large amounts of wastewater, the method that should be chosen is:
    A. Water purification and sterilization
    B. Chemical coagulation and disinfection
    C. Filtration
    D. Direct discharge
    Answer:
    
    C
    
    Please, which of the following is the correct order of steps for executing a program?
    A. Compiler first, then interpreter
    B. Interpreter first, then compiler
    C. Compiler and interpreter simultaneously
    D. Compilation and execution simultaneously
    Answer:
    
    A
    
    Which of the
    ===============================
    Prompt: The future of AI is
    Generated text:  not linear or straight, but rather, it is more of a continuous journey that evolves over time. Artificial Intelligence (AI) is the ability of an artificial system to mimic, perform, or reproduce the human cognitive processes, with the goal of achieving certain tasks. AI can be broadly categorized into two main types - rule-based and neural network-based.
    Rule-based AI is a type of AI where the AI is designed to mimic human decision-making processes. The AI is programmed to execute tasks based on a set of rules that are defined by the developer. Rule-based AI systems can be used in various applications such as financial modeling, recommendation systems, and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry], and I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling metropolis with a rich history and culture, and is a popular tourist destination. It is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a city that has a unique blend of old and new, and is a must-visit destination for anyone interested in French culture and history. 
    
    The French capital is a vibrant and dynamic city, with a diverse
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in various industries, from manufacturing to healthcare to transportation. This will lead to increased efficiency, productivity, and cost savings for businesses and individuals alike.
    
    2. Improved privacy and security: As AI systems become more sophisticated, we can expect to see increased concerns about privacy and security. This will likely lead to more stringent regulations and standards for AI development
    


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
    Generated text:  [Your Name], and I'm here to introduce myself. I have a strong work ethic and a positive attitude, and I'm always looking for new challenges. I'm enthusiastic about learning new things and exploring different ideas, and I'm eager to make a positive impact in my community. I'm committed to using my skills and experiences to help others and contribute to a better world. Thank you. 
    
    (Note: Please make sure to adjust the name and pronoun as you see fit for the character, and adjust the tone and style of the self-introduction to fit your writing style.) "Hello, my name is [Your Name] and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as the "City of Love." It is known for its beautiful architecture, stunning cathedrals, and rich cultural heritage. Paris is the birthplace of many famous artists and intellectuals, such as Vincent van Gogh and Pablo Picasso. The city is also known for its annual Eiffel Tower celebration, which attracts millions of visitors annually.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities, both positive and negative. Some possible future trends in AI include:
    
    1. Improved accuracy: As technology continues to advance, we can expect to see greater accuracy in AI systems, as well as more precise predictions and forecasts.
    
    2. Increased integration: As AI systems become more complex, there will be an increased focus on integration and communication between them and other systems, to ensure that they work effectively together.
    
    3. Autonomous and intelligent agents: We can expect to see the emergence of autonomous and intelligent agents that can operate on their own without human intervention.
    
    4. More ethical considerations: As AI systems become more complex and sophisticated


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

    ].

     I

     am

     a

     [

    age

    ]

     year

     old

     human

    ,

     with

     a

     [

    national

    ity

    ]

     background

    .

     I

     grew

     up

     in

     [

    city

     or

     country

    ]

     and

     have

     always

     been

     passionate

     about

     [

    what

     you

     do

    ],

     whether

     it

     be

     [

    job

     title

    ],

     [

    m

    otto

    ],

     or

     [

    personal

     interest

    ].

     I

     have

     a

     natural

     talent

     for

     [

    the

     skill

     or

     hobby

     you

     excel

     in

    ],

     and

     I

    'm

     always

     striving

     to

     push

     the

     boundaries

     of

     what

    's

     possible

    .

     I

     believe

     in

     hard

     work

    ,

     perseverance

    ,

     and

     a

     love

     for

     learning

    ,

     and

     I

    'm

     always

     eager

     to

     share

     my

     knowledge

     and

     insights

     with

     anyone

     who

     will

     listen

    .

     I

     love

     [

    what

     you

     do

    ]

     because

     it

     allows

     me

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     

    1

    9

    th

     largest

     city

     in

     the

     world

     by

     population

     and

     the

     largest

     French

     metropolitan

     area

    .

     It

     is

     known

     as

     "

    la

     ville

     du

     village

    "

     (

    the

     village

     city

    )

     due

     to

     its

     small

     size

     and

     rural

     character

    ,

     and

     is

     home

     to

     many

     world

    -ren

    owned

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

     the

     Notre

    -D

    ame

     Cathedral

    .

     France

    's

     capital

     city

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     known

     for

     its

     vibrant

     arts

     scene

    ,

     festivals

    ,

     and

     historic

     architecture

    .

     It

     is

     also

     an

     important

     cultural

     and

     economic

     center

     for

     the

     country

     and

     a

     symbol

     of

     France

    's

     unity

     and

     identity

    .

     Paris

     is

     the

     world

    's

     most

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     advances

     in

     areas

     such

     as

     machine

     learning

    ,

     deep

     learning

    ,

     and

     natural

     language

     processing

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

     that

     could

     shape

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     technology

    :
    


    1

    .

     Increased

     AI

     integration

     with

     human

     AI

    :

     As

     AI

     continues

     to

     become

     more

     sophisticated

    ,

     we

     can

     expect

     to

     see

     a

     greater

     integration

     of

     AI

     with

     human

     AI

     systems

    .

     This

     could

     lead

     to

     a

     more

     seamless

     and

     intuitive

     user

     experience

    ,

     as

     the

     two

     systems

     can

     work

     together

     to

     perform

     tasks

     that

     were

     once

     performed

     by

     a

     human

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     we

     can

     expect

     to

     see

     a

     greater

    



```python
llm.shutdown()
```
