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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.60it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.59it/s]


    2026-05-02 02:08:47,855 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 02:08:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<06:13,  6.55s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<06:13,  6.55s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:06<06:13,  6.55s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:06<06:13,  6.55s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:06<06:13,  6.55s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:52,  1.00it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:52,  1.00it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:06<00:52,  1.00it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:06<00:52,  1.00it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:06<00:52,  1.00it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:06<00:52,  1.00it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:06<00:52,  1.00it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:06<00:52,  1.00it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:06<00:15,  3.03it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:06<00:15,  3.03it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:06<00:05,  6.92it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:02, 10.89it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:02, 10.89it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:02, 10.89it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:07<00:01, 16.53it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:07<00:00, 24.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 16.18it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 16.18it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.18it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 17.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 17.86it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 17.86it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 17.86it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.46it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.37it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.37it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.17it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.49it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.95it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 39.16it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 39.16it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 39.16it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 39.16it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 39.16it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 39.16it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.46it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.52it/s]

    Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.52it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.25it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.66it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 36.37it/s]


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
    Generated text:  Sarah and I’m a writer from the UK. I have a story called “The Year of the Reluctant Keeper” that I’m currently working on. The book is about a woman who wakes up and realizes she’s been meant to be a keeper for a long time. She’s just not sure if she wants to be a keeper. She’s left all her bonds and loves in order to be a keeper. 
    
    I hope you get a chance to read it. I’m really excited to be able to write it. Is there a particular book you’ve read recently that you think would be a great fit for my story? I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a public official who is nominated and elected by the people of the United States and serves a term of four years.
    A. 正确
    B. 错误
    Answer:
    
    A
    
    The Constitution of the People's Republic of China stipulates that the state respects and protects human rights.
    A. 正确
    B. 错误
    Answer:
    
    A
    
    The term of office for the President of the People's Republic of China is four years, and he can be re-elected for up to two terms.
    A. 正确
    B. 错误
    Answer:
    
    A
    
    The Constitution of the People's Republic
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Rome
    D. Washington
    Answer:
    A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Rome
    D. Washington
    Answer:
    A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Rome
    D. Washington
    Answer:
    A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Rome
    D. Washington
    Answer:
    A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Rome
    D. Washington
    Answer:
    
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but its potential benefits are wide-ranging. Here’s a look at how AI can be deployed in your organization to drive digital transformation.
    
    As an AI technology expert, I’m always looking for new ways to engage with the new wave of AI and how it’s transforming every industry. I’ve been excited to see how AI is transforming the way we work and the impact it has on society. One of the ways AI is transforming the way we work is by automating repetitive and time-consuming tasks, but it also has the potential to transform the way we do business and work. Here are some ways AI can be deployed in your organization to


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


    Generated text:  [Name] and I'm a [occupation] with [number] years of experience in [field]. I'm a [character trait] and I'm always ready to learn and grow. I'm [character trait] and I'm always ready to help others. I'm [character trait] and I'm always ready to make a positive impact on the world. I'm [character trait] and I'm always ready to take on new challenges. I'm [character trait] and I'm always ready to be a good friend. I'm [character trait] and I'm always ready to be a good listener. I'm [character
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism, making it a popular destination for tourists and locals alike. The city is known for its annual festivals and events, including the Eiffel Tower Parade and the Paris Fashion Week. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to continue to be used for a wide range of applications, from improving healthcare outcomes to enhancing customer service and financial decision-making. As AI continues to evolve, it is likely to have a significant impact on society, and we can expect to see a range of new challenges and opportunities emerge. However, it is important to note that the potential benefits of AI are often outweighed by the
    


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
    Generated text:  [name] and I'm a [character's occupation or background]. I'm a [insert the number of years since graduation, which is a clue to your age, or a small integer that represents the length of your character's lifespan]. I'm always eager to learn and grow, and I love to challenge myself in various ways, whether it's through writing, speaking, or participating in activities that make me feel alive and engaged. 
    
    Let's start with our conversation. What would you like to talk about today? Let's make it fun and light-hearted. If you're not sure what to talk about, you can always tell me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
    Paris, officially the Île-de-France region of the French Department of Paris, is the capital city of the country of France. It is the largest city in France, with a population of about 11 million people. The city has a rich history dating back to ancient times, with its ancient city walls, including the Grande Arche, dating back to the 13th century. It is also home to numerous museums and galleries, as well as numerous churches and cathedrals. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating and constantly evolving. Here are some potential trends that could shape the AI landscape in the coming years:
    
    1. AI will become more natural and human-like - AI systems will continue to learn from and interact with humans, improving their performance and accuracy. This could lead to more natural interactions between people and AI, as well as more human-like decision-making.
    
    2. AI will be more integrated into everyday life - AI systems will become more integrated into our lives, from our phones and computers to smart home devices. This could lead to more efficient and convenient ways of using technology, as well as new applications and services that don't require physical interaction


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

     am

     [

    Age

    ].

     I

     am

     a

     [

    field

     or

     occupation

    ]

     with

     [

    relevant

     experience

     or

     education

    ].

     I

     enjoy

     [

    interest

     or

     hobby

    ]

     and

     strive

     to

     [

    describe

     my

     goal

     or

     purpose

     in

     life

    ].

     I

     am

     someone

     who

     [

    describe

     your

     personality

     type

     or

     traits

    ].

     If

     you

     could

     describe

     me

     in

     one

     sentence

    ,

     what

     would

     you

     say

    ?

     [

    Name

    ]

     [

    Tell

     us

     a

     bit

     about

     yourself

    .

     What

     do

     you

     like

     to

     do

    ?

     What

     do

     you

     hope

     to

     achieve

    ?

     What

     makes

     you

     unique

    ?

     What

     is

     your

     passion

     for

    ?

     What

     makes

     you

     a

     good

     listener

    ?]

     [

    Name

    ]

     [

    Tell

     us

     about

     yourself

    .

     What

     do

     you

     like

     to

     do

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

    ,

     with

     a

     population

     of

     over

     

    2

     million

     people

    ,

     making

     it

     the

     capital

     of

     the

     European

     Union

     and

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    .

     It

     is

     home

     to

     many

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

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     culture

     and

     cuisine

    ,

     with

     many

     restaurants

    ,

     cafes

    ,

     and

     theaters

     catering

     to

     visitors

     and

     locals

     alike

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     vibrant

     culture

     and

     a

     world

    -ren

    owned

     cuisine

    ,

     making

     it

     an

     exciting

     destination

     for

     tourists

     and

     locals

     alike

    .

     The

     city

     has

     undergone

     several

     changes

     and

     transformations

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     a

     diverse

     and

     rapidly

     evolving

     field

    ,

     with

     many

     new

     possibilities

     and

     challenges

     waiting

     to

     be

     explored

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     integration

     with

     human

     decision

    -making

    :

     AI

     systems

     will

     become

     more

     integrated

     with

     human

     decision

    -making

     processes

    ,

     allowing

     for

     more

     complex

     and

     nuanced

     decision

    -making

    .

     This

     could

     lead

     to

     more

     ethical

     and

     sustainable

     decision

    -making

     in

     areas

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     public

     policy

    .
    


    2

    .

     Autonomous

     robots

     and

     drones

    :

     Robots

     and

     drones

     with

     artificial

     intelligence

     will

     become

     more

     prevalent

     in

     areas

     such

     as

     manufacturing

    ,

     construction

    ,

     and

     logistics

    .

     They

     will

     be

     able

     to

     perform

     tasks

     with

     higher

     precision

     and

     efficiency

     than

     human

     workers

    ,

     but

     will

     also

    



```python
llm.shutdown()
```
