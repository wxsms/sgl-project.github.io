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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]


    2026-05-06 18:33:51,055 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 18:33:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.44 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.43 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.43 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.43 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.43 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.42 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.41 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.41 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=75.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.40 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.39 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.39 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.10 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.06 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=960 avail_mem=74.39 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s] Capturing num tokens (num_tokens=896 avail_mem=74.39 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.39 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=832 avail_mem=74.39 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=768 avail_mem=74.38 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=704 avail_mem=74.38 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=640 avail_mem=74.38 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=576 avail_mem=74.38 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=512 avail_mem=74.36 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=512 avail_mem=74.36 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=480 avail_mem=74.38 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=448 avail_mem=74.38 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=416 avail_mem=74.37 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=384 avail_mem=74.37 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.37 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=352 avail_mem=74.37 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=320 avail_mem=74.36 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=288 avail_mem=74.36 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=256 avail_mem=74.36 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=240 avail_mem=74.35 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=224 avail_mem=74.35 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=224 avail_mem=74.35 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.98it/s]Capturing num tokens (num_tokens=208 avail_mem=74.17 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.98it/s]Capturing num tokens (num_tokens=192 avail_mem=73.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.98it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=144 avail_mem=73.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=128 avail_mem=73.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.44it/s] Capturing num tokens (num_tokens=80 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=32 avail_mem=73.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]

    Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=8 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.41it/s] Capturing num tokens (num_tokens=4 avail_mem=73.21 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=4 avail_mem=73.21 GB): 100%|██████████| 58/58 [00:01<00:00, 41.59it/s]


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
    Generated text:  Leo. I'm a bookworm. I have a number of interests, but most importantly, I enjoy reading. I will be reading "Little Women" by Louisa May Alcott, "The Handmaid's Tale" by Rick Riordan, "Pride and Prejudice" by Jane Austen, "The Odyssey" by Homer, "The Hare Krishna Hymnal" by Swami Vivekananda, and "The Art of War" by Sun Tzu. Do you know any books that can help me cope with stress and anxiety? Yes, there are several books that can help you cope with stress and anxiety
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader who is chosen by the people. He or she is the head of the executive branch of the government. The president’s powers are generally considered the highest of all government officials. He or she is also the commander-in-chief of the armed forces of the United States. The president is the President of the United States. His office has only one term and he is re-elected. The president is also the commander-in-chief of the armed forces of the United States. The president is also considered to be the leader of the government in the United States. The president and the vice president are both appointed by the president and confirmed by the
    ===============================
    Prompt: The capital of France is
    Generated text:  the ____.
    A. Paris
    B. Lyon
    C. Bordeaux
    D. Nice
    Answer:
    
    A
    
    The capital of France is Paris. The capital of the United States is ______.
    A. Washington
    B. New York
    C. Los Angeles
    D. Houston
    Answer:
    
    A
    
    The capital of the United States is ______.
    A. Washington
    B. New York
    C. Los Angeles
    D. Houston
    Answer:
    
    A
    
    Among the following four options, which one is not an abbreviation for a railway bureau?
    A. CR
    B. NCR
    C. SD
    D. CB
    Answer:
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of scientists and engineers. While the latest tech titans have demonstrated how AI can power innovation and create new products, the potential impact of AI on society is vast. As AI continues to advance, we are seeing the rise of ethical concerns that need to be addressed. The rise of artificial intelligence has changed the landscape of the world and the business world. It has revolutionized the way we live our lives, work, and interact with each other. AI has brought about significant changes in the way that we work and communicate with each other. It has given us tools that are able to analyze complex data and generate insights that can help


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


    Generated text:  Paris, the city that was founded in 900 AD and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the largest city in the European Union. Paris is a cultural and historical center with a rich history and a vibrant nightlife. It is a popular tourist destination and a major economic hub. The city is known for its cuisine, fashion, and art. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has been a center of learning and culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies such as IoT, blockchain, and quantum computing. This integration will enable AI to perform tasks that are currently beyond the capabilities of any single technology.
    
    2. Enhanced privacy and security: As AI becomes more integrated with other technologies, there will be an increased need for privacy and security measures to protect user data. This will require advancements in AI that can better handle and analyze large amounts of data.
    
    3
    


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
    Generated text:  Emily. I'm a quiet, nerdy computer science major who enjoys reading sci-fi and fantasy books, spending time with my loyal dog, and helping out in the local community. What kind of person are you? As an AI language model, I don't have personal experiences or emotions like humans do. However, I can tell you that I've been trained to understand and respond to natural language, making me useful in a variety of applications and tasks. Can I help you with anything today? Or do you have a specific question or topic you'd like me to discuss? Let me know and I'll do my best to assist you.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the seat of the government of France. Paris is known for its historical landmarks, such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also known for its vibrant cultural scene, including the Notre-Dame Cathedral庙 and the Louvre Museum. Paris is a cosmopolitan city with a rich history and a diverse population. Its cultural significance is reflected in its numerous museums, museums, and art galleries. The city is also known for its fashion industry and the annual Eiffel Tower Festival. The city has a rich history and is a hub of culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be one of rapid advancement and innovation. Here are some potential trends that could shape the technology:
    
    1. Increased Integration of AI into Everyday Life: AI will become even more integrated into our daily lives, from self-driving cars and home security systems to virtual assistants and chatbots that can answer a wide range of questions.
    
    2. Enhanced Personalization: With the increasing use of AI, it is expected that we will see more personalized experiences, with AI being able to learn and adapt to our preferences and behavior.
    
    3. Improved Efficiency: AI will continue to improve efficiency in areas such as manufacturing, transportation, and logistics, leading to increased


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

     Sarah

    ,

     and

     I

    'm

     an

     intro

    verted

     and

     confident

     person

    .

     I

    'm

     a

     writer

     and

     I

     write

     for

     various

     platforms

     including

     the

     internet

    ,

     social

     media

    ,

     and

     websites

    .

     My

     background

     is

     in

     the

     creative

     field

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ideas

     and

     techniques

     to

     develop

     my

     skills

    .

     I

     love

     learning

     and

     trying

     new

     things

    ,

     and

     I

    'm

     always

     eager

     to

     improve

     myself

     and

     expand

     my

     knowledge

     in

     my

     field

    .

     I

    'm

     patient

     and

     dedicated

     to

     my

     work

    ,

     and

     I

     enjoy

     helping

     others

     find

     their

     own

     ways

     to

     achieve

     their

     goals

    .

     I

    'm

     a

     strong

     team

     player

     and

     a

     strong

     leader

    ,

     and

     I

     believe

     in

     the

     power

     of

     teamwork

     and

     collaboration

     to

     overcome

     any

     challenge

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     renowned

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     historical

     landmarks

    ,

     and

     diverse

     cultural

     scene

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

     the

     nation

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     cuisine

    ,

     while

     also

     being

     known

     for

     its

     fashion

    ,

     architecture

    ,

     and

     theater

    .

     It

     is

     an

     important

     hub

     of

     global

     trade

     and

     finance

    .

     Paris

     is

     also

     home

     to

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

     notable

     landmarks

    .

     The

     city

     is

     known

     for

     its

     romantic

    ,

     romantic

    ,

     romantic

     atmosphere

    ,

     and

     is

     considered

     a

     Paris

    ian

    ,

     Paris

    ian

    .

     The

     capital

     of

     France

     is

     Paris

    .

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     largely

     unknown

    .

     It

    's

     possible

     that

     AI

     will

     continue

     to

     evolve

     in

     a

     variety

     of

     interesting

     ways

    ,

     but

     we

     can

     expect

     to

     see

     significant

     advances

     in

     some

     areas

     such

     as

     natural

     language

     processing

    ,

     machine

     learning

    ,

     computer

     vision

    ,

     robotics

    ,

     and

     deep

     learning

    .

     
    


    AI

     has

     already

     revolution

    ized

     many

     industries

    ,

     including

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     manufacturing

    .

     It

    's

     also

     possible

     that

     AI

     will

     be

     used

     in

     completely

     new

     and

     unexpected

     ways

     in

     the

     future

    ,

     such

     as

     in

     personal

     assistants

    ,

     self

    -driving

     cars

    ,

     and

     virtual

     assistants

    .

     
    


    One

     trend

     that

     could

     be

     particularly

     significant

     is

     the

     development

     of

     AI

    -driven

     autonomous

     vehicles

    .

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     could

    



```python
llm.shutdown()
```
