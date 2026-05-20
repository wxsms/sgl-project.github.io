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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.47it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.46it/s]


    2026-05-20 17:14:26,768 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 17:14:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:47,  3.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:47,  3.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:47,  3.98s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:47,  3.98s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:47,  3.98s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.86it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.86it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 12.82it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 12.82it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 12.82it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 12.82it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 12.82it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 12.82it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 12.82it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.26it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.26it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.26it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.26it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.26it/s]

    Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.26it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.26it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.26it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 23.56it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 23.56it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 23.56it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 23.56it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 23.56it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 23.56it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 23.56it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 23.56it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 30.11it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 30.11it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 30.11it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 30.11it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 30.11it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 30.11it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 30.11it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 30.11it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 36.67it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 36.67it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 36.67it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 36.67it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 36.67it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 36.67it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 36.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.18it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.11it/s]Capturing num tokens (num_tokens=960 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.11it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.11it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.11it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=768 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.98it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.98it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.98it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.98it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.93it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.39it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 41.91it/s]


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
    Generated text:  Oscar and I am a dreamer.
    As a dreamer, I’m always looking for opportunities to create and be creative, and I believe that the future is bright for everyone.
    I am passionate about helping people discover their potential and pursuing their dreams. I believe that everyone deserves a chance at success and that with hard work and dedication, it is possible for anyone to achieve their goals.
    I believe that education is key to unlocking one's potential and that it is important to learn from mistakes and use them as opportunities to grow and learn.
    I am committed to helping people succeed and are always looking for ways to support them on their journey to success
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to go to war with China. He has a certain probability of success. If he goes to war, there is a 1/2 chance that he will succeed and a 1/3 chance that he will fail. If he fails, there is a 1/4 chance that he will succeed and a 1/2 chance that he will fail. If he wins, he will receive $500,000. If he loses, he will receive nothing. What is the probability of winning if he does not go to war? Express your answer as a decimal.
    To determine the probability of
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. London
    
    B. Paris
    
    C. Amsterdam
    
    D. Rome
    The capital of France is:
    
    B. Paris
    
    Therefore, the correct answer is B. Paris. 
    
    To further elaborate, Paris is the capital city of France and serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as its vibrant neighborhood scenes and rich history. Other options like London or Rome are not capital cities in the same sense. The other options (London and Rome) are not capital cities in
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and AI is likely to have a significant impact on every aspect of our lives. From making our lives easier, to creating new forms of entertainment, to improving our understanding of the world, AI is the future. However, with this potential comes a new set of challenges and ethical considerations that must be taken into account. Here are some of the most pressing issues facing AI in the future:
    The first issue facing AI is privacy. As AI systems become more sophisticated, they are more likely to access personal information. This can lead to a loss of personal privacy and autonomy, which can have serious consequences for individuals and society as a whole.


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] [Ability] who has always been [Positive Trait] in my heart. I'm passionate about [What I Love to Do], and I'm always looking for ways to [What I Want to Improve]. I'm [What I Want to Do Next], and I'm excited to [What I Want to Achieve]. I'm [What I Want to Do Next], and I'm excited to [What I Want to Achieve]. I'm [What I Want to Do Next], and I'm excited to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Academy of Sciences and the French National Museum of Modern Art. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient solutions to complex problems.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see more widespread adoption in healthcare.
    
    3. Increased use of AI in finance: AI
    


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
    Generated text:  [Name], and I am a [Name] who has been trained in [Skill] for [Number] years. I am constantly seeking out new challenges and learning opportunities, and I am excited to meet new people and make a difference in the world. What is your name, and what skill or occupation do you specialize in? I am happy to have the opportunity to learn more about you and get to know you better. [Name] [Skill] [Experience] [Education] [Hobbies] [Achievements] [Skills] [Languages] [Interests] [References] [Education] [Languages] [Interests]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the "City of Light" and "The Eternal City." It is a historic center with many notable landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral. Paris is home to over 10 million people and is known for its rich cultural heritage, fine dining, and lively nightlife. It is a popular tourist destination and a major economic and financial center. It was founded in the 9th century and is the oldest continuously inhabited city in Europe. The city's status as a cultural center, a major economic hub, and its historical significance have made it a popular destination
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving, and there are many possible trends that could shape the field in the coming years. Here are some possible future trends in AI:
    
    1. Autonomous vehicles: As self-driving cars become more common, we may see more autonomous vehicles in the future, using AI algorithms to navigate roads and make decisions about when and where to stop.
    
    2. Chatbots and virtual assistants: AI is already being used in many forms of automated customer service, but we may see even more sophisticated chatbots and virtual assistants that can understand and respond to a wide range of queries and concerns.
    
    3. AI in healthcare: AI is already being used to


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

     character

    's

     name

    ].

     I

    'm

     a

     natural

    -born

     historian

    ,

     with

     a

     passion

     for

     uncover

    ing

     forgotten

     truths

     and

     decipher

    ing

     the

     mysteries

     of

     the

     past

    .

     I

    'm

     a

     skilled

     storyt

    eller

    ,

     with

     a

     knack

     for

     crafting

     compelling

     narratives

     that

     transport

     my

     audience

     to

     different

     er

    as

     and

     cultures

    .

     I

     love

     to

     explore

     new

     places

    ,

     learn

     new

     languages

    ,

     and

     discover

     hidden

     treasures

     in

     the

     most

     unlikely

     places

    .

     I

    'm

     a

     lifelong

     learner

    ,

     always

     eager

     to

     expand

     my

     knowledge

     and

     contribute

     to

     the

     greater

     good

    .

     Whether

     I

    'm

     reading

     a

     book

    ,

     writing

     a

     blog

     post

    ,

     or

     teaching

     history

    ,

     my

     goal

     is

     to

     share

     my

     love

     for

     history

     with

     others

     and

     inspire

     them

     to

     do

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     city

     in

     the

     country

     and

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    .

     It

     is

     also

     one

     of

     the

     world

    ’s

     most

     important

     cultural

    ,

     political

    ,

     and

     economic

     centers

    .

     With

     a

     population

     of

     over

     

    2

    .

     

    5

     million

     people

    ,

     Paris

     is

     a

     thriving

     met

    ropolis

     with

     a

     rich

     history

     and

     diverse

     population

    .

     The

     city

     is

     known

     for

     its

     beautiful

     architecture

    ,

     world

    -class

     museums

    ,

     and

     world

    -ren

    owned

     cuisine

    .

     Paris

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

     every

     year

    .

     Its

     status

     as

     the

     world

    ’s

     most

     important

     city

     is

     well

    -d

    ocumented

     and

     celebrated

    ,

     making

     it

     a

     major

     cultural

     and

     economic

     hub

     in

     the

     world

    .

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     dynamic

    ,

     with

     a

     wide

     range

     of

     possibilities

     and

     potential

     applications

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     More

     powerful

     and

     more

     capable

     machines

    :

     AI

     will

     continue

     to

     get

     faster

     and

     more

     powerful

    ,

     with

     the

     ability

     to

     process

     more

     data

    ,

     learn

     from

     new

     examples

    ,

     and

     understand

     complex

     patterns

     in

     data

    .

     This

     will

     enable

     machines

     to

     solve

     more

     complex

     problems

     and

     perform

     tasks

     that

     are

     currently

     beyond

     their

     current

     capabilities

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     powerful

     and

     capable

    ,

     it

     will

     be

     integrated

     more

     deeply

     into

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

     robotics

    .

     This

     will

     enable

    



```python
llm.shutdown()
```
