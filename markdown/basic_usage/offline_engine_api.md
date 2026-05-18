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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.29it/s]


    2026-05-18 11:30:28,913 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 11:30:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]

    Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]

    Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:05<00:01, 19.56it/s]Compiling num tokens (num_tokens=176):  57%|█████▋    | 33/58 [00:05<00:01, 19.56it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 27.91it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 36.38it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 36.38it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 36.38it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 36.38it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 36.38it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 36.38it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 36.38it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 36.38it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.65 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.63 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.63 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.62 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.62 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.61 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=69.61 GB):  21%|██        | 12/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.61 GB):  21%|██        | 12/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.61 GB):  21%|██        | 12/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.60 GB):  21%|██        | 12/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.60 GB):  21%|██        | 12/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.60 GB):  21%|██        | 12/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s]Capturing num tokens (num_tokens=960 avail_mem=69.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.01it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=69.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=896 avail_mem=69.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=832 avail_mem=69.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=768 avail_mem=69.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=704 avail_mem=69.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=640 avail_mem=69.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=640 avail_mem=69.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.28it/s]Capturing num tokens (num_tokens=576 avail_mem=69.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.28it/s]Capturing num tokens (num_tokens=512 avail_mem=69.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.28it/s]Capturing num tokens (num_tokens=480 avail_mem=69.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.28it/s]Capturing num tokens (num_tokens=448 avail_mem=69.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.28it/s]Capturing num tokens (num_tokens=416 avail_mem=69.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.28it/s]

    Capturing num tokens (num_tokens=416 avail_mem=69.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.05it/s]Capturing num tokens (num_tokens=384 avail_mem=69.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.05it/s]Capturing num tokens (num_tokens=352 avail_mem=69.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.05it/s]Capturing num tokens (num_tokens=320 avail_mem=69.55 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.05it/s]Capturing num tokens (num_tokens=288 avail_mem=69.55 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.05it/s]Capturing num tokens (num_tokens=256 avail_mem=69.55 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.05it/s]Capturing num tokens (num_tokens=256 avail_mem=69.55 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=240 avail_mem=69.54 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=224 avail_mem=69.54 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=208 avail_mem=69.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=192 avail_mem=69.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=176 avail_mem=69.53 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.26it/s]

    Capturing num tokens (num_tokens=176 avail_mem=69.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=160 avail_mem=69.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=144 avail_mem=69.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=128 avail_mem=69.52 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=112 avail_mem=69.52 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=96 avail_mem=69.52 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s] Capturing num tokens (num_tokens=96 avail_mem=69.52 GB):  81%|████████  | 47/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=80 avail_mem=69.51 GB):  81%|████████  | 47/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=64 avail_mem=69.51 GB):  81%|████████  | 47/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=48 avail_mem=69.51 GB):  81%|████████  | 47/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=32 avail_mem=69.50 GB):  81%|████████  | 47/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=28 avail_mem=69.50 GB):  81%|████████  | 47/58 [00:01<00:00, 46.21it/s]

    Capturing num tokens (num_tokens=28 avail_mem=69.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=24 avail_mem=69.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=20 avail_mem=69.49 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=16 avail_mem=69.49 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=12 avail_mem=69.49 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=8 avail_mem=69.48 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.37it/s] Capturing num tokens (num_tokens=8 avail_mem=69.48 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=4 avail_mem=69.48 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=4 avail_mem=69.48 GB): 100%|██████████| 58/58 [00:01<00:00, 39.56it/s]


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
    Generated text:  Milan.
    It's very nice meeting you.
    I am going to try to learn more about the role of compassion in promoting positive mental health.
    I have always been a person who values the importance of kindness, empathy, and understanding. I have always been mindful of how my actions can affect others, and I am constantly striving to do my best to make a positive impact on the world.
    So far, I have learned that compassion is essential in helping people cope with stress and anxiety. It is the ability to connect with others and understand their perspectives, which can help reduce feelings of isolation and help others feel less alone in their struggles.
    I have
    ===============================
    Prompt: The president of the United States is
    Generated text:  supposed to have a doctorate in philosophy. The president of the United States has a family of 5 children, with the eldest daughter being 3 years younger than the youngest child, and the second eldest daughter being 4 years older than the second youngest child. If the average age of the children is 15 years, what is the age of the youngest child?
    
    Let's denote the age of the youngest child as \( x \). Since the eldest daughter is 3 years younger than the youngest child, her age is \( x - 3 \). The second eldest daughter is 4 years older than the second youngest child,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the country’s largest city, and it is the only city in the world that has served as both capital and center of government for more than 500 years.
    The French government has always been committed to creating a clean and green environment for everyone, and the city of Paris is well known for its contribution to this. The city has been built over 1000 years, and the 17th century city has become one of the most unique and well-preserved cities in the world.
    The capital of France, Paris, is a truly remarkable city that has endured over 1000 years of history
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it’s not without challenges. Experts predict that with continued advancements in AI, there will be a significant impact on our daily lives. However, there are also some potential risks and challenges that need to be addressed. In this post, we will explore the latest developments in AI and the potential risks and challenges that may arise.
    
    One of the main challenges with AI is its impact on jobs. As AI systems become more advanced, it may be challenging for people to find new jobs in the workforce. This could lead to unemployment and a sense of uncertainty for many people. To address this challenge, experts recommend that workers should learn new skills


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I do for you today? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career. What can I do for you today? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career. What can I do for you today? [Name] is a [job title] at [company name], and I'm excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city) due to its floating population. It is the largest city in Europe by area and population, and is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many other iconic landmarks. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is also home to the French Parliament, the French Academy of Sciences, and the French Parliament building. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. The city is also known for its fashion industry, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to improve, we can expect to see even more widespread adoption of AI in healthcare.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud detection, and investment decision-making. As AI technology continues to improve, we can expect to see even more widespread adoption of AI in
    


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
    Generated text:  [insert fictional character's name], and I'm a [insert fictional character's age and gender]. I'm a [insert fictional character's occupation or profession], and I've always been interested in [insert fictional character's hobbies or interests]. I'm [insert fictional character's personality traits or background]. I enjoy [insert fictional character's hobbies or interests]. I'm always [insert fictional character's personality traits or background]. I'm [insert fictional character's personality traits or background]. I'm [insert fictional character's personality traits or background].
    [Insert fictional character's name]. I am [insert fictional character's name]. My name is [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    The French capital city, Paris, is renowned for its vibrant cultural scene, romantic ambiance, and historic architecture. The city's annual Eiffel Tower Tour is a popular tourist attraction, showcasing the city's iconic skyline and iconic landmark. Paris is also home to many museums, art galleries, and theaters, making it a favorite among visitors. With its rich history and diverse cultural offerings, Paris continues to be a major hub of international business and tourism. Paris is also known for its delicious food scene, with its famous French cuisine and delicious French wines. The city is also known for its fashion industry, with many designers and boutiques
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, and there are several trends that are likely to continue or evolve in the coming years. Some of the possible future trends in AI include:
    
    1. Increased Integration: As AI technology continues to advance, it is likely that there will be more integration between AI and other technologies such as the Internet of Things (IoT), blockchain, and the cloud. This integration will enable AI to perform tasks that are currently done manually or require specialized skills, and it will also enable the development of new AI-powered solutions.
    
    2. Enhanced Data Handling: With the increasing volume and complexity of data, it is likely that AI will need to be able


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

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

     am

     currently

     living

     in

     [

    City

    ,

     Country

    ].

     My

     [

    Job

     Title

    ]

     has

     been

     with

     me

     for

     [

    Number

     of

     Years

    ].

     I

     am

     passionate

     about

     [

    reason

     for

     passion

    ].

     I

     enjoy

     [

    a

     hobby

    ,

     interest

    ,

     or

     activity

     that

     interests

     me

    ].

     I

     am

     excited

     to

     meet

     you

     here

    .
    


    I

     hope

     you

     enjoy

     our

     conversation

     and

     I

     look

     forward

     to

     discussing

     my

     experience

     and

     the

     opportunities

     that

     may

     be

     available

     to

     me

     in

     the

     future

    .

     How

     can

     I

     assist

     you

     today

    ?

     
    


    P

    .S

    .

     I

    'm

     always

     open

     to

     learning

     new

     things

     and

     engaging

     in

     new

     activities

    .

     Thanks

     for

     taking

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

     and

     is

     a

     historical

    ,

     cultural

     and

     architectural

     city

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

     It

     is

     also

     famous

     for

     its

     vibrant

     cultural

     scene

    ,

     with

     a

     diverse

     range

     of

     music

    ,

     dance

    ,

     literature

    ,

     and

     cuisine

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     a

     mix

     of

     modern

     and

     traditional

     architecture

    ,

     and

     is

     a

     popular

     tourist

     destination

     for

     its

     beautiful

     natural

     surroundings

     and

     historic

     sites

    .

     It

     is

     a

     cultural

     and

     intellectual

     center

     of

     Europe

     and

     plays

     a

     central

     role

     in

     France

    's

     political

     and

     economic

     life

    .

     The

     city

     is

     also

     home

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     improvements

     in

     data

     analysis

     and

     machine

     learning

     algorithms

    ,

     and

     increased

     focus

     on

     ethical

     considerations

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

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     technology

     becomes

     more

     prevalent

     in

     our

     daily

     lives

    ,

     there

     is

     an

     increasing

     need

     for

     ethical

     considerations

    .

     This

     includes

     questions

     about

     privacy

    ,

     bias

    ,

     and

     fairness

    .

     As

     a

     result

    ,

     there

     is

     likely

     to

     be

     increased

     investment

     in

     AI

     ethics

     research

     and

     development

    .
    


    2

    .

     Improved

     algorithms

    :

     With

     the

     help

     of

     machine

     learning

     algorithms

    ,

     AI

     systems

     are

     becoming

     more

     capable

     of

     learning

     and

     improving

     over

     time

    .

     This

     means

     that

     AI

    



```python
llm.shutdown()
```
