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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.39it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.39it/s]


    2026-05-20 05:38:01,039 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 05:38:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.70it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.78it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.15it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.13it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.04it/s]Capturing num tokens (num_tokens=320 avail_mem=75.73 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.04it/s]Capturing num tokens (num_tokens=288 avail_mem=75.73 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.04it/s]Capturing num tokens (num_tokens=256 avail_mem=75.73 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.04it/s]Capturing num tokens (num_tokens=240 avail_mem=75.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.04it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.04it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=208 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=192 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.01it/s]

    Capturing num tokens (num_tokens=144 avail_mem=75.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.59it/s] Capturing num tokens (num_tokens=80 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=32 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=24 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.86it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=12 avail_mem=74.97 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=8 avail_mem=74.96 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.29it/s] Capturing num tokens (num_tokens=4 avail_mem=74.96 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=4 avail_mem=74.96 GB): 100%|██████████| 58/58 [00:01<00:00, 41.92it/s]


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
    Generated text:  Paul, I am a single father, and I’m working full-time. I’m in college, I’m just starting to look for a job, and I’m a total loner.
    I’m struggling a bit in some areas but I’m not looking for a full-time job because I am doing this because I want to know more about my situation. I’m sure there’s a great way to find out how to be successful at my job without working full-time. I know that finding a job where I can be successful is something that many people want to know. I have some questions regarding that.
    Is there a better way to find
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what gift to give to his favorite teacher. He has just received a letter from the teacher, written by an anonymous sender. The letter includes a series of problems related to the teacher's field of expertise, with each problem having a unique solution. The president decides to give the student a gift that solves all of the problems.
    
    The president decides to choose one problem from each of the three categories: Arithmetic, Algebra, and Geometry. He starts by solving the Arithmetic problems. Then, he solves the Algebra problems, followed by the Geometry problems. If the president solves exactly one problem from each category, what is the probability that he
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A. Paris B. London C. Moscow D. New York
    Answer:
    
    A
    
    In this question, which of the following statements about the essay is true?
    A. It is a type of short essay
    B. It is an essay of two paragraphs
    C. It can be divided into two parts
    D. The core content is the passage that is to be discussed
    Answer:
    
    A
    
    China's present education system emphasizes the cultivation of all-round development, which is based on the principle of ____. 
    A. The unity of teaching and learning
    B. Teaching according to individual aptitude
    C. Teaching according
    ===============================
    Prompt: The future of AI is
    Generated text:  coming! It is the future of education and the future of the workplace. With these updates coming fast, businesses must adapt to the changing landscape. It is a lot easier to adapt to the new technology with a solid knowledge of the new developments and the strategies to take advantage of them.
    The pace of change is unpredictable, and it is important to stay up to date with the latest trends and developments to stay ahead of the curve. In this article, we will explore the current trends in AI and how businesses can use them to their advantage.
    Artificial Intelligence (AI) is a branch of computer science that focuses on the development of intelligent machines


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] who is always looking for ways to [job title] my skills and knowledge. I'm a [job title] who is always looking for ways to [job title] my skills and knowledge. I'm a [job title] who is always looking for ways to [job title] my skills and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. It is a popular tourist destination, with millions of visitors each year. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a vibrant and dynamic city that continues to be a major center of culture and politics in France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more
    


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
    Generated text:  [Name], and I'm a writer and graphic designer based in [Location]. I'm passionate about [past experience] and have developed a unique approach to [what you do here]. I enjoy creating a cohesive and visually stunning story or illustration that captures the essence of [name] through the use of [specific tools or techniques]. My ultimate goal is to [what you hope to achieve here], and I'm excited about the opportunity to share my talents with you all. I'm available to discuss my work and how I can contribute to your projects. [Contact information]. Welcome to the world of my writing and design, where I bring my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the fifth-largest city in the world. It is the 13th most populous city in the world, with an estimated population of approximately 1, 371, 000. It is an important cultural and economic center, known for its history, architecture, cuisine, and fashion. Paris is also known for its romantic, artistic, and romantic atmosphere, as well as its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a vibrant and diverse city with a rich tapestry of art, literature,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities, but we also face some significant challenges that need to be addressed. Here are some potential future trends in AI:
    
    1. Artificial General Intelligence (AGI): This would be a hypothetical level of AI that could solve complex, multi-faceted problems on its own, without the need for human input. This could lead to a new era of automation and efficiency in many industries.
    
    2. Superintelligence: The idea of AI superintelligence could emerge in the future, capable of solving problems and achieving complex goals that even the most advanced AI currently can't. This could have significant implications for society and technology as a whole


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ].

     I

     was

     born

     in

     [

    Birth

    place

    ],

     a

     [

    City

    /T

    own

    ]

     with

     [

    Name

    ],

     a

     [

    City

    /T

    own

    ]

     with

     [

    Job

     or

     Profession

    ].

     I

     have

     [

    Number

    ]

     close

     friends

    ,

     and

     I

    'm

     [

    Love

    /T

    alk

     about

     it

    ],

     trying

     to

     balance

     my

     career

     and

     personal

     life

    .

     I

    'm

     happy

     to

     chat

     about

     my

     day

     or

     week

    ,

     and

     I

    'm

     always

     up

     for

     a

     good

     laugh

    .

     How

     can

     I

     be

     a

     part

     of

     your

     conversation

    ?

     Well

    ,

     that

    's

     all

     I

     have

     for

     now

    .

     Please

     feel

     free

     to

     ask

     me

     anything

     you

    'd

     like

     to

     know

    ,

     and

     I

    'll

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     renowned

     for

     its

     beautiful

     architecture

    ,

     iconic

     landmarks

    ,

     and

     vibrant

     culture

    .

     It

     is

     also

     known

     for

     hosting

     major

     events

     such

     as

     the

     E

    iff

    el

     Tower

    's

     annual

     July

     

    1

    4

    th

     celebrations

     and

     the

     annual

     M

    ardi

     Gr

    as

     festival

    .

     Additionally

    ,

     Paris

     is

     a

     hub

     for

     international

     business

     and

     finance

    ,

     and

     is

     home

     to

     many

     of

     the

     world

    's

     most

     renowned

     museums

    ,

     theaters

    ,

     and

     galleries

    .

     Paris

     is

     also

     known

     for

     its

     French

     cuisine

     and

     fashion

     industry

    ,

     which

     has

     made

     it

     a

     global

     leader

     in

     these

     areas

    .

     The

     city

     is

     known

     for

     its

     art

    ,

     music

    ,

     and

     literature

    ,

     and

     has

     been

     a

     center

     of

     learning

     and

     culture

     for

     centuries

    .

     As

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     exciting

     developments

     that

     are

     sure

     to

     change

     our

     lives

     in

     profound

     ways

    .

     Here

     are

     some

     of

     the

     potential

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Aug

    mented

     reality

    :

     AR

     will

     become

     more

     widespread

    ,

     allowing

     people

     to

     interact

     with

     the

     real

     world

     through

     digital

     screens

     and

     displays

    .

     This

     could

     revolution

    ize

     education

    ,

     healthcare

    ,

     and

     more

    .
    


    2

    .

     Brain

    -com

    puter

     interfaces

    :

     The

     integration

     of

     AI

     with

     the

     brain

     could

     lead

     to

     more

     advanced

     technology

     like

     AI

    -powered

     brain

    -com

    puter

     interfaces

    ,

     which

     could

     enable

     people

     to

     control

     devices

     and

     machines

     through

     their

     thoughts

    .
    


    3

    .

     AI

     ethics

     and

     safety

    :

     AI

     is

     increasingly

     being

     used

     in

     fields

     like

     healthcare

     and

     finance

    ,

     but

     it

    



```python
llm.shutdown()
```
