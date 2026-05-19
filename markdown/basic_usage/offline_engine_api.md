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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.77it/s]


    2026-05-19 21:08:07,877 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-19 21:08:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:46,  3.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:46,  3.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:46,  3.98s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:46,  3.98s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:46,  3.98s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.83it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 34.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.16it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.16it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.16it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.16it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.65it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.65it/s]Capturing num tokens (num_tokens=640 avail_mem=75.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.65it/s]Capturing num tokens (num_tokens=576 avail_mem=74.92 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.65it/s]Capturing num tokens (num_tokens=512 avail_mem=74.91 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.65it/s]Capturing num tokens (num_tokens=512 avail_mem=74.91 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=480 avail_mem=74.92 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=448 avail_mem=74.92 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=416 avail_mem=74.92 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.92 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=352 avail_mem=74.91 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=352 avail_mem=74.91 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=320 avail_mem=74.91 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=288 avail_mem=74.91 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=256 avail_mem=74.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=240 avail_mem=74.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=224 avail_mem=74.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=224 avail_mem=74.90 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=208 avail_mem=74.89 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=192 avail_mem=74.89 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=176 avail_mem=74.89 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.00it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.89 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=144 avail_mem=74.88 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=144 avail_mem=74.88 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=128 avail_mem=74.88 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=112 avail_mem=74.88 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=96 avail_mem=74.87 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.81it/s] Capturing num tokens (num_tokens=80 avail_mem=74.87 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=64 avail_mem=74.87 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=64 avail_mem=74.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=48 avail_mem=74.86 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=32 avail_mem=74.86 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=28 avail_mem=74.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.89it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=20 avail_mem=74.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=20 avail_mem=74.85 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=16 avail_mem=74.85 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=12 avail_mem=74.84 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=8 avail_mem=74.84 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s] Capturing num tokens (num_tokens=4 avail_mem=74.84 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=4 avail_mem=74.84 GB): 100%|██████████| 58/58 [00:01<00:00, 40.93it/s]


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
    Generated text:  Geri. I’m a 26-year-old female. I’m 6’2” tall and weigh 175 pounds. I have curly brown hair and blue eyes. I’m passionate about cooking and fitness, and I love to exercise on the elliptical. I live in Long Beach and enjoy eating there for lunch. I love to travel and eat out to new places. My favorite books are The Lord of the Rings and The Hobbit. I have a 4-year-old daughter. I have a friend, Rebecca, who has a 3-year-old daughter. We have been married for 13 years,
    ===============================
    Prompt: The president of the United States is
    Generated text:  now trying to decide who should be the next Chief Justice of the United States. There are four candidates. Each candidate has a different amount of years of experience: the first has 10 years of experience, the second has 15 years of experience, the third has 20 years of experience, and the fourth has 25 years of experience. The president decides to select the candidate with the most experience to become the Chief Justice. 
    
    If the president selects the candidate with the most experience, what is the total number of years of experience the president has selected for the Chief Justice? To determine the total number of years of
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. New York
    D. Tokyo
    A. Paris
    
    Paris is the capital of France and is the largest city in the country. While New York and Tokyo are also major cities in the United States, Paris is the largest city in France. London is the capital of the United Kingdom, and Tokyo is the capital of Japan. The other options listed are smaller cities or countries, so they are not considered capitals of countries.
    ===============================
    Prompt: The future of AI is
    Generated text:  looking much better now that it has been making significant strides in areas like image recognition, speech recognition, and even facial recognition. However, you cannot ignore the growing concern of AI’s impact on privacy and security, and the ethical considerations surrounding the use of artificial intelligence in society. With the rapid advancement of technology, AI is taking a bigger role in the lives of people all over the world. In this article, we will take a look at how AI is impacting society, and the ethical considerations surrounding the use of AI.
    AI is having a significant impact on society, and it is transforming the way we live and work. From the way we


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm always eager to learn and grow, and I'm always looking for new opportunities to contribute to the company. What's your name? What's your job title? What's your company name? What's your favorite hobby? What's your favorite book? What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many of the world's most famous museums and attractions. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. Its status as the world's most populous city is also reflected in its population of over 20 million people. The city is also home to many international organizations and institutions, including the European Union and the United Nations. Paris is a city of contrasts, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is a growing push for ethical AI. This includes developing AI that is transparent, accountable, and respects human values and rights.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. As these technologies continue to evolve, it is likely that AI will be integrated with other technologies to create even more advanced and integrated
    


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
    Generated text:  [Name] and I'm an [Age] year old girl who has been exploring the world on my own for the past [Number] years. I am a [Occupation] who love to be [Why] because [What you like to do] and I love [Anything you find meaningful] in life. I have [Number] friends who share my interests and I am always looking for new experiences to try. I enjoy [What you enjoy doing that you don't get from a job or a hobby]. My personal goal is to [What you want to achieve or achieve in the next 5 years]. I believe [What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Parthenon" due to its classical architecture and grandiose beauty. It is the largest city in France and the fifth-largest city in the world by population. Paris is home to many world-renowned landmarks, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also known for its unique culture, cuisine, and fashion, and is a popular tourist destination for many visitors. Paris has a rich history, dating back to the Roman Empire and the French Revolution, and continues to be a vibrant and dynamic city today. Its strategic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several trends, including:
    
      1. Increasing reliance on AI for routine tasks such as customer service, automation of repetitive processes, and routine healthcare tasks.
      2. AI will continue to become more sophisticated and autonomous, with the ability to learn and adapt to new situations.
      3. AI will continue to be used in a wider range of applications beyond just the work-place, including in fields such as finance, transportation, and transportation, where it is expected to continue to play an increasingly important role.
      4. AI will continue to be used for both positive and negative applications, with the potential to revolution


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

    First

     Name

    ]

     and

     I

    'm

     a

     [

    Title

    ]

     at

     [

    Company

     Name

    ].

     I

    'm

     passionate

     about

     [

    what

     you

     do

    ],

     and

     I

     have

     a

     strong

     desire

     to

     learn

     and

     grow

     in

     my

     career

    .

     I

     love

     being

     creative

    ,

     and

     I

     thrive

     on

     taking

     on

     new

     challenges

     and

     pursuing

     my

     passion

     for

     [

    what

     you

     do

    ].

     If

     you

     need

     any

     help

     or

     support

    ,

     please

     don

    't

     hesitate

     to

     reach

     out

     to

     me

    .

     Thank

     you

     for

     considering

     me

     for

     your

     interest

    .

     Can

     you

     elaborate

     on

     your

     specific

     interests

     and

     passions

     related

     to

     your

     career

     at

     [

    Company

     Name

    ]?

     Certainly

    !

     As

     a

     seasoned

     [

    Title

    ]

     at

     [

    Company

     Name

    ],

     I

    'm

     dedicated

     to

     achieving

     exceptional

     results

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     historical

     landmarks

     and

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     unpredictable

    ,

     with

     no

     clear

     path

     that

     one

     can

     follow

     with

     certainty

    .

     However

    ,

     here

     are

     some

     possible

     future

     trends

     in

     AI

     that

     could

     impact

     our

     lives

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     will

     become

     more

     common

     in

     the

     future

    ,

     with

     cars

     equipped

     with

     advanced

     AI

     that

     can

     drive

     safely

     and

     efficiently

     on

     the

     roads

    .

     This

     will

     revolution

    ize

     transportation

    ,

     making

     it

     faster

    ,

     more

     fuel

    -efficient

    ,

     and

     safer

    .
    


    2

    .

     Smart

     homes

    :

     AI

     will

     be

     integrated

     into

     homes

    ,

     enabling

     automation

     in

     daily

     life

    .

     Smart

     homes

     will

     have

     home

     automation

    ,

     energy

     management

    ,

     and

     security

     features

     that

     will

     improve

     convenience

    ,

     reduce

     waste

    ,

     and

     enhance

     the

     quality

     of

     life

    .
    


    3

    .

    



```python
llm.shutdown()
```
