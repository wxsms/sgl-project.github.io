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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.79it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.90it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.90it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.90it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.90it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.90it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.90it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.90it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 22.90it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 22.90it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 22.90it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 22.90it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.50 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.49 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.48 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.45 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.42 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.29it/s]Capturing num tokens (num_tokens=960 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.29it/s] Capturing num tokens (num_tokens=896 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.29it/s]

    Capturing num tokens (num_tokens=832 avail_mem=73.43 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.29it/s]Capturing num tokens (num_tokens=832 avail_mem=73.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.72it/s]Capturing num tokens (num_tokens=768 avail_mem=73.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.72it/s]Capturing num tokens (num_tokens=704 avail_mem=73.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.72it/s]Capturing num tokens (num_tokens=640 avail_mem=73.42 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.72it/s]Capturing num tokens (num_tokens=576 avail_mem=73.42 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.72it/s]Capturing num tokens (num_tokens=512 avail_mem=73.41 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.72it/s]Capturing num tokens (num_tokens=512 avail_mem=73.41 GB):  50%|█████     | 29/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=480 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=448 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=416 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=384 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 42.98it/s]

    Capturing num tokens (num_tokens=352 avail_mem=73.41 GB):  50%|█████     | 29/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=352 avail_mem=73.41 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.53it/s]Capturing num tokens (num_tokens=320 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.53it/s]Capturing num tokens (num_tokens=288 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.53it/s]Capturing num tokens (num_tokens=256 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.53it/s]Capturing num tokens (num_tokens=240 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.53it/s]Capturing num tokens (num_tokens=224 avail_mem=73.39 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.53it/s]Capturing num tokens (num_tokens=224 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=208 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.61it/s]Capturing num tokens (num_tokens=192 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=176 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=160 avail_mem=73.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.61it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=144 avail_mem=73.38 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=128 avail_mem=73.38 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=112 avail_mem=73.38 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=96 avail_mem=73.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.27it/s] Capturing num tokens (num_tokens=80 avail_mem=73.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=64 avail_mem=73.36 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=64 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=48 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.49it/s]

    Capturing num tokens (num_tokens=32 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=28 avail_mem=73.35 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=24 avail_mem=73.35 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=20 avail_mem=73.35 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=20 avail_mem=73.35 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.70it/s]

    Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.70it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 30.81it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 35.98it/s]


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
    Generated text:  Justin. I have a very odd, wondrous, and somewhat eccentric experience where I have completed four hours of sleep, had one meal, and had a short break during the day. I’m not sure what it is about my condition that makes me feel very refreshed and alive despite my lack of sleep, and I’m wondering what it is that causes this phenomenon. What is it?
    
    Your experiences of sleeping for four hours, having one meal, and a short break during the day can indeed be quite unique and refreshing. There are several factors that might contribute to your experience, but it's important to note that these experiences are not uncommon and
    ===============================
    Prompt: The president of the United States is
    Generated text:  now considering a policy to introduce a new tax on imported goods. The tax will be applied to a specific percentage of each imported item, and the tax rate is proportional to the weight of the item. The president wants to ensure that the tax is effective in reducing the price of imported goods without significantly impacting the purchasing power of the American people. To do this, the president has decided to introduce a tax on imported goods that is proportional to the weight of the item, and the tax rate is 1% per unit weight. However, the president also wants to ensure that the tax rate is affordable to the average American taxpayer, so he decides
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it is located in which of the following regions?
    A. Central Europe
    B. Western Europe
    C. Southern Europe
    D. Northern Europe
    Answer:
    B
    
    Which of the following statements about the historical development of ancient China and the cultural achievements in its achievements is incorrect?
    A. The formation of the Chinese nation is marked by the prosperity of the Yellow Emperor and the prosperous Tang Dynasty.
    B. The Cultural Revolution was a form of cultural revolution initiated by Mao Zedong.
    C. The Qin Dynasty was the first to truly establish a unified multi-ethnic state.
    D. The founding of New China represents a new
    ===============================
    Prompt: The future of AI is
    Generated text:  leading the next wave of change in how people live and work. It’s currently in the early stages of growth, but the potential is immense.
    In this article, we’ll explore the impact of AI on all aspects of our lives, from jobs to business operations to personal relationships.
    But before we get into that, let’s briefly discuss the current state of AI. We’ll start with some background information on machine learning, artificial intelligence and how these technologies are transforming our world.
    So, let’s dive in!
    What is AI?
    Artificial intelligence (AI) is the simulation of human intelligence processes by computer systems. These systems can learn from


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new experiences and learning new things. What are some of your favorite things to do? I love [insert a short description of your favorite activities or hobbies]. I'm always looking for new adventures and trying new things. What's your favorite book or movie? I love [insert a short description of your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in the country and the seat of the French government and the country's cultural, political, and economic center. Paris is known for its rich history, beautiful architecture, and vibrant culture, and is a popular tourist destination. The city is also home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a major hub for international trade and diplomacy, and is a cultural and educational center for France. It is also a major financial center, with many financial institutions and companies headquartered there. Paris is a city that is constantly evolving and changing
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced machine learning capabilities: AI is likely to become more capable of learning from large datasets and making more accurate predictions and decisions.
    
    3. Increased use of AI in healthcare: AI is likely to play a more significant role in healthcare, with machines being used to diagnose and treat diseases, as well as to help doctors make more informed decisions.
    
    4. Increased use of AI in transportation: AI is
    


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
    Generated text:  __________. I'm a/an ________ and I'm from the city of __________. I've been working in the __________ department at __________ for over __________ years. I enjoy doing __________ and __________. As a/an __________, I always want to learn new things, especially __________. I'm always up for challenges, and I thrive on trying new things. As a/an __________, I'm always looking for ways to make my work easier and more efficient. I'm a/an __________, and I'm always looking for ways to improve my skills and knowledge. As a/an __________, I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city with the most iconic landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. 
    
    Let me know if you need any other information. Paris is known for its rich history and beautiful architecture. It's a city with a diverse population, from the trendy neighborhoods like Montmartre and Le Marais to the more traditional areas of Paris. 
    
    Paris is known for its vibrant culture and festival season, with the Christmas market on January 7th and the anniversary of the 1944 defeat of Nazi Germany on July 20th. It's also home to the World Trade
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of factors, including advances in computing power and the availability of data, as well as changing societal and cultural attitudes towards AI. Here are some possible future trends in artificial intelligence:
    
    1. Increased integration with other technologies: AI is becoming increasingly integrated with other technologies, such as big data, sensors, and Internet of Things (IoT) devices. This integration will likely lead to more efficient and effective use of AI, as well as the development of new applications and industries that are possible only with the help of AI.
    
    2. Higher levels of automation: As AI becomes more integrated with other technologies, it will


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

    H

    obby

    /

    Interest

    ]

     enthusiast

    .

     I

     have

     been

     studying

     this

     hobby

     for

     [

    Number

    ]

     years

    ,

     and

     it

     has

     had

     a

     profound

     impact

     on

     my

     life

    .

     I

     am

     [

    Number

    ]

     years

     old

    ,

     and

     I

     love

     to

     [

    describe

     your

     hobby

     and

     how

     it

     has

     made

     you

     feel

    ]

     every

     day

    .

     You

     can

     reach

     me

     at

     [

    Your

     Phone

     Number

    ]

     or

     [

    Your

     Email

     Address

    ].

     What

     better

     way

     to

     introduce

     yourself

     to

     someone

     who

     might

     not

     know

     me

     already

    ?

     You

     can

     imagine

     how

     exciting

     it

     would

     be

     to

     meet

     someone

     who

     has

     something

     in

     common

     with

     you

     and

     can

     learn

     so

     much

     about

     you

    .

     I

     am

     excited

     to

     share

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     political

    ,

     cultural

    ,

     and

     economic

     center

     of

     France

    .

     It

     is

     also

     known

     as

     the

     City

     of

     Light

     and

     the

     “

    City

     of

     a

     Hundred

     Flowers

    .”

     Paris

     has

     a

     long

     and

     rich

     history

     dating

     back

     to

     the

     Roman

     era

    ,

     and

     it

     has

     been

     the

     capital

     of

     France

     since

     

    1

    8

    7

    1

    .

     The

     city

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

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

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     its

     art

     scene

    ,

     and

     its

     celebration

     of

     the

     arts

    .

     The

     city

     is

     also

     home

     to

     many

     international

     institutions

     and

     organizations

    ,

     including

     the

     European

     Parliament

     and

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     currently

     focused

     on

     exploring

     and

     developing

     new

     technologies

     to

     enhance

     and

     expand

     the

     capabilities

     of

     AI

     systems

    .

     Here

     are

     some

     potential

     trends

     that

     may

     shape

     the

     development

     of

     AI

     in

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     focus

     on

     privacy

     and

     security

    :

     As

     more

     data

     and

     personal

     information

     is

     becoming

     accessible

     through

     AI

    ,

     there

     is

     a

     growing

     need

     for

     measures

     to

     protect

     this

     data

     from

     being

     mis

    used

     or

     compromised

    .

     This

     could

     mean

     increased

     regulations

     on

     data

     handling

     practices

    ,

     encryption

    ,

     and

     anonym

    ization

    .
    


    2

    .

     Adv

    ancements

     in

     natural

     language

     processing

    :

     Natural

     language

     processing

     is

     a

     key

     area

     of

     AI

     research

     that

     aims

     to

     enable

     machines

     to

     understand

     and

     respond

     to

     human

     speech

    .

     This

     could

     lead

     to

     more

    



```python
llm.shutdown()
```
