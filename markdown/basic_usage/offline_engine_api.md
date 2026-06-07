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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.56it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.72it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.88it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  50%|█████     | 29/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=384 avail_mem=75.75 GB):  50%|█████     | 29/58 [00:00<00:00, 42.73it/s]

    Capturing num tokens (num_tokens=352 avail_mem=75.74 GB):  50%|█████     | 29/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=352 avail_mem=75.74 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=320 avail_mem=75.74 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=288 avail_mem=75.71 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=256 avail_mem=75.03 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=240 avail_mem=75.02 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.54it/s]Capturing num tokens (num_tokens=208 avail_mem=75.02 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.54it/s]Capturing num tokens (num_tokens=192 avail_mem=75.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.54it/s]

    Capturing num tokens (num_tokens=144 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=144 avail_mem=75.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s] Capturing num tokens (num_tokens=80 avail_mem=74.99 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=32 avail_mem=74.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=24 avail_mem=74.98 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=12 avail_mem=74.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s] Capturing num tokens (num_tokens=4 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=4 avail_mem=74.54 GB): 100%|██████████| 58/58 [00:01<00:00, 41.59it/s]


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
    Generated text:  Tuna. I'm a computer game character that originates from the Japan video game industry. I'm an AI based character, and I am designed to bring people joy and enhance their gaming experience.
    
    How did you become the character you are today? As a language model, I was created by Alibaba Cloud to provide users with a wide range of information and answers to their queries. This includes information on various subjects, such as history, culture, science, and technology. I was created to assist and provide useful information to users who may be interested in these topics. I don't have personal experiences or emotions, so I don't have a personal
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a budget for the upcoming fiscal year. The budget should be based on the economic conditions of the country. Given the following table, what is the president's budget for the upcoming fiscal year?
    To determine the president's budget for the upcoming fiscal year, we need to follow these steps:
    
    1. Identify the economic conditions of the country.
    2. Determine the levels of government spending and revenues.
    3. Calculate the total budget based on these conditions.
    
    Since the specific economic conditions and government spending/revenue levels are not provided in the table, I will assume a hypothetical scenario for the purpose of this exercise. Let's assume the following data:
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Brussels C. Lyon D. Berlin
    A. Paris
    答案:
    
    A
    
    在劳动过程中，工人对环境条件及作业场地的观察，这种观察活动属于哪种工作观察？
    A. 安全
    B. 环境
    C. 职业健康安全
    D. 作业
    答案:
    
    B
    
    请选出与英文短语“have fun”最匹配的中文翻译：
    A. 喜欢
    B. 好的
    C. 有
    D. 乐趣
    答案:
    
    D. 乐趣
    
    如何正确地添加软件
    ===============================
    Prompt: The future of AI is
    Generated text:  about human skills, not tech
    
    Technology can help people do things better, but it can't replace the value of human skills and empathy.
    
    AI can make people more productive, but it can't replace the value of human skills and empathy.
    
    It's true that AI is incredibly powerful, and it can make the world a better place. It has the potential to be used to help people make better decisions, as well as to support them in their work.
    
    For instance, by using algorithms to analyze and evaluate data, AI can predict and prevent potential crises or disasters.
    
    However, AI can't replace the value of human skills and empathy. Our


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is also home to many notable French artists and writers, including Pablo Picasso and André Breton. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire and captivate people around the world. The city is also home to many international organizations and institutions, including the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. This includes issues such as bias in AI algorithms, privacy concerns, and the potential
    


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
    Generated text:  [Name] and I am a [occupation] with [number] years of experience. I bring a unique blend of [mention a skill or trait] and a passion for [mention a hobby or interest]. I'm looking for a [specific job title] role and would love to learn more about what kind of tasks and responsibilities would be involved in this position. Please let me know if there are any additional requirements or preferences I should be aware of. (Remember to be as brief as possible and maintain a neutral and professional tone throughout the introduction.) Hello, my name is [Name] and I am a [occupation] with [number
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its stunning architecture, vibrant culture, and rich history. It is home to several world-renowned museums, such as the Louvre, the Musée d'Orsay, and the Centre Pompidou, as well as a bustling market town known as the "Cité des Vins" for its vineyards and wine industry. Paris is a popular tourist destination, with millions of visitors annually making it a major city in Europe. The city's charm and food culture continue to make it a beloved destination for many tourists. Paris is also a popular destination for cultural events, including the famous annual Mardi Gr
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of factors, including advances in technology, changes in society, and shifts in human behavior. Here are some possible future trends in AI:
    
    1. Increased integration with human emotions: With AI becoming more integrated with the human brain, it is likely that we will see a greater understanding of how humans and AI work together. This could lead to more nuanced and empathetic interactions, as well as more effective collaboration and communication.
    
    2. Enhanced natural language processing: As AI becomes more capable of understanding and generating human language, we may see more sophisticated approaches to natural language processing, such as chatbots and virtual assistants that


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

    职业

    ]

     with

     [

    Education

    ]

     in

     [

    Field

     of

     Study

    ].

     I

     am

     passionate

     about

     [

    amb

    ition

    ],

     and

     I

     am

     always

     looking

     for

     [

    h

    obbies

    /

    interest

    s

    ].

     I

     enjoy

     [

    past

    ime

    /s

    port

    ]

     and

     I

     am

     always

     prepared

     for

     [

    challenge

    /

    competition

    ].

     I

     believe

     that

     [

    reason

     for

     success

    ]

     and

     I

     am

     always

     eager

     to

     learn

    .

     My

     work

     ethic

     and

     dedication

     are

     infectious

    ,

     and

     I

     am

     committed

     to

     [

    ultimate

     goal

    /

    amb

    ition

    ].

     Thank

     you

     for

     considering

     me

     as

     a

     potential

     match

    .

     I

     look

     forward

     to

     discussing

     how

     I

     can

     contribute

     to

     your

     team

    .

     [

    Name

    ]

     [

    Experience

    ]

     [

    Education

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     The

     city

     is

     known

     for

     its

     cultural

     significance

    ,

     including

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

     numerous

     museums

     and

     historical

     landmarks

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     nightlife

     and

     popular

     tourist

     destinations

    .

     The

     city

     is

     a

     major

     economic

     and

     political

     center

     in

     Europe

    ,

     hosting

     many

     notable

     events

     and

     festivals

     throughout

     the

     year

    .

     It

     is

     a

     popular

     destination

     for

     tourists

     from

     around

     the

     world

    .

     Paris

     is

     the

     most

     populous

     city

     in

     France

     by

     a

     significant

     margin

    ,

     with

     a

     population

     of

     over

     

    1

    0

     million

     people

    .

     The

     city

     is

     home

     to

     many

     international

     organizations

     and

     institutions

    ,

     such

     as

     the

     European

     Union

     and

     the

     International

     Organization

     for

     Migration

    .

     Paris

     is

     a

     major

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     constantly

     evolving

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

     possible

     trends

     in

     the

     AI

     field

    :
    


    1

    .

     Increased

     automation

     and

     robotic

    ization

    :

     As

     AI

     continues

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     more

     automation

     and

     robotic

    ization

     in

     various

     industries

    .

     This

     could

     include

     tasks

     such

     as

     manufacturing

    ,

     transportation

    ,

     and

     customer

     service

    ,

     where

     AI

    -powered

     systems

     are

     expected

     to

     perform

     tasks

     more

     efficiently

     and

     effectively

     than

     human

     beings

    .
    


    2

    .

     Integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     can

     expect

     to

     see

     more

     AI

    -driven

     innovations

    .

     This

     could

     include

     the

     development

     of

     AI

    -powered

     personal

     assistants

     and

     virtual

     assistants

     that

     can

     assist

     with

    



```python
llm.shutdown()
```
