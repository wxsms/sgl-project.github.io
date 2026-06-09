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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.36it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.74it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.83it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.83it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.83it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.83it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.83it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.83it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.83it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.84it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.84it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.84it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.84it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.84it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.84it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.84it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.71it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.71it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.71it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.71it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.70it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.61it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.31it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.96it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.40it/s]


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
    Generated text:  Nicole. I love to travel and exploring new places. I also love to cook. I'm a third generation American chef who grew up in a house where the family moved between America, Europe, and China. It was the first time I really discovered the cuisine of New York City. So, I came from New York City to help a friend make a restaurant called "Pistola." My friend's passion for Thai food is very strong and I wanted to be a part of that. So I decided to work in the kitchen and to try to teach the people what they can do in the kitchen. I love to cook not just as
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to issue a new mandate. The new mandate is to give a certain percentage of the income from federal income tax to the wealthiest 0.01% of the population. If the wealthiest 0.01% of the population has $50,000,000 in income, how much money should the president allocate to the wealthiest 0.01% of the population? To determine how much money the president should allocate to the wealthiest 0.01% of the population, we need to calculate 0.01% of $50,000,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the capital of Germany is Berlin. 
    
    If you travel to Paris, you can only get the best views of the city and everything within it, while you can only travel to Berlin, you can get access to the lesser known spots.
    
    Traveling to Paris and Berlin are two different experiences, but both offer a unique and unforgettable experience.
    
    The Parisian views of the city, the bustling streets, the historical landmarks, the fashionable cafes, and the delicious food. On the other hand, the lesser known spots of Berlin, like the trendy bars, the underground cafes, the vibrant markets, the charming historic districts, the cozy underground trains
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    In the past few years, there has been a significant shift in the development of artificial intelligence (AI). In particular, the field of machine learning has made remarkable progress, enabling machines to learn from data and develop algorithms to perform tasks. In addition, the development of deep learning, a subset of machine learning, has led to major breakthroughs in tasks such as image and speech recognition, natural language processing, and autonomous driving. This has led to the creation of intelligent systems that can perform a wide range of complex tasks that were previously impossible or very difficult to achieve. This has had a significant impact on various industries, including healthcare,


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Number] years of experience in the field of [Field of Work]. I'm a [Number] year old, [Number] years of experience in the field of [Field of Work]. I'm a [Number] year old, [Number] years of experience in the field of [Field of Work]. I'm a [Number] year old, [Number] years of experience in the field of [Field of Work]. I'm a [Number] year old, [Number] years of experience in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its delicious cuisine, including French cuisine, and its fashion industry. It is a popular tourist destination and a major economic center in Europe. The city is home to many cultural institutions and events throughout the year, including the annual Eiffel Tower Festival and the annual Parisian Film Festival. Paris is a city of contrasts, with its modern architecture and historical
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient care, from personalized treatment plans to predictive analytics for disease diagnosis. As AI technology continues to improve, we can expect to see even more sophisticated and personalized healthcare solutions.
    
    2. Enhanced automation of routine tasks: AI is already being used to automate routine tasks in industries such as manufacturing and transportation, freeing up human workers to focus on more complex tasks. As AI technology continues to improve, we can expect to see even more automation in other industries.
    
    3. Increased use of AI in education:
    


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
    Generated text:  [Name], and I'm an experienced software developer with a proven track record in creating innovative and functional software solutions. I enjoy working with teams to design and develop custom solutions, taking ownership of the end result, and continuously seeking ways to improve and innovate. I'm a hard worker, detail-oriented, and enjoy learning new technologies and methodologies. I'm excited to meet you and let's build something amazing together! Have a great day! Let's get started. [Your Name] - [Your Position] - [Your Experience] - [Your Skills] - [Your Interests and Goals]
    [Your Name]: Short and friendly introduction for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, often referred to as the "City of Love" or "City of Light, " is the largest city in France and its capital. It is located on the Île de France, an islet in the Lower Seine river, and is the oldest continuously inhabited city in Europe. The city is famous for its art, culture, cuisine, and fashion, and is a cultural, political, and economic center in continental Europe. Paris has a rich history dating back to the Roman Empire and is home to many UNESCO World Heritage sites. Today, it continues to be a major center for business, tourism, and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a significant shift towards more advanced and integrated AI systems that can handle more complex tasks and interactions. Some potential trends include:
    
    1. Increased integration of AI with other technologies: AI will continue to be integrated into a wide range of technologies, including healthcare, transportation, security, and smart homes. AI systems will become more integrated with other technologies, such as blockchain, to create new possibilities for innovative applications.
    
    2. Emergence of new AI forms: AI forms will continue to evolve, with new forms emerging as more complex and sophisticated. This could include new AI-based systems for customer service, natural language processing, and autonomous vehicles.
    
    


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

     name

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

    ].

     I

    'm

     a

     [

    insert

     age

    ,

     gender

    ,

     race

    ,

     etc

    .]

     who

     has

     always

     been

     fascinated

     by

     [

    insert

     something

     related

     to

     your

     profession

     or

     interests

    ].

     I

     recently

     discovered

     [

    insert

     a

     new

     hobby

     or

     skill

    ],

     and

     I

    'm

     excited

     to

     share

     what

     I

    've

     learned

     with

     you

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?


    I

    'm

     always

     looking

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     happy

     to

     share

     my

     experiences

    .

     So

     if

     you

     have

     any

     questions

    ,

     concerns

    ,

     or

     topics

     you

    'd

     like

     to

     discuss

    ,

     please

     feel

     free

     to

     ask

     and

     I

    'll

     do

     my

     best

     to

     answer

    .

     You

     can

     rest

     assured

     that

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which is

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     a

     major

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     the

     country

    .


    You

     are

     to

     answer

     this

     question

    :

     What

     is

     the

     capital

     of

     France

    ?

     Paris

    


    You

     will

     be

     given

     the

     task

     to

     provide

     a

     concise

     factual

     statement

     about

     the

     given

     subject

    .

     In

     this

     case

    ,

     the

     subject

     is

     "

    France

    's

     capital

     city

    ".

     The

     statement

     should

     be

     a

     brief

    ,

     accurate

     description

     of

     where

     the

     capital

     of

     France

     is

     located

    .

     The

     response

     should

     not

     include

     any

     additional

     information

     not

     relevant

     to

     the

     given

     question

    .

     The

     response

     should

     be

     between

     

    4

    -

    6

     sentences

     long

    .

     The

     statement

     should

     be

     in

     English

    .

     Answer

     in

     English

    :

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     continued

     advances

     in

     the

     following

     areas

    :
    


    1

    .

     Better

     understanding

     of

     human

     behavior

    :

     As

     AI

     continues

     to

     learn

     and

     improve

    ,

     it

     is

     likely

     to

     be

     able

     to

     better

     understand

     human

     behavior

     and

     emotions

    .

     This

     could

     lead

     to

     more

     personalized

     and

     efficient

     interaction

     with

     users

    ,

     as

     well

     as

     more

     effective

     collaboration

     and

     communication

    .
    


    2

    .

     Increased

     reliance

     on

     AI

     in

     different

     industries

    :

     AI

     is

     already

     being

     used

     in

     a

     variety

     of

     industries

    ,

     from

     healthcare

     to

     finance

     to

     transportation

    .

     It

     is

     likely

     to

     continue

     to

     be

     an

     important

     tool

     for

     businesses

     and

     governments

     to

     improve

     their

     operations

     and

     make

     better

     decisions

    .
    


    3

    .

     Greater

     use

     of

     AI

     in

     autonomous

     vehicles

    :

     As

     self

    -driving

     cars

    



```python
llm.shutdown()
```
