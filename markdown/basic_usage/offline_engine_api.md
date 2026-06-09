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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:04,  8.15it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]

    Compiling num tokens (num_tokens=640):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=256):  47%|████▋     | 27/58 [00:04<00:02, 14.48it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]

    Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 22.64it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 28.77it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 28.77it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 28.77it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 28.77it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 28.77it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 28.77it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 28.77it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 37.94it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 37.94it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 37.94it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 37.94it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.51 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.48 GB):   3%|▎         | 2/58 [00:00<00:03, 15.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.48 GB):   3%|▎         | 2/58 [00:00<00:03, 15.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.48 GB):   3%|▎         | 2/58 [00:00<00:03, 15.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 15.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.26it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.85it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.85it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 40.37it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.29it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.44it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.68it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.10it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 37.72it/s]


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
    Generated text:  Megan. I'm 24 years old and I'm a single mother of a 12-year-old son. We are planning to move to Hawaii for summer vacation. I would like to ask the question for a list of hotels that have a swimming pool, and also, do they have a business center? Thank you very much for your help and support.
    
    As a single mom with a 12-year-old son and a plan to move to Hawaii for summer vacation, I would like to inquire about hotels that offer a swimming pool and a business center for the hotel. I would also like to know if the hotel has any specific
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 45 years old. If the president of the United States is currently 4 times older than the president of Brazil, how old is the president of Brazil?
    To determine the age of the president of Brazil, we start by identifying the given information and setting up the equation accordingly.
    
    1. The president of the United States is currently 45 years old.
    2. The president of the United States is currently 4 times older than the president of Brazil.
    
    Let's denote the age of the president of Brazil as \( x \).
    
    According to the problem, the president of the United States is 4 times older than the president
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) London
    C) Rome
    D) Madrid
    E) Brussels
    
    To determine the capital of France, let's consider the following points:
    
    1. The capital of France is Paris.
    2. Paris is not only the capital of France but also the largest city in France.
    3. Madrid is the capital of Spain, which is not France.
    
    Given these points, the capital of France is Paris. Therefore, the correct answer is:
    
    A) Paris
    
    So, the final answer is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be built in the world of the digital workplace. With the deployment of Artificial Intelligence and big data, many organizations have begun to improve the efficiency of the operations by using it. However, at the same time, it has caused some negative effects on the human being. This article will help you to understand the impact of AI and big data on the human being.
    
    The Smart Society
    
    The future of AI and big data will be the smart society. This society will be developed by utilizing big data and artificial intelligence. This society will be based on a more intelligent system. The smart society will be a society that is built by AI and


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


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is located on the Seine River and is the seat of government for the French Republic. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also home to many famous museums, including the Musée d'Orsay, the Musée Rodin, and the Musée d'Orsay. Paris is a cultural and historical center that is home to many world-renowned artists, writers, and musicians. It is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud
    


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
    Generated text:  Sarah, and I'm a creative writer specializing in short stories. I have a degree in English and have been writing since I was 16. I enjoy experimenting with different writing styles, from romance to horror, and I'm always looking for new and unique writing prompts to try out. I'm a firm believer in the importance of good storytelling and am always looking for ways to share my own experiences and those of others. I'm passionate about helping others find their own voice and make their stories their own. I believe in the power of words to connect people and to create meaningful connections. I'm excited to explore more ways to help you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historical landmarks, vibrant culture, and annual fashion and sports events. Can you provide some additional information on the city's unique charm or historical significance that makes it a must-visit destination for visitors? Yes, Paris is known for its historical landmarks such as Notre-Dame Cathedral, Louvre Museum, Arc de Triomphe, and the Seine River. It is also famous for its fashion and sports events, including the Olympics and World Cup soccer. Paris is also a cultural and historical center, with many museums, art galleries, and theaters. It is also known for its cuisine, particularly its famous cheese, Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, as the field is constantly evolving and changing. However, some potential trends that are likely to shape the AI landscape in the next few years include:
    
    1. Increased reliance on machine learning: More companies are moving towards implementing machine learning and other forms of artificial intelligence to improve their operations and reduce costs. As AI continues to become more sophisticated, it is likely to play a bigger role in decision-making and problem-solving.
    
    2. Increased automation: The automation of certain tasks, such as manufacturing and transportation, is expected to increase in the coming years. AI is expected to continue to be used in these areas, but it will likely be


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

     a

     [

    title

    ]

     at

     [

    company

     name

    ].

     I

     am

     passionate

     about

     [

    your

     interest

     or

     career

     goal

    ].

     I

     am

     always

     looking

     for

     ways

     to

     [

    your

     passion

     or

     goal

    ]

     and

     I

     am

     eager

     to

     share

     my

     knowledge

     and

     skills

     with

     others

    .

     I

     am

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ],

     and

     I

     thrive

     on

     collaboration

     and

     teamwork

    .

     I

     love

     trying

     new

     things

     and

     exploring

     new

     challenges

    ,

     and

     I

     am

     always

     looking

     for

     ways

     to

     grow

     and

     improve

    .

     I

     am

     a

     [

    interest

     or

     hobby

    ]

     at

     [

    related

     hobby

     or

     activity

    ],

     and

     I

     enjoy

     learning

     from

     others

     and

     sharing

     my

     experiences

    .

     I

     am

     always

     eager

     to

     learn

     and

     grow

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    )

     Incorrect

    


    B

    )

     Correct

    


    B

    )

     Correct

    
    


    Paris

     is

     the

     capital

     of

     France

    ,

     which

     is

     located

     in

     the

     south

     of

     the

     country

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

     beautiful

     architecture

    ,

     and

     diverse

     culture

    .

     It

     is

     one

     of

     the

     world

    's

     most

     populous

     cities

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

    .

    1

     million

     inhabitants

    .

     Paris

     is

     home

     to

     many

     iconic

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

     The

     city

     also

     has

     a

     vibrant

     nightlife

     and

     fashion

     scene

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     As

     the

     political

     and

     cultural

     center

     of

     France

    ,

     Paris

     plays

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     undoubtedly

     changing

    ,

     and

     we

     can

     expect

     many

     exciting

     trends

     to

     develop

     as

     technology

     advances

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

     AI

    :
    


    1

    .

     AI

     will

     become

     more

     accessible

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     that

     it

     will

     become

     more

     accessible

     to

     the

     general

     public

    .

     This

     means

     that

     we

     will

     see

     more

     AI

    -powered

     tools

     and

     services

     that

     are

     accessible

     to

     people

     of

     all

     backgrounds

    ,

     ages

    ,

     and

     demographics

    .

     This

     will

     help

     to

     democrat

    ize

     access

     to

     AI

     technology

     and

     provide

     people

     with

     the

     tools

     and

     resources

     they

     need

     to

     make

     informed

     decisions

    .
    


    2

    .

     AI

     will

     be

     more

     integrated

     into

     our

     daily

     lives

    :

     As

     AI

     technology

     continues

     to

     evolve

    ,

     it

     will

     be

    



```python
llm.shutdown()
```
