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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]

    Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 11.82it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]

    Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 18.41it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 25.95it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]

    Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 44.31it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 44.31it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 44.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=960 avail_mem=58.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s] Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]

    Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.85it/s]Capturing num tokens (num_tokens=768 avail_mem=55.87 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.85it/s]Capturing num tokens (num_tokens=704 avail_mem=55.81 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.85it/s]Capturing num tokens (num_tokens=640 avail_mem=53.08 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.85it/s]Capturing num tokens (num_tokens=576 avail_mem=53.08 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.85it/s]Capturing num tokens (num_tokens=512 avail_mem=53.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.85it/s]Capturing num tokens (num_tokens=512 avail_mem=53.07 GB):  50%|█████     | 29/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=480 avail_mem=53.08 GB):  50%|█████     | 29/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=448 avail_mem=53.08 GB):  50%|█████     | 29/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=416 avail_mem=53.08 GB):  50%|█████     | 29/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=384 avail_mem=53.08 GB):  50%|█████     | 29/58 [00:00<00:00, 42.70it/s]

    Capturing num tokens (num_tokens=352 avail_mem=53.07 GB):  50%|█████     | 29/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=352 avail_mem=53.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=320 avail_mem=53.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=288 avail_mem=53.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=256 avail_mem=53.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=240 avail_mem=53.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=224 avail_mem=53.05 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=224 avail_mem=53.05 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.22it/s]Capturing num tokens (num_tokens=208 avail_mem=53.05 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.22it/s]Capturing num tokens (num_tokens=192 avail_mem=53.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.22it/s]Capturing num tokens (num_tokens=176 avail_mem=53.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.22it/s]Capturing num tokens (num_tokens=160 avail_mem=53.04 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.22it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.04 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.22it/s]Capturing num tokens (num_tokens=144 avail_mem=53.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=128 avail_mem=53.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=112 avail_mem=53.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=96 avail_mem=53.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.25it/s] Capturing num tokens (num_tokens=80 avail_mem=53.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=64 avail_mem=53.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=64 avail_mem=53.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=48 avail_mem=53.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=32 avail_mem=53.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=28 avail_mem=53.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=24 avail_mem=53.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]

    Capturing num tokens (num_tokens=20 avail_mem=53.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=20 avail_mem=53.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=16 avail_mem=53.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=12 avail_mem=53.00 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=8 avail_mem=53.00 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s] Capturing num tokens (num_tokens=4 avail_mem=52.99 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=4 avail_mem=52.99 GB): 100%|██████████| 58/58 [00:01<00:00, 42.04it/s]


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
    Generated text:  "Ivan", I’m a Senior Engineer at DigitalOcean and I work on a team of 13 people that are our "digital" team. I work on large, complex projects that involve a wide variety of technologies. I lead the development of our "digital" platform and I work with clients to help them grow their business with digital products. Before joining DigitalOcean, I worked in a number of roles at Google and worked on a number of projects. I've had experience with several different technologies, including Docker, Kubernetes, and other cloud platforms, and I've worked with a variety of clients to help them optimize their use of
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to cut healthcare funding for the federal government or not. He knows that:
    
    a) The cost of healthcare is 20% of GDP.
    
    b) The cost of a standard healthcare package is $400 per person for a year.
    
    c) The government has $10,000,000,000,000 in annual revenues from taxes and $40,000,000,000,000 in annual costs in healthcare.
    
    What is the minimum amount of money the president should allocate to funding healthcare to avoid deficit spending, if
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. As the capital, it is located at the southern end of the island of France. The capital is situated in the North of the country. It is the political, economic and cultural centre of France.
    The capital city of France is Paris. This is the capital city in the southern end of the island of France, in the northern part of the country. Paris is the political, economic and cultural centre of France. The capital city of France is located in the northern end of the island of France.
    The capital of France, Paris, was originally named Parthenon. It was founded by the Greeks and started as a theatre.
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and an emerging market, but the science behind it is growing rapidly. By embracing new technologies and exploring their potential, businesses can revolutionize their operations and improve their customer experiences.
    In this blog post, we'll explore how AI is changing the way we do business and how businesses can take advantage of these new technologies. We'll also look at some of the challenges and opportunities that lie ahead, and offer some advice for businesses looking to embrace AI and stay ahead of the curve.
    AI is rapidly changing the business landscape, and we're just scratching the surface of its potential. But with the right strategies, businesses can harness the power of AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    This statement encapsulates the key features of Paris, including its historical significance, architectural landmarks, and cultural attractions, making it a concise yet comprehensive overview of the city's importance. 
    
    To further emphasize, the statement could be expanded to include additional details such as the city's role in French politics, its status as a major tourist destination, or its role in French cuisine and gastronomy. However, the original statement provides a clear and concise overview of Paris's importance and cultural significance. 
    
    Overall, the statement effectively communicates the key aspects
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some possible future trends in AI:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to customer service. We may see more automation in industries such as manufacturing, healthcare, and transportation, where machines can perform tasks that were previously done by humans.
    
    2. Improved privacy and security: As AI becomes more integrated into our lives, there will be a need for better privacy and security measures to protect our data and personal information. This may lead to the
    


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
    Generated text:  [Name], and I am a [Name]! I have always been curious about the world around me and have always wanted to learn more. I am a [Name] with a passion for [Name] and a deep understanding of [Name]. I am always looking for new experiences and learning opportunities, and I am eager to embrace them. I am a [Name] who is always seeking out new ways to connect with others and make a positive impact in the world. I am a [Name] who is passionate about [Name] and I am always ready to help others achieve their goals. I am [Name] who loves to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France, located on the northern coast of the French island of Corsica, and is the capital of France. It is known for its rich history, art, and cuisine. The city is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many other famous landmarks. Paris is also a popular tourist destination, known for its fashion, gastronomy, and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, with many possibilities and potential areas of focus. Here are some potential trends that could shape the AI landscape in the coming years:
    
    1. Increased Integration with Natural Language Processing (NLP): One of the most promising areas of AI is the integration of NLP with AI systems. This could allow machines to understand and interpret human language in a way that is more nuanced and contextually aware than current systems. This could lead to more sophisticated language-based decision-making and applications, such as chatbots and virtual assistants.
    
    2. Increased Integration with Perception: AI systems are becoming more capable of detecting and understanding the world around them. As


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

    ].

     I

    ’m

     a

     [

    type

     of

     character

    ]

     who

     has

     a

     [

    interest

     or

     hobby

    ].

     I

    ’m

     confident

    ,

     competent

    ,

     and

     passionate

     about

     [

    what

     you

    ’d

     like

     to

     share

     with

     the

     world

    ].

     I

    ’m

     known

     for

     my

     [

    aspect

     of

     my

     character

     that

     makes

     me

     stand

     out

    ].

     Whether

     I

    'm

     working

     on

     a

     project

    ,

     giving

     a

     presentation

    ,

     or

     engaging

     in

     a

     conversation

    ,

     I

     always

     strive

     to

     [

    mention

     a

     positive

     trait

     you

     have

    ].

     I

    ’m

     committed

     to

     [

    what

     you

    ’d

     like

     to

     accomplish

     in

     the

     future

    ],

     and

     I

    ’m

     always

     ready

     to

     learn

     and

     grow

    .

     I

     believe

     in

     [

    why

     you

     believe

     in

     yourself

    ],

     and

     I

    ’m

     always

     eager

     to

     help

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    (Note

    :

     This

     answer

     is

     based

     on

     the

     correct

     factual

     statement

    .

     There

     is

     no

     additional

     information

     to

     add

     to

     it

     beyond

     the

     given

     information

    .

     )

     
    


    Now

    ,

     I

     would

     like

     you

     to

     have

     the

     task

     of

     re

    ph

    rasing

     this

     statement

     into

     a

     comparative

     statement

     that

     highlights

     the

     similarity

     between

     Paris

     and

     another

     capital

     city

     in

     France

    ,

     but

     only

     using

     "

    Paris

    "

     as

     the

     main

     subject

    .

     Can

     you

     do

     that

    ?

     Yes

    ,

     I

     can

     re

    phrase

     the

     statement

     into

     a

     comparative

     statement

     using

     "

    Paris

    "

     as

     the

     main

     subject

    .

     Here

    's

     one

     possible

     re

    ph

    r

    ased

     statement

    :

     "

    Paris

     is

     the

     capital

     of

     France

    ."

     
    


    Now

    ,

     I

     would

     like

     you

     to

     have

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     concerns

     about

     AI

    's

     potential

     impact

     on

     society

     increase

    ,

     there

     is

     a

     growing

     emphasis

     on

     developing

     AI

     that

     is

     transparent

    ,

     accountable

    ,

     and

     respects

     human

     rights

    .

     This

     includes

     developing

     AI

     that

     can

     operate

     within

     ethical

     frameworks

     and

     standards

    ,

     and

     ensuring

     that

     AI

     systems

     are

     designed

     and

     developed

     with

     the

     input

     of

     diverse

     perspectives

     and

     expertise

    .
    


    2

    .

     Continued

     development

     of

     deep

     learning

     techniques

    :

     Deep

     learning

     is

     a

     key

     area

     of

     AI

     research

    ,

     and

     there

     is

     likely

     to

     be

     continued

     development

     of

     more

     powerful

     and

     sophisticated

     models

     that

     can

     perform

     tasks

     that

     were

     previously

     impossible

     with

     existing

     algorithms

    .
    


    3

    .

     Increased

    



```python
llm.shutdown()
```
