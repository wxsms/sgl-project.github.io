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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.83it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.83it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.83it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.83it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.83it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.83it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.83it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.83it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 21.16it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 29.48it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:05<00:00, 29.48it/s] Compiling num tokens (num_tokens=4):  79%|███████▉  | 46/58 [00:05<00:00, 29.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 42.63it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 18.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 18.66it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 18.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 18.66it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.56it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.56it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.56it/s]Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.11it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.11it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.11it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.11it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.11it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.76it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.76it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.76it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.12it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.12it/s] Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.42it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.42it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.42it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.42it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.42it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.52it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 34.50it/s]


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
    Generated text:  Jourdan. I am 17 years old and I have been playing video games since I was 6 years old. I have two brothers and I am the youngest of the two. What games are they playing? I'm not sure.
    You didn't mention your brothers' names, so I can't provide information about their specific games. However, most boys between the ages of 13 and 16 who play video games have a variety of genres, including action, adventure, first-person, simulation, and puzzle games. Some boys may also play games with more complex narratives or storytelling, which often include themes of friendship
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use the military to solve an economic crisis. The president is considering four options: Option A: increase taxes, Option B: increase public spending, Option C: decrease taxes, and Option D: decrease public spending. The president believes that using the military to solve an economic crisis can lead to more job creation and economic growth, but there is a risk of additional costs and losses. He wants to calculate the expected cost of each option and determine which one the president should choose.
    
    Calculate the expected cost for each option based on the given criteria:
    
    - Increase taxes: 30% chance of causing additional cost and losses
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population in 2018 was 2.16 million. How many million people live in Paris? Let's think: We know that 2.16 million is equal to 216,000. To convert this to millions, we need to divide by 1,000,000. So 2.16 million / 1,000,000 = 0.0216 millions. That means 0.0216 of a million is equal to 216,000 people. Therefore,
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be shaped by three big forces: the increase in the number of neural networks, the rapid growth of the internet of things (IoT), and the increased complexity of the data we store. The impact of these forces is being felt across a wide range of industries, from healthcare to finance to retail. In this blog post, we will explore the potential of AI in healthcare, how it can revolutionize patient care, and the challenges it presents. We will also look at how AI is being used to improve data privacy and security, and how AI can help to identify potential risks and vulnerabilities in the data we store. While AI is


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love About My Profession]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Favorite Hobby] and I enjoy [What I Like to Do]. I'm always looking for ways to improve myself and make the world a better place. I'm excited to meet you and learn more about you. [Name] [Age] [Occupation] [Skill] [Favorite Hobby] [What I Like
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, with a rich history dating back to the Roman Empire. Paris is a bustling city with a diverse population and a strong sense of French identity. It is a popular tourist destination and a cultural hub for the world. The city is also home to many international organizations and institutions, including UNESCO and the European Union. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is a growing emphasis on developing AI that is more ethical and responsible. This could mean developing AI that is designed to minimize harm to individuals and society as a whole, or that is designed to be transparent and accountable.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and there is a growing expectation that it will be used even more
    


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
    Generated text:  [Name]. I'm a [Type] who have [number of years] years of experience in [field of interest]. I'm currently [status]. I love [interest or hobby] and I'm always eager to learn new things. I'm friendly, patient, and always ready to help others. I'm [number of books] years of reading background and I'm always looking for new ideas and inspiration to keep my work fresh and engaging. I'm excited to meet you and discuss how I can help you achieve your goals. When you're done, you can say, "Hello [Name] and I'm [Your Name].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Sorcière". Its name comes from a fairy tale and is considered the birthplace of the city. Paris is a vibrant and diverse city with a rich cultural and historical heritage, including the Louvre Museum, the Eiffel Tower, the Notre-Dame Cathedral, and the Champs-Élysées. Its urban landscape is shaped by its picturesque medieval walls and modern architecture, making it a city of contrasts and beauty. Paris has been the capital of France since 1804 and is considered a unique and captivating city with a strong sense of Parisian identity and heritage. 
    
    (Note:
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, with many possibilities for breakthroughs and innovations. Here are some possible trends that could shape the development of AI in the coming years:
    
    1. Improved Natural Language Processing (NLP): As more and more tasks require the ability to understand and generate human-like language, the field of NLP is likely to continue to evolve. AI systems will be able to better understand and respond to natural language queries, and will be able to generate human-like responses to prompts.
    
    2. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient care, predict disease outbreaks, and identify potential treatments. As AI technology continues to


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

    /an

     [

    Job

     Title

     or

     Role

    ].

     I

     have

     always

     been

     [

    insert

     a

     personal

     trait

     or

     qualification

     here

    ,

     such

     as

     "

    ded

    icated

    ",

     "

    creative

    ",

     "

    hard

    working

    ",

     etc

    .]

     and

     have

     been

     working

     with

     [

    insert

     a

     relevant

     work

    -related

     company

     or

     organization

    ].

     I

     am

     [

    insert

     a

     personality

     trait

     or

     characteristic

     of

     a

     character

    ,

     such

     as

     "

    fun

    -loving

    ",

     "

    hum

    orous

    ",

     etc

    .]

     and

     enjoy

     [

    insert

     an

     activity

     or

     hobby

    ,

     such

     as

     "

    playing

     sports

    ",

     "

    s

    ailing

    ",

     "

    reading

    ",

     etc

    .

    ].

     I

     am

     passionate

     about

     [

    insert

     a

     hobby

     or

     interest

    ,

     such

     as

     "

    phot

    ography

    ",

     "

    music

    ",

     "

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     renowned

     for

     its

     rich

     history

     and

     art

     scene

    ,

     and

     is

     a

     popular

     tourist

     destination

     for

     visitors

     from

     all

     over

     the

     world

    .

     Paris

     has

     a

     vibrant

     and

     diverse

     culture

    ,

     with

     many

     different

     ethnic

     groups

     and

     traditions

     within

     its

     urban

     landscape

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     galleries

    ,

     and

     theaters

    ,

     as

     well

     as

     a

     variety

     of

     restaurants

     and

     cafes

     that

     cater

     to

     various

     tastes

     and

     preferences

    .

     Overall

    ,

     Paris

     is

     a

     beautiful

     and

     unique

     city

     that

     is

     a

     must

    -

    visit

     for

     anyone

     looking

     to

     experience

     France

     and

     its

     cultural

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     highly

     unpredictable

     and

     challenging

    ,

     with

     many

     potential

     developments

     that

     could

     impact

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     technology

    .

     Some

     potential

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     specialization

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     be

     able

     to

     perform

     a

     wider

     range

     of

     tasks

    ,

     from

     autom

    ating

     mundane

     tasks

     to

     developing

     new

     technologies

    .

     This

     could

     lead

     to

     a

     shift

     towards

     more

     specialized

     AI

    ,

     where

     AI

     is

     used

     to

     solve

     specific

     problems

     or

     tasks

     that

     require

     a

     high

     level

     of

     expertise

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Self

    -driving

     cars

    ,

     drones

    ,

     and

     other

     autonomous

     vehicles

     are

     already

     becoming

     increasingly

     common

    ,

     and

     it

     is

     likely

     that

     AI

     will

     play

     a

     larger

     role

    



```python
llm.shutdown()
```
