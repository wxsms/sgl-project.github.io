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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.36it/s]


    2026-05-17 09:33:22,466 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 09:33:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.54it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.17it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.36it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:04, 13.16it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.01 GB):   3%|▎         | 2/58 [00:00<00:04, 13.16it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.01 GB):   3%|▎         | 2/58 [00:00<00:04, 13.16it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:04, 11.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:04, 11.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.00 GB):   7%|▋         | 4/58 [00:00<00:04, 11.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.00 GB):  10%|█         | 6/58 [00:00<00:03, 13.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.50 GB):  10%|█         | 6/58 [00:00<00:03, 13.17it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=70.49 GB):  10%|█         | 6/58 [00:00<00:03, 13.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.33 GB):  10%|█         | 6/58 [00:00<00:03, 13.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.33 GB):  10%|█         | 6/58 [00:00<00:03, 13.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.33 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.32 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.32 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.32 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.32 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.31 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.81it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=70.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.30 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.28 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.00it/s]Capturing num tokens (num_tokens=960 avail_mem=70.29 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.00it/s] Capturing num tokens (num_tokens=896 avail_mem=70.29 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.00it/s]Capturing num tokens (num_tokens=832 avail_mem=70.29 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.00it/s]Capturing num tokens (num_tokens=768 avail_mem=70.28 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.00it/s]Capturing num tokens (num_tokens=768 avail_mem=70.28 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=704 avail_mem=70.28 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.42it/s]

    Capturing num tokens (num_tokens=640 avail_mem=70.28 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=576 avail_mem=70.28 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=512 avail_mem=70.26 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=480 avail_mem=70.28 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=448 avail_mem=70.28 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=448 avail_mem=70.28 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=416 avail_mem=70.27 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=384 avail_mem=70.27 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=352 avail_mem=70.27 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=320 avail_mem=70.26 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=288 avail_mem=70.26 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.25it/s]

    Capturing num tokens (num_tokens=288 avail_mem=70.26 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=256 avail_mem=70.26 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=240 avail_mem=70.25 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=224 avail_mem=70.25 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=208 avail_mem=70.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=192 avail_mem=70.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=176 avail_mem=70.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=176 avail_mem=70.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=160 avail_mem=70.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=144 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=128 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=112 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s] Capturing num tokens (num_tokens=96 avail_mem=70.23 GB):  81%|████████  | 47/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=80 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=64 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=48 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=32 avail_mem=70.21 GB):  81%|████████  | 47/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=28 avail_mem=70.21 GB):  81%|████████  | 47/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=28 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=24 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=20 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=16 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=12 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.29it/s]

    Capturing num tokens (num_tokens=8 avail_mem=70.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.29it/s] Capturing num tokens (num_tokens=8 avail_mem=70.19 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.94it/s]Capturing num tokens (num_tokens=4 avail_mem=70.19 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.94it/s]Capturing num tokens (num_tokens=4 avail_mem=70.19 GB): 100%|██████████| 58/58 [00:01<00:00, 37.05it/s]


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
    Generated text:  Maria. I'm a student of the 3rd grade. I like to listen to music and play my guitar. My favorite instrument is the guitar because I like to sing and play my guitar. It makes me happy. I have a big family of 4. They have three sons and a little boy. My father is a teacher. He works in a school. I have a nice school. I go to school every day. It is a very nice school. There is a playground on the playground. There are three big trees on the playground. And there is a big sports field outside the school. I like the playground.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a hereditary position. There are 10 representatives from each state and two additional representatives from the District of Columbia. What is the probability of picking a representative from the District of Columbia? Express your answer as a common fraction.
    
    To determine the probability of picking a representative from the District of Columbia, we need to follow these steps:
    
    1. Identify the total number of representatives.
    2. Identify the number of representatives from the District of Columbia.
    3. Calculate the probability by dividing the number of representatives from the District of Columbia by the total number of representatives.
    
    First, let's identify the total number of representatives. The problem states that there
    ===============================
    Prompt: The capital of France is
    Generated text:  located on which continent?
    A. North America
    B. South America
    C. Europe
    D. Asia
    Answer: C
    
    In an HTML document, which tag is used to set the title of the page?
    A. <body>
    B. <title>
    C. <table>
    D. <img>
    Answer: B
    
    Which of the following is a common feature of all e-commerce platforms?
    A. Strong online shopping capabilities
    B. Support for international transactions
    C. Instant delivery service
    D. Unique user interface
    Answer: B
    
    Which of the following statements about the capital of France is incorrect?
    A. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  everywhere, from the smart homes of the future, to the chatbots and virtual assistants of the present. In 2020, a paper titled "The Future of Artificial Intelligence: Challenges, Opportunities, and Solutions" by the International Council on Industrial Electrotechnology and the Future of Industrial Electronics Foundation revealed how the future of AI is not just about the latest gadgets, but about the future of work. AI is making a huge impact on the workplace, and the following are some of the ways AI is changing the workplace:
    
    1. Chatbots and Virtual Assistants
    
    Chatbots are becoming more advanced and intelligent every day. These


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


    Generated text:  [Name] and I'm a [occupation] who has been working in [field] for [number] years. I'm passionate about [reason for passion] and I'm always looking for ways to [action or achievement]. I'm [age] years old and I'm [gender] and I have [number] children. I'm [occupation] and I'm [number] years old. I'm [name] and I'm [occupation] and I'm [number] years old. I'm [name] and I'm [occupation] and I'm [number] years old. I'm [name] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its cuisine, fashion, and art, and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also home to a diverse population, with many French people and immigrants living there. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, such as manufacturing, transportation, and healthcare. This will lead to increased efficiency, reduced costs, and improved quality of life for many people.
    
    2. AI-powered healthcare: AI is already being used to improve the accuracy and speed of medical diagnosis and treatment. As AI technology continues to advance, we can expect to see even more sophisticated
    


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
    Generated text:  [Name], and I'm here to help with my journey to becoming a better, more confident person.
    
    If you're not sure how I came to be here, my path began back when I was a teenager growing up in a small town in [Country]. I was a shy, introverted child, and it took a lot of hard work and constant practice to develop my social skills and confidence. I've always had a soft spot for animals, and as an adult, I've become a pet lover and animal advocate.
    
    I believe in giving back to the community through my work as a volunteer, so I've taken on the challenge of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower and the Louvre Museum. It is also home to some of the world’s most famous restaurants, fashion houses, and art museums. Paris has a strong economy and is an important center for research, education, and business. Its population has grown to over 2.5 million people, making it the most populous city in Europe. It is known for its romantic and cultural atmosphere, and has been a UNESCO World Heritage Site since 1985. Paris is a popular tourist destination and a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant advancements in both hardware and software. Some possible trends include:
    
    1. Increased use of AI in healthcare: As AI becomes more sophisticated, it will be able to analyze large amounts of medical data to identify patterns that may be overlooked by humans. This could lead to earlier detection of diseases, better treatment of existing conditions, and personalized medicine.
    
    2. AI in finance: With the increasing use of digital assets like cryptocurrencies, there will be a growing need for AI-driven risk management tools. AI will be able to analyze financial data in real-time and identify fraud and other risks, helping to protect investors and banks.
    
    3


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

    job

     title

     or

     position

    ]

     with

     over

     [

    number

     of

     years

    ]

     years

     of

     experience

    .

     I

     am

     passionate

     about

     [

    specific

     field

     or

     area

     of

     interest

    ],

     and

     I

    'm

     confident

     in

     my

     ability

     to

     [

    describe

     an

     accomplishment

     or

     achievement

    ].

     I

    'm

     always

     looking

     for

     opportunities

     to

     [

    describe

     a

     new

     challenge

     or

     project

    ],

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

    .

     I

     have

     a

     strong

     work

     ethic

     and

     a

     positive

     attitude

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     contribute

     to

     [

    specific

     community

     or

     cause

    ].

     I

     thrive

     on

     [

    describe

     a

     positive

     attribute

     or

     trait

    ].

     I

     look

     forward

     to

     [

    describe

     the

     next

     step

     or

     next

     opportunity

     you

     will

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historical

     and

     cultural

     center

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     
    


    (Note

    :

     The

     given

     answer

     is

     a

     general

     factual

     statement

     that

     does

     not

     include

     a

     specific

     city

     name

     or

     its

     capital

    .)

     
    


    Some

     more

     detailed

     information

    :
    


    -

     Paris

     is

     the

     largest

     city

     in

     France

     by

     population

    ,

     with

     a

     population

     of

     approximately

     

    2

    ,

    1

    8

    4

    ,

    0

    0

    0

     as

     of

     

    2

    0

    2

    1

    .


    -

     It

     is

     located

     on

     the

     Left

     Bank

     of

     the

     Se

    ine

    ,

     about

     

    1

    1

     kilometers

     (

    7

     miles

    )

     east

     of

     the

     River

     Se

    ine

    .


    -

     Paris

     has

     a

     rich

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     involves

     a

     multitude

     of

     potential

     developments

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Improved

     general

    ization

     and

     adapt

    ability

    :

     As

     AI

     models

     are

     trained

     on

     more

     data

    ,

     they

     become

     better

     at

     general

    izing

     and

     adapting

     to

     new

     situations

    ,

     which

     could

     lead

     to

     more

     complex

     and

     adaptive

     systems

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

    -powered

     diagnostic

     tools

    ,

     personalized

     medicine

    ,

     and

     artificial

     intelligence

     in

     AI

    -ass

    isted

     surgery

     could

     significantly

     improve

     healthcare

     outcomes

    .
    


    3

    .

     Increased

     reliance

     on

     AI

     in

     transportation

    :

     AI

    -powered

     autonomous

     vehicles

    ,

     smart

     traffic

     management

    ,

     and

     AI

    -ass

    isted

     ride

    -h

    ailing

     could

     revolution

    ize

     the

     transportation

     industry

    .
    


    4

    .

     Enhanced

     AI

     ethics

     and

     privacy

    



```python
llm.shutdown()
```
