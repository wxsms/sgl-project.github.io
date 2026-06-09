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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.14it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.27it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.27it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.27it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.27it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.27it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 23.27it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.27it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.27it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.27it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.27it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.27it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.97 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.96 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.95 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.95 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.94 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.94 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.96it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=56.93 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.93 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.93 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.93 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.93 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.92 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.92 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.92 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.92 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.91 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.90 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.00it/s]

    Capturing num tokens (num_tokens=960 avail_mem=56.91 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.00it/s] Capturing num tokens (num_tokens=896 avail_mem=56.91 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=832 avail_mem=56.89 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=832 avail_mem=56.89 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.20it/s]Capturing num tokens (num_tokens=768 avail_mem=56.88 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.20it/s]Capturing num tokens (num_tokens=704 avail_mem=56.88 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.20it/s]Capturing num tokens (num_tokens=640 avail_mem=56.87 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.20it/s]

    Capturing num tokens (num_tokens=576 avail_mem=56.87 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.20it/s]Capturing num tokens (num_tokens=576 avail_mem=56.87 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.23it/s]Capturing num tokens (num_tokens=512 avail_mem=56.36 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.23it/s]Capturing num tokens (num_tokens=480 avail_mem=56.29 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.23it/s]Capturing num tokens (num_tokens=448 avail_mem=56.29 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.23it/s]Capturing num tokens (num_tokens=416 avail_mem=56.29 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.23it/s]Capturing num tokens (num_tokens=384 avail_mem=56.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.23it/s]Capturing num tokens (num_tokens=384 avail_mem=56.28 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.93it/s]Capturing num tokens (num_tokens=352 avail_mem=56.28 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.93it/s]Capturing num tokens (num_tokens=320 avail_mem=56.27 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=288 avail_mem=56.27 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=256 avail_mem=56.27 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.93it/s]

    Capturing num tokens (num_tokens=240 avail_mem=56.26 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=240 avail_mem=56.26 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=224 avail_mem=56.26 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=208 avail_mem=56.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=192 avail_mem=56.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=176 avail_mem=56.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=160 avail_mem=56.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=160 avail_mem=56.25 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=144 avail_mem=56.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=128 avail_mem=56.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.41it/s]

    Capturing num tokens (num_tokens=112 avail_mem=56.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=96 avail_mem=56.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.41it/s] Capturing num tokens (num_tokens=80 avail_mem=56.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=80 avail_mem=56.22 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=64 avail_mem=56.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=48 avail_mem=56.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=32 avail_mem=56.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=28 avail_mem=56.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.33it/s]

    Capturing num tokens (num_tokens=24 avail_mem=56.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=24 avail_mem=56.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.08it/s]Capturing num tokens (num_tokens=20 avail_mem=56.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.08it/s]Capturing num tokens (num_tokens=16 avail_mem=56.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.08it/s]Capturing num tokens (num_tokens=12 avail_mem=56.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.08it/s]Capturing num tokens (num_tokens=8 avail_mem=56.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.08it/s] Capturing num tokens (num_tokens=4 avail_mem=56.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.08it/s]Capturing num tokens (num_tokens=4 avail_mem=56.17 GB): 100%|██████████| 58/58 [00:01<00:00, 35.36it/s]Capturing num tokens (num_tokens=4 avail_mem=56.17 GB): 100%|██████████| 58/58 [00:01<00:00, 34.52it/s]


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
    Generated text:  Gwendolyn and I'm the Director of Children's Services at the Law & Justice Center for Children. The children are under 14 years old.
    It seems that in the past year or so, the bullying incidents have been getting worse. There were incidents involving some of the kids not returning to school. Some kids were simply not being picked up by the parents in the morning. And when they got home, the home was full of other kids from the school. In addition, there was a lot of fighting in the home. Some of the kids were being treated as if they were not being allowed to be themselves.
    It's
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person who holds the most powerful position in the executive branch of the government of the United States. The president is the head of the federal government and is elected for a five-year term. The president's duties are to issue the executive orders, to nominate the members of the Cabinet, to fill vacancies in the Cabinet, and to veto bills passed by Congress. The president is also known as the "chief executive," "federal executive," or "emperor of the executive branch."
    Is the following statement correct based on the passage: "The President can veto bills passed by Congress and not be held in office for more than a year."
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the oldest city in the world, and it has the longest name in the world, the name of the capital of the United Kingdom is ____
    A. London
    B. Edinburgh
    C. Oxford
    D. London
    Answer: A
    
    Based on the following information, answer the questions. The overall demand for housing in a certain region is 180,000 units, and the supply is 140,000 units. The price elasticity of demand for housing is 0.5. When the price of housing is increased by 10%, the total revenue will change by ____.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  in the age of blockchain
    
    With the ongoing growth of the Internet of Things, many of us are seeing the ability to connect devices, such as sensors and microcontrollers, to form a smart grid that enables a safer environment.
    
    In the field of energy management, one of the main components is a smart grid that runs on the blockchain, as the blockchain technology can make it possible to store data securely and efficiently.
    
    Today, the smart grid is fully developed, but it will still require the collaboration of developers and industry stakeholders to ensure that it is fully compatible and complies with the rules and regulations of the IoT sector.
    
    As the Internet of Things


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity], and I'm always looking for new ways to explore and discover new things. What's your favorite book or movie? I love [book or movie], and I'm always looking for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also known for its rich history, art, and culture. Paris is a popular tourist destination and a major economic center in France. The city is home to many international organizations and events, including the World Cup and the Olympics. Paris is a vibrant and diverse city with a rich cultural heritage that continues to thrive today. The city is also known for its cuisine,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient solutions to complex problems.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be a greater need for privacy and security measures to protect against potential misuse of data. This could lead to the development of new technologies and protocols that are designed to protect user
    


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
    Generated text:  [Name], and I am a [Occupation] from [Location]. I have always loved [What your hobby or interest is]. I love [What you would like to share about yourself]. My favorite [Hobby or Skill] is [What it is]. I am also a [What is your job or occupation?]. I am here to [What you want to say to the reader]. How can I get to know you better? I would love to hear from you and discuss your interests and experiences. [Your Name]. It's a pleasure to meet you. What are your hobbies or interests? What do you like to do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as the "City of Love" due to its historical and cultural importance.
    
    Cite any sources supporting your statement. Paris, also known as the "City of Love," is the capital and largest city of France. According to historical and cultural references, it has a rich and diverse history, including being the birthplace of many famous French figures like Napoleon Bonaparte and Charles de Gaulle. The city is also known for its iconic Eiffel Tower and the vibrant nightlife, making it a popular tourist destination. Its status as the world's most populous city has also contributed to its global fame. The statement is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be more advanced, autonomous, and flexible than it is today. Here are some potential trends:
    
    1. Real-time AI: AI will be able to process and analyze data in real-time, with the ability to adapt and learn from new data.
    
    2. Deep learning: Neural networks that use deep learning will become more powerful, with faster training times and better performance.
    
    3. Explainable AI: AI will become more transparent and explainable, with better understanding of how it makes decisions.
    
    4. Virtual reality: AI-powered virtual assistants and devices will become more prevalent in everyday life, with improved communication and interaction.
    
    5. Interdisciplinary


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

    ].

     I

     am

     [

    insert

     occupation

    ]

     and

     I

     come

     from

     [

    insert

     nationality

    ].

     I

     am

     passionate

     about

     [

    insert

     why

     I

     love

     this

     occupation

    ].

     I

     am

     a

     [

    insert

     age

    ],

     [

    insert

     gender

    ],

     [

    insert

     nationality

    ]

     and

     I

     am

     [

    insert

     profession

    ].

     I

     have

     a

     [

    insert

     favorite

     hobby

    ]

     that

     I

     like

     to

     [

    insert

     how

     they

     spend

     their

     free

     time

    ].

     I

     am

     [

    insert

     any

     other

     information

     that

     might

     be

     relevant

     to

     your

     character

    ].

     I

    'm

     a

     [

    insert

     nationality

    ]

     and

     I

     live

     in

     [

    insert

     location

    ].

     I

     am

     currently

     [

    insert

     occupation

    ]

     and

     I

     enjoy

     [

    insert

     hobbies

     or

     activities

    ].

     In

     my

     free

     time

    ,

     I

     like

     to

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

     and

     the

     City

     of

     Fine

     Arts

    .
    


    How

     does

     Paris

     compare

     to

     other

     European

     capitals

    ?

     
    


    1

    .

     Brussels

     (

    City

     of

     Culture

     and

     Tourism

    ):

     Brussels

     is

     the

     capital

     of

     Belgium

     and

     has

     a

     population

     of

     approximately

     

    3

    4

    1

    ,

    0

    0

    0

    .

     It

     is

     known

     for

     its

     rich

     culture

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     its

     annual

     Van

     G

    ogh

     Week

    ,

     which

     celebrates

     the

     works

     of

     the

     famous

     Dutch

     painter

    .
    


    2

    .

     Amsterdam

     (

    City

     of

     Art

     and

     Science

    ):

     Amsterdam

     is

     the

     capital

     of

     the

     Netherlands

     and

     is

     known

     for

     its

     beautiful

     can

    als

    ,

     art

     galleries

    ,

     and

     scientific

     institutions

    .

     It

     also

     hosts

     the

     International

     Science

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     of

     the

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     industry

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     home

     automation

     to

     self

    -driving

     cars

    .

     This

     will

     enable

     more

     people

     to

     benefit

     from

     AI

     and

     reduce

     the

     need

     for

     traditional

     human

     intervention

    .
    


    2

    .

     AI

     will

     become

     more

     personalized

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     become

     more

     personalized

     to

     the

     needs

     of

     individual

     users

    .

     This

     will

     allow

     for

     more

     effective

     and

     efficient

     use

     of

     resources

    .
    


    3

    .

     AI

     will

     become

     more

     ethical

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     become

     more

     ethical

    



```python
llm.shutdown()
```
