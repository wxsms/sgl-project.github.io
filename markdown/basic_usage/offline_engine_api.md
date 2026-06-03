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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.84it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.69it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.43it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 18.12it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 25.80it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 35.29it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 35.29it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 35.29it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 35.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.14it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.53it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.21it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.11it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.15it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.15it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.68 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.15it/s]Capturing num tokens (num_tokens=704 avail_mem=76.68 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.15it/s]Capturing num tokens (num_tokens=640 avail_mem=76.67 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.15it/s]Capturing num tokens (num_tokens=640 avail_mem=76.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.40it/s]Capturing num tokens (num_tokens=576 avail_mem=76.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.40it/s]Capturing num tokens (num_tokens=512 avail_mem=76.65 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.40it/s]Capturing num tokens (num_tokens=480 avail_mem=76.58 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.40it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.58 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.13it/s]Capturing num tokens (num_tokens=448 avail_mem=76.58 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.13it/s]Capturing num tokens (num_tokens=416 avail_mem=76.58 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.13it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.01 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.13it/s]Capturing num tokens (num_tokens=384 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.80it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.80it/s]Capturing num tokens (num_tokens=320 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.80it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.80it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.80it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:01, 19.25it/s]

    Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=176 avail_mem=75.98 GB):  64%|██████▍   | 37/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=176 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.45it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.45it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.45it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.45it/s]Capturing num tokens (num_tokens=112 avail_mem=75.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.45it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.45it/s] Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 28.84it/s]

    Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  81%|████████  | 47/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.52it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.52it/s]Capturing num tokens (num_tokens=20 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.52it/s]Capturing num tokens (num_tokens=16 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.52it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.52it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.06it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.06it/s] Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.06it/s]

    Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:02<00:00, 27.21it/s]


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
    Generated text:  Alex and I am a computer science student. I am very passionate about computers and programming and have spent the last 5 years building my own computer and experimenting with programming languages. I am still working on learning more about computer science in general. I would like to ask you a question about computer science. What is your favorite programming language? I can help you with your questions if you tell me what programming language you are most familiar with. Let me know if you are interested in learning more about programming languages. If you want to ask me a question about computer science, just type your question in the box below. # Provide the correct answer:
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to take a trip to Hawaii or not. The president is considering the following options:
    
    1. To travel to Hawaii, the president would need to pay a $10,000 tax, which is half the cost of a vacation package. The vacation package costs $5,000.
    2. The president would take a vacation with the main team, which would cost $20,000 but also include a $1,000 monthly travel allowance.
    3. The president would take a vacation without a team, which would cost $20,000 but without any travel
    ===============================
    Prompt: The capital of France is
    Generated text: _____. A. Paris B. Lyon C. London D. Moscow
    A. Paris
    B. Lyon
    C. London
    D. Moscow
    Answer:
    A
    
    The capital of France is ____
    A. Paris
    B. Lyon
    C. London
    D. Moscow
    Answer:
    A
    
    When an aircraft is engaged in operations, the crew member in charge of communications should ensure that the ___ is properly prepared, accurately displayed, and clearly understood by the crew.
    A. Communication
    B. Communication Channel
    C. Communication Equipment
    D. Communication Strategy
    Answer:
    B
    
    The primary responsibility of the Crew Alert System is
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and it is imperative to understand the nuances of the industry. This course provides a thorough understanding of Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), Natural Language Processing (NLP), Natural Language Understanding (NLU), and the three broad areas of AI: Vision, Robotics, and Robotics. This course will also teach the participants the process of developing AI systems. Practical sessions will be held, during which hands-on experience with real-world AI projects will be provided. The course will be taught by experts who will cover topics like the difference between machine learning and artificial intelligence, an overview of the current AI landscape,


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


    Generated text:  Paris, the city that is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its cuisine, fashion, and art scene. The city is home to many famous French artists, writers, and musicians, and it is a popular destination for tourists from around the world. Paris is a city that is a true reflection of French culture and history. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased Integration of AI into Everyday Life: As AI becomes more integrated into our daily lives, we are likely to see more widespread adoption of AI technologies. This could include things like smart homes, self-driving cars, and virtual assistants that can assist with tasks like scheduling appointments, managing finances, and even ordering food.
    
    2. Greater Use of AI for Medical and Healthcare: AI is already being used in medical and healthcare settings to improve
    


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
    Generated text:  [insert name here]. I am an experienced [insert profession here], working as a [insert role here]. I have a passion for [insert something specific about my profession here]. I have always loved learning new things and expanding my knowledge. I am always looking for ways to improve myself and always have a sense of curiosity. I am passionate about [insert something specific about what I am passionate about here]. I am confident in myself and believe in my abilities. I enjoy working with people and building relationships. I am friendly and approachable, always ready to help and support others. I am a reliable and dependable friend to those I spend time
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville Blanche".
    Paris is France's largest city and the country's capital. The city is known for its historical architecture, vibrant culture, and annual festivals, including the World of Christmas. It is also famous for its fashion and gastronomy. The city is home to many iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Arc de Triomphe. It has been a center of government, culture, and diplomacy for centuries. Paris is a bustling hub of activity and entertainment for millions of visitors each year. It is the seat of government for the French Republic and the second
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends:
    
    1. Increased Efficiency: As AI becomes more accurate and capable, we can expect to see a significant increase in efficiency in various industries. For example, AI-powered systems can process vast amounts of data faster than human workers, reducing the need for manual data entry and improving decision-making processes.
    
    2. AI Will Be More Personalized: As AI learns to understand and respond to the unique needs and preferences of individuals, we can expect to see a significant increase in the level of personalization in technology products and services. This could include personalized recommendations, targeted advertising, and even the ability to create more realistic


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

     person

     who

     is

     [

    What

     you

     do

     or

     have

     done

    ].

     I

    'm

     a

     [

    What

     you

     do

     in

     your

     daily

     life

    ].

     I

    'm

     a

     [

    What

     your

     personality

     type

     is

    ]

     person

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     As

     a

     [

    What

     you

     do

     in

     your

     daily

     life

    ],

     I

     enjoy

     reading

     books

    ,

     playing

     sports

    ,

     and

     trying

     new

     things

    .

     I

     have

     a

     great

     sense

     of

     humor

    ,

     and

     I

     enjoy

     being

     in

     a

     loud

    ,

     energetic

     environment

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     As

     a

     [

    What

     you

     do

     in

     your

     daily

     life

    ],

     I

     enjoy

     reading

     books

    ,

     playing

     sports

    ,

     and

     trying

     new

     things

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     a

     bustling

     city

     known

     for

     its

     medieval

     architecture

    ,

     romantic

     atmosphere

    ,

     and

     diverse

     cultural

     scene

    .

     It

    's

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     art

     galleries

    ,

     as

     well

     as

     its

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     a

     center

     of

     global

     finance

     and

     politics

    ,

     with

     its

     government

     based

     in

     the

     nearby

     city

    -state

     of

     Paris

    .

     Paris

     is

     known

     for

     its

     rich

     culture

    ,

     beautiful

     landscapes

    ,

     and

     vibrant

     nightlife

    ,

     making

     it

     one

     of

     the

     world

    's

     most

     beloved

     cities

    .

     The

     French

     capital

     is

     a

     city

     of

     contrasts

     and

     a

     living

     museum

    ,

     showcasing

     the

     diversity

     and

     unique

     character

     of

     the

     French

     people

    .

     According

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     an

     increasing

     focus

     on

     developing

     more

     advanced

     models

     and

     technologies

     that

     can

     handle

     increasingly

     complex

     and

     varied

     tasks

    .

     This

     may

     include

     advancements

     in

     natural

     language

     processing

    ,

     computer

     vision

    ,

     and

     deep

     learning

    ,

     as

     well

     as

     the

     development

     of

     new

     hardware

     and

     software

     solutions

     that

     can

     process

     and

     analyze

     vast

     amounts

     of

     data

    .

     Additionally

    ,

     the

     increasing

     use

     of

     AI

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

     may

     lead

     to

     further

     innovation

     and

     development

    .

     Overall

    ,

     the

     future

     of

     AI

     is

     likely

     to

     be

     one

     of

     continued

     growth

     and

     development

    ,

     as

     well

     as

     continued

     refinement

     and

     improvement

    .

    



```python
llm.shutdown()
```
