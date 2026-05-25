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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.33it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.33it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.33it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.33it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.33it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.33it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.33it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.33it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:09,  3.00it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:09,  3.00it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:09,  3.00it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:09,  3.00it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:09<00:09,  3.00it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:09<00:09,  3.00it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:09<00:06,  4.06it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:09<00:06,  4.06it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:09<00:06,  4.06it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:09<00:06,  4.06it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:09<00:06,  4.06it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:09<00:06,  4.06it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:09<00:03,  5.46it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:09<00:03,  5.46it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:09<00:03,  5.46it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:09<00:03,  5.46it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:09<00:03,  5.46it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:09<00:03,  5.46it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:09<00:03,  5.46it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:09<00:03,  5.46it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:09<00:01,  8.18it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:09<00:01,  8.18it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:09<00:01,  8.18it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:09<00:01,  8.18it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:09<00:01,  8.18it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:10<00:01,  8.18it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:10<00:01,  8.18it/s]

    Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:10<00:01,  8.18it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:10<00:01,  8.18it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:10<00:01,  8.18it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:10<00:01,  8.18it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:10<00:00, 13.41it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:10<00:00, 13.41it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:10<00:00, 13.41it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:10<00:00, 13.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.35 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.34 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.34 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.34 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=38.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.33 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.32 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.32 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=38.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=960 avail_mem=38.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=38.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=832 avail_mem=38.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=832 avail_mem=38.28 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=768 avail_mem=38.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=704 avail_mem=38.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=640 avail_mem=38.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=576 avail_mem=38.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=512 avail_mem=38.25 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=512 avail_mem=38.25 GB):  50%|█████     | 29/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=480 avail_mem=38.27 GB):  50%|█████     | 29/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=448 avail_mem=38.26 GB):  50%|█████     | 29/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=416 avail_mem=38.26 GB):  50%|█████     | 29/58 [00:00<00:00, 42.07it/s]

    Capturing num tokens (num_tokens=384 avail_mem=38.26 GB):  50%|█████     | 29/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=352 avail_mem=38.25 GB):  50%|█████     | 29/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=352 avail_mem=38.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.65it/s]Capturing num tokens (num_tokens=320 avail_mem=38.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.65it/s]Capturing num tokens (num_tokens=288 avail_mem=38.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.65it/s]Capturing num tokens (num_tokens=256 avail_mem=38.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.65it/s]Capturing num tokens (num_tokens=240 avail_mem=38.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.65it/s]Capturing num tokens (num_tokens=224 avail_mem=38.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.65it/s]Capturing num tokens (num_tokens=224 avail_mem=38.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=208 avail_mem=38.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=192 avail_mem=38.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=176 avail_mem=38.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]

    Capturing num tokens (num_tokens=160 avail_mem=38.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=144 avail_mem=38.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=144 avail_mem=38.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=128 avail_mem=38.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=112 avail_mem=38.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=96 avail_mem=38.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.69it/s] Capturing num tokens (num_tokens=80 avail_mem=38.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=64 avail_mem=38.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=64 avail_mem=38.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=48 avail_mem=38.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=32 avail_mem=38.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=28 avail_mem=38.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.97it/s]

    Capturing num tokens (num_tokens=24 avail_mem=38.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=20 avail_mem=38.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=20 avail_mem=38.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=16 avail_mem=38.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=12 avail_mem=38.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=8 avail_mem=38.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.52it/s] Capturing num tokens (num_tokens=4 avail_mem=38.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=4 avail_mem=38.18 GB): 100%|██████████| 58/58 [00:01<00:00, 40.94it/s]


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
    Generated text:  Ken and I am a freelance translator and writer.
    As a translator, I am qualified to produce accurate and clear written translations of content from multiple languages to any of the 24 official languages of the United Nations. My expertise includes languages such as English, Spanish, Portuguese, French, Italian, Chinese, and many others.
    As a writer, my primary focus is to create captivating and engaging content that resonates with readers. I enjoy crafting stories, novels, and articles that educate and inspire. I also enjoy collaborating with other writers and producers to bring my vision to life through compelling writing and storytelling. As a freelance writer, I aim to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking official of the government, who has the authority to issue laws, command armies, and handle foreign affairs. A president is elected by the people to serve a term of 4 years. If the current president is Thomas Jefferson, and the term of office for the next president is 2 years, how many terms of office will the next president have to serve?
    To determine how many terms of office the next president will have to serve, we need to follow these steps:
    
    1. Identify the current term of office of the president.
    2. Determine the length of the term of office for the next president.
    3. Calculate the
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A．Paris B．Rome C．London D．New York
    A. Paris
    B. Rome
    C. London
    D. New York
    答案: A
    
    设集合A由所有满足条件x^2+2x-3<0的实数x组成，集合B由所有满足条件x^2-5x+4≥0的实数x组成。求A与B的交集A∩B。给出的选项如下：A. {-1,3} B. {1,3} C. {-3,1,3} D. {-3,1
    ===============================
    Prompt: The future of AI is
    Generated text:  quite exciting and promising, with many researchers, developers, and companies working to develop AI that can adapt to and learn from different types of data, language, and context. However, as with any new technology, there are also concerns and challenges associated with AI. One of the biggest concerns is the potential for bias in AI systems, which can result in unfair treatment of certain groups of people or objects. This can lead to discriminatory practices and unfair outcomes, especially in fields such as healthcare, hiring, and criminal justice.
    One way to address this concern is to ensure that AI systems are developed and used in a way that minimizes bias. This


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has been a hub of culture, politics, and commerce for centuries. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies such as blockchain, IoT, and quantum computing, which could lead to new applications and opportunities.
    
    2. Greater emphasis on ethical considerations: As AI becomes more prevalent in various industries, there will be a greater emphasis on ethical considerations and responsible use of AI. This could lead to new regulations and standards to govern the development and use of AI.
    
    3. Increased use of AI in healthcare: AI is already
    


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
    Generated text:  [Name]. I am a creative artist with a strong passion for capturing the beauty of nature. My work is inspired by the unique stories and legends of the Native American people, and I strive to bring these stories to life in my art. I am a storyteller and poet, and I use my artistic skills to turn the world around. I am always looking for new and exciting projects to work on, and I am excited to bring new and unique perspectives to the world. Please feel free to ask me anything, and I will do my best to answer all of your questions. Together, we can make a difference in the world! [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is the most populous city in Europe, with a population of over 7 million residents. Paris is known for its historical architecture, vibrant culture, and cultural institutions, such as the Louvre Museum and the Musée d'Orsay. It is also famous for its fashion industry, and many famous designers and celebrities have their residences or offices in the city. Paris is a bustling and exciting city that attracts visitors from all over the world. It is often referred to as the "City of Love" due to the city's romantic atmosphere, and many cultural events and festivals are held throughout
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and diverse, with potential developments in several key areas:
    
    1. Advanced Neural Networks: The future of AI will likely see the development of even more complex and powerful neural networks. These networks will be able to learn from vast amounts of data, making them more capable of achieving human-level performance on certain tasks.
    
    2. Augmented Intelligence: The future of AI will likely see a significant increase in the integration of AI with the human brain. This could lead to more advanced forms of artificial intelligence that are able to perform tasks that are not currently possible with human-level intelligence.
    
    3. Ethics and Responsibility: There will be a growing emphasis on


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

    ]

     and

     I

    'm

     a

     [

    Name

     of

     fictional

     profession

    ].

     I

    've

     always

     been

     fascinated

     by

     the

     world

     of

     [

    brief

    ly

     describe

     your

     profession

    ],

     and

     I

    've

     been

     researching

     and

     learning

     about

     it

     for

     years

    .

     I

    've

     traveled

     to

     many

     places

     and

     gained

     a

     deep

     understanding

     of

     [

    name

     of

     profession

    's

     history

    ,

     culture

    ,

     and

     skills

    ].

     As

     a

     result

    ,

     I

    'm

     able

     to

     perform

     [

    describe

     your

     job

     role

     or

     expertise

    ].

     Whether

     you

    're

     looking

     for

     a

     job

    ,

     need

     advice

    ,

     or

     just

     want

     to

     learn

     more

     about

     this

     field

    ,

     I

    'm

     here

     to

     help

    .

     Let

    's

     connect

    !

     #

    Professional

     #

    Job

    Seek

    er

     #

    Learn

    More

     #

    Inter

    ests

     #

    History

    
    
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

    ,

     which

     is

     a

     historical

     and

     cultural

     center

     and

     one

     of

     the

     most

     popular

     tourist

     destinations

     in

     the

     country

    .

     It

     is

     located

     on

     the

     Se

    ine

     river

     in

     the

     Î

    le

     de

     France

     region

     and

     is

     home

     to

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

     numerous

     other

     iconic

     landmarks

    .

     Paris

     is

     known

     for

     its

     vibrant

     culture

    ,

     rich

     history

    ,

     and

     unique

     architecture

    ,

     and

     is

     a

     must

    -

    visit

     destination

     for

     tourists

     and

     locals

     alike

    .

     It

     is

     also

     home

     to

     many

     notable

     French

     arts

     and

     cultural

     institutions

    ,

     including

     the

     Mus

    ée

     Rod

    in

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     French

     capital

     is

     a

     thriving

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     range

     of

     technological

     advancements

     and

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

     Increased

     cognitive

     abilities

    :

     AI

     systems

     are

     expected

     to

     become

     more

     capable

     of

     understanding

     and

     processing

     complex

     human

     language

    ,

     recognizing

     patterns

     and

     relationships

    ,

     and

     generating

     human

    -like

     creativity

     and

     reasoning

    .
    


    2

    .

     Integration

     with

     other

     fields

    :

     AI

     is

     expected

     to

     become

     more

     integrated

     with

     other

     fields

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    ,

     leading

     to

     new

     applications

     and

     solutions

    .
    


    3

    .

     Improved

     safety

     and

     ethics

    :

     As

     AI

     systems

     become

     more

     prevalent

     in

     our

     daily

     lives

    ,

     there

     will

     be

     a

     growing

     concern

     about

     the

     safety

     and

     ethical

     implications

     of

     AI

    ,

     including

     issues

     related

     to

     bias

    ,

    



```python
llm.shutdown()
```
