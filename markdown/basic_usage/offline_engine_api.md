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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.72it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.71it/s]


    2026-05-12 22:56:07,199 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 22:56:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.45it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 31.99it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 31.99it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 31.99it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.99it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.07it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.07it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.07it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.07it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.77it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.11it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.38it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.65it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.57it/s]


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
    Generated text:  Xiaoxue. I am 10 years old. I speak fluent English. But I like to eat and play in my rooms, so I often forget to eat in the morning. One day, I accidentally dropped my glass. I felt very sad and I couldn't eat anything. I wanted to hide it but my parents didn't want me to. So I went to a doctor. The doctor told me it was nothing serious. He said I can't go to school but I can go to a special school for kids with special needs. I will go there because it is nice. I feel much better. 
    
    What can you
    ===============================
    Prompt: The president of the United States is
    Generated text:  a type of government official. Before the 20th century, the United States had only one president, and they were called the President of the United States. That president was President of the United States. The term of office of the President of the United States was twenty four years. The United States had a written constitution, and this constitution was called the U. S. Constitution. The United States was not a federal country, but a confederation of thirteen states, each of which had its own government. After the Civil War, the United States was called a "Republic". The Republican Party won the presidency of the United States in
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. True
    B. False
    Answer: A
    
    In financial products, the risk that can be transferred through derivatives and other derivative instruments is called ____.
    A. Credit Risk
    B. Liquidity Risk
    C. Reputation Risk
    D. Market Risk
    Answer: D
    
    The risk that can be transferred through derivatives and other derivative instruments is called ____.
    A. Credit risk
    B. Liquidity risk
    C. Reputation risk
    D. Market risk
    Answer: D
    
    In a securities market, when the number of securities in circulation reaches a certain size, it is referred to as a ____.
    A. Market
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. An AI expert can make it happen!
    But building and deploying an AI system is not a one-man show. It will require a diverse team of experts with different skills, areas of expertise and roles. In order to do so, a team composition should be identified and designed by the client. An AI development team, for example, must consist of a team of data scientists, computer scientists, engineers, and business people. The role of the AI expert is very important and should be regarded as a critical part of the team. It is the AI expert who will be responsible for the overall system design, development, implementation, deployment and


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working at [company name] for [number of years] years. I am a [job title] with a passion for [job title] and I am always looking for ways to [job title] and improve my skills. I am a [job title] and I am always looking for ways to [job title] and improve my skills. I am a [job title] and I am always looking for ways to [job title] and improve my skills. I am a [job title] and I am always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its cuisine, with many famous French dishes and restaurants. The city is a major economic and cultural center in Europe, and is home to many important institutions and organizations. Paris is a vibrant and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This will likely lead to increased regulation and oversight of AI systems, as well as new
    


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
    Generated text:  [Name], I'm an AI language model. I have access to vast amounts of data and can provide useful and informative responses to a wide range of questions. Can you tell me a little bit about yourself? [Insert a brief introduction that includes a name, title, and a little bit about what you can do. This should give the reader a sense of who you are and what you're good at. This should also be neutral and straightforward.]
    Hello, my name is [Name], I'm an AI language model. I have access to vast amounts of data and can provide useful and informative responses to a wide range of questions.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, and it is known as the "City of Love" for its iconic landmarks, such as the Eiffel Tower and Notre-Dame Cathedral, as well as its diverse cuisine and cultural scene. The city is also famous for its annual Mardi Gras celebration, which draws millions of visitors each year. France's capital is a bustling metropolis that offers a wide range of attractions, entertainment, and dining options for visitors. However, Paris is also known for its strict dress code, so it is important to dress appropriately when visiting the city. Overall, Paris is a vibrant and interesting city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting, and there are many potential areas where it may develop and evolve. Here are some possible trends that are likely to shape the future of AI:
    
    1. Improved precision and accuracy: With ongoing advancements in machine learning and artificial intelligence, it's possible that we'll see even more accurate and precise predictions of future events. This could lead to better decision-making and more effective use of resources in various fields such as healthcare, finance, and transportation.
    
    2. Increased automation and integration: As AI becomes more integrated into our daily lives, it's likely that we'll see increased automation and integration of AI systems with other systems. This could lead


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

     [

    Gender

    ]

     [

    Name

    ]

     [

    Relationship

     Status

    ]

     who

     is

     [

    Job

     Title

    ].

     I

     love

     [

    Favorite

     Hobby

    /

    Activity

    ]

     and

     I

    'm

     [

    Favorite

     Character

    /

    Person

    ].

     I

    'm

     a

     [

    Any

     other

     details

     you

    'd

     like

     to

     include

    ].

     I

    'm

     [

    Your

     Location

    ].

     I

     hope

     to

     make

     [

    Any

     goal

    /

    amb

    ition

     you

     have

    ].

     I

    've

     always

     [

    Any

     positive

     trait

     you

    'd

     like

     to

     highlight

    ].

     I

    'm

     constantly

     [

    Any

     positive

     trait

     you

    'd

     like

     to

     highlight

    ].

     I

    'm

     here

     to

     [

    Any

     positive

     trait

     you

    'd

     like

     to

     highlight

    ].

     Thank

     you

     for

     having

     me

    .

     Good

    night

    !

     To

     summarize

    :

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     beautiful

     pal

    aces

    ,

     and

     vibrant

     culture

    .

     Here

     are

     the

     key

     points

    :


    1

    .

     Paris

     is

     the

     capital

     of

     France

    .


    2

    .

     The

     city

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .


    3

    .

     Paris

     is

     a

     major

     cultural

     and

     economic

     center

    .


    4

    .

     It

     is

     home

     to

     many

     landmarks

     and

     museums

    .

     


    5

    .

     Paris

     is

     known

     for

     its

     jazz

    ,

     cinema

    ,

     and

     fashion

     scene

    .

     


    6

    .

     The

     city

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     surrounded

     by

     the

     French

     Alps

    .

     


    7

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     has

     a

     rich

     history

     dating

     back

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     and

     develop

    ,

     with

     many

     different

     trends

     shaping

     its

     direction

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     specialization

     and

     specialization

    :

     AI

     will

     continue

     to

     become

     more

     specialized

     and

     targeted

    ,

     with

     researchers

     and

     developers

     focusing

     on

     specific

     problems

     or

     applications

    .

     This

     will

     lead

     to

     a

     more

     niche

    -based

     AI

    ,

     with

     different

     AI

     systems

     and

     algorithms

     tailored

     to

     different

     industries

     and

     tasks

    .
    


    2

    .

     Aug

    mented

     and

     real

    -time

     learning

    :

     AI

     will

     continue

     to

     improve

     and

     evolve

     through

     real

    -time

     learning

     and

     augmentation

    .

     This

     will

     involve

     developing

     more

     advanced

     machine

     learning

     algorithms

     that

     can

     learn

     and

     adapt

     to

     new

     situations

    ,

     and

     will

     enable

     AI

     to

     operate

     in

     real

    -time

    ,

     making

    



```python
llm.shutdown()
```
