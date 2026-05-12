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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.42it/s]


    2026-05-12 00:59:43,346 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 00:59:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.59it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.49it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.78it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.71it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.71it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   9%|▊         | 5/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.84 GB):   9%|▊         | 5/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.83 GB):   9%|▊         | 5/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.83 GB):   9%|▊         | 5/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.83 GB):   9%|▊         | 5/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.83 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.70it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.81 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.81 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.81 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.81 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.80 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.80 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.80 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.79 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.79 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.79 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.77 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.74it/s]Capturing num tokens (num_tokens=960 avail_mem=71.79 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.74it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.78 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.74it/s]Capturing num tokens (num_tokens=832 avail_mem=71.78 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.74it/s]Capturing num tokens (num_tokens=832 avail_mem=71.78 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=768 avail_mem=71.78 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=704 avail_mem=71.77 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=640 avail_mem=71.77 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=576 avail_mem=71.77 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=512 avail_mem=71.75 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=512 avail_mem=71.75 GB):  50%|█████     | 29/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=480 avail_mem=71.77 GB):  50%|█████     | 29/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=448 avail_mem=71.77 GB):  50%|█████     | 29/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=416 avail_mem=71.77 GB):  50%|█████     | 29/58 [00:00<00:00, 41.65it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.76 GB):  50%|█████     | 29/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=352 avail_mem=71.76 GB):  50%|█████     | 29/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=352 avail_mem=71.76 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=320 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=288 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=256 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.02it/s]Capturing num tokens (num_tokens=240 avail_mem=71.74 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=224 avail_mem=71.74 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=224 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=208 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=192 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=176 avail_mem=71.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.31it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=144 avail_mem=71.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=144 avail_mem=71.73 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=128 avail_mem=71.73 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=112 avail_mem=71.72 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=96 avail_mem=71.72 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.33it/s] Capturing num tokens (num_tokens=80 avail_mem=71.72 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=64 avail_mem=71.71 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=64 avail_mem=71.71 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=48 avail_mem=71.71 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=32 avail_mem=71.71 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=28 avail_mem=71.70 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.41it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.70 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=20 avail_mem=71.69 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=20 avail_mem=71.69 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=16 avail_mem=71.69 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=12 avail_mem=71.69 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=8 avail_mem=71.68 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.67it/s] Capturing num tokens (num_tokens=4 avail_mem=71.68 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=4 avail_mem=71.68 GB): 100%|██████████| 58/58 [00:01<00:00, 39.90it/s]


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
    Generated text:  Nina and I am a 25 year old nurse. I started to have a back problem, and I had to go to the doctor. The doctor's opinion was that I had spondylolisthesis. I am very relieved because I had been suffering from my back problem for many years and was not getting any better.
    
    My question is, what is spondylolisthesis? I read that it is a condition where there is a weak spot in the middle part of your back and it causes you to have pain in your lower back, but I am not sure what the 'l' in spondylolisth
    ===============================
    Prompt: The president of the United States is
    Generated text:  from which country?
    A. China
    B. India
    C. France
    D. United Kingdom
    Answer:
    D
    
    Which of the following statements about the characteristics of artificial intelligence is incorrect?
    A. The development of artificial intelligence has been accelerating.
    B. Artificial intelligence is a human initiative.
    C. Artificial intelligence has the ability to mimic human mental activities.
    D. Artificial intelligence can only solve problems autonomously.
    Answer:
    D
    
    When a government agency forms a new administrative organization, what is the critical step that fundamentally determines the essence of the new organization?
    A. Determining the organizational purpose
    B. Formulating a comprehensive set of
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Lyon
    B. Paris
    C. Marseille
    D. Montpellier
    Answer:
    B
    
    The city of Paris is located on the ____.
    A. Ile-de-France
    B. Loire Valley
    C. Eifel Mountains
    D. Rhône Valley
    Answer:
    A
    
    Which of the following is the capital of France?
    A. Lyon
    B. Paris
    C. Marseille
    D. Montpellier
    Answer:
    B
    
    In a large, open parking lot, there are 4 large trucks, 3 small trucks, 2 medium trucks, and 1 small bus parked.
    ===============================
    Prompt: The future of AI is
    Generated text:  very different and exciting. Here are the top 10 predictions in this industry:
    
    1. AI will replace some jobs, and people who are working in these jobs will have a hard time finding work.
    
    2. The use of AI will become more widespread and cost-effective.
    
    3. AI will be used to make healthcare more efficient and personalized.
    
    4. AI will help to solve some of the world's biggest problems, such as climate change and global poverty.
    
    5. AI will continue to evolve and improve as technology advances.
    
    6. AI will be used in a variety of applications, including transportation, education, and entertainment.
    
    7. AI


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major center for art, culture, and politics in France. Paris is a popular tourist destination and a major economic hub. The city is home to many world-renowned museums, theaters, and other cultural institutions. It is also a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is a cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased automation and robotics: AI is already being used in manufacturing, transportation, and other industries to automate repetitive tasks and increase efficiency. As these technologies become more advanced, we can expect to see even more automation and robotics in the future.
    
    2. Enhanced human-computer interaction: AI is already being used to enhance human-computer interaction, such as through voice recognition and natural language processing. As these technologies continue to improve, we can expect to see even more advanced interactions between humans and machines.
    
    3. AI ethics and privacy concerns: As AI becomes more integrated into our daily lives
    


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
    Generated text:  [Name], and I'm a [Specialty/Job/Activity] specialist with a [Experience/Background] in the [Field/Industry]. I have [Number of years] years of experience in this field, and I'm always looking to learn new things and expand my knowledge. I'm always looking for opportunities to share my expertise with others, and I'm always open to new challenges and opportunities. I'm excited to meet you and discuss how my skills and experience can be of use to you. How can I help you today? [Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as "The City of Light" due to its grand architecture and vibrant culture. 
    
    A young man named Jacques was given a job at a retail store in Paris. The store manager, Mr. Lavalle, was surprised to see that Jacques had a special talent for drawing with charcoal. 
    
    Jacques was a perfect match for the store's creative needs. After training for a year, he could use his skills to sell the store's unique products and attract customers interested in the arts and design. 
    
    After working there for a few years, Jacques started a successful charcoal art studio in Paris. He drew beautiful, colorful
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a wide range of trends that will shape its development and applications. Here are some possible future trends in AI:
    
    1. Increased AI transparency and explainability: AI models will become more transparent and explainable, which will allow people to understand how AI makes decisions and make informed decisions. This will make AI more accessible and acceptable to people, who will be more likely to trust AI-powered systems.
    
    2. AI personalization: AI will become more personalized, with systems that can learn and adapt to individual user behavior and preferences. This will lead to more efficient and effective user experiences, as well as more accurate and personalized recommendations.
    
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

    ].

     I

     am

     a

     [

    职业

    /

    领域

    ]

     expert

     in

     the

     [

    领域

    ]

     and

     I

     have

     been

     passionate

     about

     [

    领域

    ]

     for

     [

    数量

    ]

     years

     now

    .

     I

     have

     always

     been

     [

    mot

    iv

    ations

    ]

     and

     [

    achie

    vements

    ].

     I

     have

     a

     passion

     for

     [

    领域

    ]

     because

     [

    reason

     for

     interest

    ].

     I

     am

     currently

     [

    status

    ]

     and

     I

     am

     eager

     to

     [

    desired

     outcome

    ].

     If

     you

     have

     any

     questions

    ,

     please

     feel

     free

     to

     ask

    .

     
    


    Your

     response

     should

     be

     in

     formal

     tone

    ,

     concise

    ,

     and

     respectful

    .

     Use

     appropriate

     language

     and

     avoid

     using

     personal

     pron

    ouns

     or

     overly

     emotional

     language

    .

     Include

     a

     brief

     quote

     or

     accomplishment

     that

     demonstrates

     your

     interest

     in

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    ,

     being

     the

     world

    ’s

     oldest

     capital

     city

    ,

     and

     a

     major

     financial

     and

     cultural

     center

    .

     Its

     famous

     landmarks

     include

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     Paris

     is

     known

     for

     its

     op

    ulent

     cuisine

    ,

     fashion

    ,

     and

     fashion

    -forward

     culture

    .

     It

    's

     also

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     landmarks

     like

     the

     Lou

    vre

     Museum

    ,

     E

    iff

    el

     Tower

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     Despite

     its

     fame

    ,

     Paris

     is

     a

     charming

     and

     beautiful

     city

     that

     offers

     a

     glimpse

     into

     France

    's

     rich

     history

     and

     culture

    .

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     developing

     and

     evolving

    ,

     with

     several

     possible

     trends

     shaping

     the

     technology

    's

     future

    .

     Some

     of

     the

     key

     trends

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     various

     sectors

    :

     As

     AI

     becomes

     more

     integrated

     into

     various

     sectors

    ,

     it

     could

     lead

     to

     a

     more

     efficient

     and

     cost

    -effective

     use

     of

     resources

    .

     For

     example

    ,

     AI

     could

     be

     used

     to

     analyze

     data

     and

     make

     predictions

    ,

     which

     could

     help

     businesses

     make

     better

     decisions

     and

     reduce

     waste

    .
    


    2

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

    :

     As

     AI

     continues

     to

     become

     more

     sophisticated

    ,

     it

     could

     become

     an

     increasingly

     important

     tool

     for

     making

     decisions

     in

     fields

     such

     as

     medicine

    ,

     law

    ,

     and

     finance

    .

     AI

     could

     be

     used

     to

     analyze

     large

     amounts

     of

    



```python
llm.shutdown()
```
