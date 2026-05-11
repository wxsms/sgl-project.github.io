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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]


    2026-05-11 00:51:04,657 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 00:51:04] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.25it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.81it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.81it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.81it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.81it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.81it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.81it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.81it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.81it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.31it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 22.78it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]

    Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.60it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.36 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.60it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.09 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  21%|██        | 12/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.34 GB):  21%|██        | 12/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.34 GB):  21%|██        | 12/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.33 GB):  21%|██        | 12/58 [00:00<00:02, 21.65it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.33 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.32 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.14 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.30 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.30 GB):  31%|███       | 18/58 [00:00<00:01, 22.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.30 GB):  31%|███       | 18/58 [00:00<00:01, 22.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.30 GB):  31%|███       | 18/58 [00:00<00:01, 22.88it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.27 GB):  31%|███       | 18/58 [00:00<00:01, 22.88it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.27 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.42it/s]Capturing num tokens (num_tokens=960 avail_mem=74.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.42it/s] Capturing num tokens (num_tokens=896 avail_mem=74.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.42it/s]Capturing num tokens (num_tokens=832 avail_mem=74.27 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.42it/s]Capturing num tokens (num_tokens=768 avail_mem=74.26 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.42it/s]Capturing num tokens (num_tokens=768 avail_mem=74.26 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.61it/s]Capturing num tokens (num_tokens=704 avail_mem=74.26 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.61it/s]Capturing num tokens (num_tokens=640 avail_mem=74.25 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.61it/s]Capturing num tokens (num_tokens=576 avail_mem=74.25 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.61it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.23 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.61it/s]Capturing num tokens (num_tokens=512 avail_mem=74.23 GB):  50%|█████     | 29/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=480 avail_mem=74.24 GB):  50%|█████     | 29/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=448 avail_mem=74.22 GB):  50%|█████     | 29/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=416 avail_mem=74.23 GB):  50%|█████     | 29/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=384 avail_mem=74.23 GB):  50%|█████     | 29/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=352 avail_mem=74.22 GB):  50%|█████     | 29/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=352 avail_mem=74.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.74it/s]Capturing num tokens (num_tokens=320 avail_mem=74.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.74it/s]Capturing num tokens (num_tokens=288 avail_mem=74.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.74it/s]Capturing num tokens (num_tokens=256 avail_mem=74.20 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.74it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.20 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.74it/s]Capturing num tokens (num_tokens=224 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.74it/s]Capturing num tokens (num_tokens=224 avail_mem=74.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.66it/s]Capturing num tokens (num_tokens=208 avail_mem=74.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.66it/s]Capturing num tokens (num_tokens=192 avail_mem=74.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.66it/s]Capturing num tokens (num_tokens=176 avail_mem=74.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.66it/s]Capturing num tokens (num_tokens=160 avail_mem=74.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.66it/s]Capturing num tokens (num_tokens=144 avail_mem=74.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.66it/s]Capturing num tokens (num_tokens=144 avail_mem=74.17 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=128 avail_mem=74.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=112 avail_mem=74.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.77it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.15 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.77it/s] Capturing num tokens (num_tokens=80 avail_mem=74.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=48 avail_mem=74.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=32 avail_mem=74.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=28 avail_mem=74.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=24 avail_mem=74.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=16 avail_mem=74.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.17it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.09 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=8 avail_mem=74.09 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.17it/s] Capturing num tokens (num_tokens=4 avail_mem=74.08 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=4 avail_mem=74.08 GB): 100%|██████████| 58/58 [00:01<00:00, 31.50it/s]


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
    Generated text:  Mary and I am a passionate art student. I enjoy creating and painting various types of art. I am looking for a job that can help me get some further education or practice. I would like to know what kind of job in art is required and how to find it. Is there any website that I can use to look for job in art?  I hope you can help me and provide me with some tips or directions.
    Looking for a job in art, so I could get further education and practice. What kind of job requires art skills and how can I find it?
    What kind of job requires art skills?
    Artistic work is
    ===============================
    Prompt: The president of the United States is
    Generated text:  a popular figure in the country. He is a candidate for the presidency, and the polls show that his popularity is increasing. The president has a political strategist who uses his expertise to generate popularity for him. If the president is using the political strategist's expertise to increase his popularity, how would the strategy be classified? 
    
    A. Ad Hominem  
    B. Appeal to Pity  
    C. Appeal to Emotion  
    D. Appeal to a Group
    
    To determine how the strategy of the political strategist for increasing a popular candidate for the presidency would be classified, let's analyze each option step by step:
    
    1. **Ad Hominem**:
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The town where it is located is a major city in the north of the country. The name of the town was originally Paris. When the city was founded in 911, it was called the "City of the First Crusade". It was then called "Pars" until 1067 when the "City of the Canons" was named. The name "Paris" came from the Latin word "Partha". The first Parisians were originally people from the city of Partha. The name "Paris" means "City of Golden White" in French. There are many famous Parisian places,
    ===============================
    Prompt: The future of AI is
    Generated text:  full of exciting opportunities for businesses and governments, but there are also some major risks. For businesses, it is critical to understand the risks of AI and to take steps to mitigate them. One of the biggest risks of AI is the potential for job displacement, as machines can perform tasks that are currently performed by humans. This can lead to a decrease in the number of jobs available for people in certain industries, which can have a significant impact on the economy. Governments must also be aware of the risks of AI and take steps to protect the privacy and security of their citizens. These risks can be mitigated by investing in AI research and development,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always eager to learn and grow. I'm a [reason for interest in the industry] and I'm always eager to learn and grow. I'm a [reason for interest in the industry] and I'm always eager to learn and grow. I'm a [reason
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French Quarter, which is a historic neighborhood known for its narrow streets, narrow alleys, and charming architecture. Paris is a cultural and political center that plays a significant role in French society and politics. It is also a popular tourist destination, known for its rich history, beautiful architecture, and delicious cuisine. The city is home to many museums, theaters, and other cultural institutions, and is a major hub for business and commerce in France. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: As AI continues to improve, it is likely to become more and more integrated into our daily lives. This could lead to a significant increase in automation, where machines will take on tasks that were previously done by humans. This could lead to job losses in certain industries, but also create new opportunities for those who are skilled in AI and machine learning.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be a greater
    


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
    Generated text:  [Name] and I'm a [career field] specialist, specializing in [specific skill or expertise]. In my [number of years] of experience, I have [number of projects] successful projects that have made a positive impact on [specific field]. I have a natural talent for problem-solving and a desire to help others achieve their goals. I'm committed to providing exceptional customer service, and I'm always looking for ways to improve my skills and knowledge. I'm excited to work with you and help you achieve your goals. How can I contact you? [Name] [Email Address] [Phone Number] [LinkedIn Profile] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement is accurate, and it directly presents the fundamental information about the capital city of France. As the largest and most populous city in France, Paris serves as the political, cultural, and economic capital of the country. Its status as the world's third-largest city by population and its influence on the French national identity make it an essential part of the French psyche, influencing both the French people and the French government. This city is also known as the "City of Love," with its famous Notre-Dame Cathedral and landmarks like the Eiffel Tower. Its dual identity as a modern metropolis and a cultural center has shaped
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a continued exponential growth in capabilities and sophistication. Here are some potential future trends in AI:
    
    1. Autonomous vehicles: Autonomous cars are expected to become a major part of daily life, with widespread adoption expected in the next 10-15 years. They will be able to navigate roads, drive safely, and communicate with other vehicles, as well as with pedestrians and other objects in their path.
    
    2. Smart homes: Smart homes are expected to become more common, with home automation technologies like voice assistants, smart thermostats, and security systems becoming more advanced. They will allow people to control their homes with


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

    ...

     
    


    Hello

    ,

     my

     name

     is

     [

    Name

    ].

     What

     kind

     of

     character

     do

     you

     portray

     in

     your

     work

    ?

     As

     a

     writer

    ,

     I

     am

     a

     storyt

    eller

    .

     I

     use

     language

     as

     a

     tool

     to

     evoke

     emotions

    ,

     ideas

    ,

     and

     experiences

     in

     the

     reader

    's

     mind

    .

     I

     believe

     that

     stories

     have

     the

     power

     to

     shape

     our

     world

     and

     to

     make

     us

     who

     we

     are

    .

     I

    'm

     a

     master

     of

     dialogue

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     tell

     stories

    .

     
    


    What

     is

     one

     of

     the

     most

     important

     elements

     in

     your

     work

    ?

     In

     my

     work

    ,

     one

     of

     the

     most

     important

     elements

     is

     the

     use

     of

     language

    .

     I

     believe

     that

     language

     is

     a

     powerful

     tool

     that

     can

     shape

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     bustling

     met

    ropolis

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     fashion

    .

     It

     is

     the

     cultural

     and

     economic

     center

     of

     the

     nation

     and

     has

     been

     the

     capital

     for

     over

     four

     centuries

    .

     Paris

     is

     renowned

     for

     its

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     It

     also

     hosts

     many

     world

    -f

    amous

     festivals

     and

     events

    ,

     including

     the

     World

     Cup

     soccer

     tournament

     and

     the

     New

     Year

    's

     Eve

     celebration

    .

     Despite

     its

     reputation

     as

     a

     hub

     of

     culture

     and

     politics

    ,

     Paris

     remains

     a

     diverse

     and

     vibrant

     city

    .

     However

    ,

     with

     the

     rise

     of

     digital

     technologies

     and

     tourism

    ,

     it

     has

     seen

     a

     decline

     in

     its

     traditional

     arts

     and

     culture

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     dynamic

     and

     constantly

     evolving

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     field

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     fairness

    :

     With

     concerns

     about

     data

     privacy

    ,

     bias

    ,

     and

     ethical

     implications

     growing

    ,

     there

     is

     a

     growing

     emphasis

     on

     improving

     the

     fairness

     and

     ethics

     of

     AI

     systems

    .

     This

     could

     lead

     to

     new

     algorithms

     and

     approaches

     that

     are

     more

     transparent

     and

     accountable

    .
    


    2

    .

     Greater

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     is

     a

     growing

     need

     for

     more

     seamless

     integration

     with

     other

     technologies

    .

     This

     could

     include

     more

     integration

     with

     smart

     homes

    ,

     wearable

     devices

    ,

     and

     other

     emerging

     technologies

    .
    


    3

    .

     Increased

     use

    



```python
llm.shutdown()
```
