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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.75it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.75it/s]


    2026-05-09 15:58:55,003 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 15:58:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  9.34it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 15.54it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 23.80it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 33.17it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 33.17it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 33.17it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 33.17it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 33.17it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 33.17it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 33.17it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 33.17it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 33.17it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 33.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.57it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.57it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.57it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.57it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.12it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.12it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.12it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.12it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.12it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.12it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 44.49it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 44.49it/s]Capturing num tokens (num_tokens=448 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 44.49it/s]Capturing num tokens (num_tokens=416 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 44.49it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 44.49it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 44.49it/s]Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=256 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=208 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=192 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.61it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.90 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=144 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.57it/s] Capturing num tokens (num_tokens=80 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.69it/s]Capturing num tokens (num_tokens=48 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.69it/s]Capturing num tokens (num_tokens=32 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.69it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.69it/s]Capturing num tokens (num_tokens=24 avail_mem=70.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.69it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.69it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.25it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.25it/s]Capturing num tokens (num_tokens=12 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.25it/s]Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.25it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.25it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 42.58it/s]


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
    Generated text:  Priscilla. I'm a 5th grade student at a preschool in East Windsor, and I have a lot of questions about eating. Can you tell me about healthy eating? Absolutely, Priscilla! Eating well is all about making smart choices to keep you healthy and strong. Here are some key points to help you understand healthy eating:
    
    1. **Eat More of the Right Foods:**
       - Focus on whole foods that have all the nutrients your body needs. This includes fruits, vegetables, whole grains, lean proteins, and healthy fats like avocados, nuts, and seeds.
       - Avoid foods high in sugar
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what issue to address first in his inaugural address. The first 1000 tweets on Twitter were received from 30 different users. If every tweet mentions at least one of the top 100 hashtags, and each hashtag is associated with a different user, what is the maximum number of tweets that could have mentioned all 100 hashtags? To solve this problem, we need to understand the constraints and implications of the problem. Here's a step-by-step breakdown:
    
    1. **Identify the Problem**: We need to find the maximum number of tweets that could have mentioned all 100 hashtags
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which country?
    A. Germany
    B. Switzerland
    C. United Kingdom
    D. Belgium
    
    To determine which country the capital of France is located in, let's analyze the information step by step:
    
    1. Identify the capital of France: The capital of France is Paris.
    2. Consider the options provided: Germany, Switzerland, United Kingdom, and Belgium.
    
    None of these countries have Paris as their capital.
    
    Given the information above, the correct answer is:
    
    \boxed{D}
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and there’s no question about it. But have you ever wondered what the future might hold? Who knows, if that’s what you are going to ask me one day. Today, we are going to go back in time to the year 2043 and discuss the future of AI. We’ll see what the future holds for the future of AI. But let’s not forget to keep a few things in mind.
    1. AI will be around everywhere. From the smart homes to the smart cities, the impact of AI will be far-reaching. The future of AI is bright and it’s here to stay.
    2.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many famous museums and historical sites. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. It is a popular tourist destination and a major economic and cultural hub in Europe. The city is also known for its cuisine, including its famous Parisian dishes such as croissants and escargot. Paris is a city of contrasts, with its modern architecture and high-tech industries
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI will likely continue to be used for tasks such as fraud detection, cybersecurity, and environmental monitoring, as well as for tasks such as language translation and image recognition. As AI becomes more integrated into our daily lives, we may see a shift towards more personalized and context-aware AI that can adapt to our needs and preferences. However, there are also potential risks and challenges associated with the use of AI, including
    


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
    Generated text:  [insert name] and I'm a [insert occupation/role]. I love [insert something related to your occupation or character]. I enjoy spending my time [insert something related to your occupation or character], whether it's reading books, watching TV shows, or playing video games. I'm very passionate about [insert something related to your occupation or character], and I believe that my passions should guide my every move. I'm always looking for ways to improve my skills and knowledge, and I try to learn something new every day. I'm a [insert your occupation or character's title] who is always eager to learn and grow. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city in Europe. 
    
    (A) Paris is a medieval city with a history dating back to the 9th century. 
    (B) Paris is a modern city located on the western bank of the Seine River. 
    (C) Paris is a medieval city with a history dating back to the 9th century and is located on the western bank of the Seine River. 
    (D) Paris is a modern city located on the western bank of the Seine River and is a historical city with a history dating back to the 9th century.
    
    (D) Paris is a modern city located on the western bank of the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a wide range of trends, including advancements in computing power, new algorithms, and machine learning techniques. Some possible future trends in AI include:
    
    1. Increased automation and robotics: As AI continues to advance, we may see more automation and robotics that can perform tasks with precision and efficiency. This could lead to the creation of new industries and jobs, as well as a shift towards more distributed and collaborative work.
    
    2. Increased dependence on AI: As AI becomes more advanced and ubiquitous, it is likely to become a more significant part of our daily lives. This could lead to greater reliance on AI-driven technologies in areas such


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

     am

     [

    Your

     Age

    ]

     years

     old

    .

     I

     am

     from

     [

    Your

     Location

    ]

     and

     my

     occupation

     is

     [

    Your

     Profession

    ].

     Throughout

     my

     life

    ,

     I

     have

     always

     had

     a

     strong

     passion

     for

     [

    Your

     Passion

    ].

     I

     believe

     that

     everyone

     should

     be

     given

     the

     opportunity

     to

     live

     their

     dreams

     and

     achieve

     their

     goals

    .

     I

     have

     been

     inspired

     by

     [

    Your

     Inspiration

    ],

     who

     has

     inspired

     me

     to

     pursue

     my

     passion

    .

     What

    's

     your

     most

     memorable

     experience

    ,

     and

     how

     did

     it

     shape

     you

     as

     a

     person

    ?
    


    I

     hope

     you

     enjoy

     your

     visit

     to

     my

     website

    .

     Let

     me

     know

     if

     you

     have

     any

     questions

     or

     if

     there

    's

     anything

     I

     can

     do

     to

     assist

     you

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

    ,

     capital

     of

     the

     Î

    le

     de

     la

     C

    ité

    ,

     an

     island

     in

     the

     Se

    ine

     River

    .

     Paris

     is

     known

     for

     its

     iconic

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

     other

     landmarks

    ,

     as

     well

     as

     its

     rich

     history

    ,

     culture

    ,

     and

     cuisine

    .

     Its

     economy

     is

     heavily

     dependent

     on

     tourism

     and

     its

     influence

     extends

     into

     its

     surrounding

     areas

    .

     The

     city

     is

     often

     referred

     to

     as

     the

     “

    city

     of

     love

    ”

     due

     to

     its

     Paris

    ian

     charm

     and

     romantic

     atmosphere

    .

     As

     of

     

    2

    0

    2

    1

    ,

     the

     population

     of

     Paris

     is

     approximately

     

    2

    .

    2

     million

    .

     Paris

     is

     a

     global

     cultural

     and

     economic

     hub

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     difficult

     to

     predict

    ,

     but

     here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     development

     of

     this

     technology

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     machine

     learning

     capabilities

    :

     As

     AI

     technology

     continues

     to

     evolve

    ,

     we

     may

     see

     greater

     advancements

     in

     machine

     learning

    ,

     which

     can

     make

     AI

     more

     accurate

    ,

     efficient

    ,

     and

     personalized

    .

     This

     could

     lead

     to

     more

     personalized

     and

     user

    -friendly

     AI

     experiences

    ,

     as

     well

     as

     increased

     accuracy

     in

     certain

     tasks

    .
    


    2

    .

     Integration

     with

     more

     diverse

     and

     complex

     systems

    :

     AI

     is

     currently

     limited

     to

     specific

     applications

    ,

     but

     as

     we

     continue

     to

     develop

     more

     complex

     systems

     and

     applications

    ,

     we

     may

     see

     more

     integration

     between

     AI

     and

     other

     systems

    ,

     such

     as

     medical

     imaging

    ,

     financial

    



```python
llm.shutdown()
```
