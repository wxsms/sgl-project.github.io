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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.93it/s]


    2026-05-07 21:28:41,453 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 21:28:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.68it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.45it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.17it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.19it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.19it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.19it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.19it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.19it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.19it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.19it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   7%|▋         | 4/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   7%|▋         | 4/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   7%|▋         | 4/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):  10%|█         | 6/58 [00:00<00:03, 16.24it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  10%|█         | 6/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  10%|█         | 6/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.08it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  21%|██        | 12/58 [00:00<00:02, 20.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:02, 20.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:02, 20.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:02, 20.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:02, 20.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.17it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.97it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.96it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.96it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.75it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.67it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.67it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.20it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.20it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.20it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.20it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.20it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.20it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.89it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 34.82it/s]


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
    Generated text:  Suyash Bhagwat and I am currently a rising senior in the class of 2016 at the University of Michigan. I am studying a minor in English and Political Science and a double major in Political Science and English. My favorite subject has to be Political Science and I have been deeply involved in the political process in my local community, particularly with the local government and our local political leadership. I have volunteered in my community as an AmeriCorps member, helping to organize the annual two-day Fair Vote Fair, which is a community-based political education and public education initiative. I have also been involved in the local Democratic
    ===============================
    Prompt: The president of the United States is
    Generated text:  22 years younger than his vice president. If the president's age is a prime number and the vice president's age is a composite number, what is the sum of their ages?
    Let the president's age be \( P \) and the vice president's age be \( V \). According to the problem, the president is 22 years younger than his vice president, so we have the equation:
    \[ P = V - 22. \]
    
    Additionally, the president's age is a prime number, and the vice president's age is a composite number. We need to find a pair of numbers that satisfy both conditions
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the capital city of the country of France.
    The capital of India is New Delhi. It is the capital city of the country of India. [The capital cities of other countries are also mentioned here]
    Is the above information correct?
    Available options:
     *yes;
     *no; To determine if the information provided is correct, I'll analyze the facts about the capital cities of France and India:
    
    1. Capital cities of France:
       - Paris (also known as the capital of France)
    
    2. Capital cities of India:
       - New Delhi (also known as the capital of India)
    
    The information given is:
    - The
    ===============================
    Prompt: The future of AI is
    Generated text:  now
    
    If you don’t want to be replaced by technology, you need to understand the future of AI
    
    by | Mar 31, 2018
    
    The future of AI is now
    
    The days of AI being done by humans are almost over. It’s a new frontier that is built on a new set of rules and challenges.
    
    Today, machines can do far more tasks than humans alone. From writing books to making decisions and decisions on things, machines are now the go-to for everything from making a meal to building a rocket.
    
    And while many are taking comfort in the fact that AI will never replace us, that


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also home to the French Riviera, a popular tourist destination for its beaches and warm climate. The city is known for its cuisine, including French cuisine, and is a popular destination for food lovers. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be greater emphasis on ethical and social considerations. This could lead to more responsible and accountable AI systems.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more integrated with human intelligence, it is likely
    


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
    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm an [occupation] by profession, [Name]. I've always been [interest, hobby, or skill] that makes me stand out, and I enjoy sharing my knowledge and expertise with others. I believe in [value, belief, or moral] and work hard to always improve myself, both professionally and personally. I'm a [attitude, personality, or temperament] individual, and I always strive to create a positive and fulfilling life for myself. I'm here to share my experiences and knowledge, and to help others reach their potential. Thank
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the “City of Light,” and is located in the south of the country. It is also the seat of the French government and is home to many of the country’s most famous landmarks, including Notre Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Paris is a vibrant and diverse city with a rich cultural heritage and is the largest metropolitan area in Europe. Its historic center is full of old-world charm and its modern suburbs are characterized by high-rise buildings and a modern, cosmopolitan atmosphere. In addition to its architectural wonders, Paris also has a vibrant arts scene, and the city is home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  predicted to be a rapidly evolving and complex field with numerous possibilities and advancements. Here are some possible future trends in artificial intelligence:
    
    1. More Personalization: With the rise of big data and machine learning, more personalized AI will be developed. AI will be able to learn and adapt to individual needs and preferences, leading to a more efficient and effective use of resources.
    
    2. Autonomous vehicles: The development of autonomous vehicles will become increasingly common, as AI will be able to help them navigate complex roads, recognize traffic signs, and avoid collisions.
    
    3. Improved healthcare: AI will be used to analyze medical data, identify patterns and trends, and


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

    ],

     and

     I

    'm

     a

     [

    A

     type

     of

     being

    ,

     such

     as

     a

     warrior

    ,

     a

     scientist

    ,

     a

     doctor

    ,

     etc

    .

    ].

     I

    'm

     a

     [

    A

     title

     or

     profession

    ].

     I

     believe

     in

     [

    A

     belief

    ,

     value

    ,

     or

     ideology

    ].

     I

     enjoy

     [

    A

     hobby

     or

     interest

     that

     you

     have

    ].

     I

     love

     [

    Something

     that

     makes

     me

     happy

    ].

     I

    'm

     passionate

     about

     [

    A

     passion

     or

     interest

     that

     you

     have

    ].

     I

     have

     always

     been

     inspired

     by

     [

    Someone

     or

     something

     that

     inspires

     me

    ].

     I

     am

     a

     [

    A

     distinguishing

     quality

    ,

     such

     as

     humorous

    ,

     brave

    ,

     hard

    working

    ,

     etc

    .

    ].

     I

     have

     always

     been

     grateful

     for

     [

    Something

     that

     gives

     me

     strength

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     and

     the

     largest

     metropolitan

     area

     in

     the

     European

     Union

    .

     Paris

     is

     located

     in

     the

     south

     of

     France

    ,

     on

     the

     northern

     bank

     of

     the

     Se

    ine

     river

    .

     It

     is

     one

     of

     the

     oldest

     cities

     in

     the

     world

    ,

     and

     is

     the

     birth

    place

     of

     many

     of

     France

    ’s

     most

     famous

     figures

    ,

     including

     the

     French

     Revolution

    's

     leader

    ,

     King

     Louis

     XVI

    ,

     and

     former

     Prime

     Minister

    ,

     Charles

     de

     Gaul

    le

    .

     The

     city

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

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

     many

     other

     famous

     landmarks

    .

     Paris

     is

     also

     a

     major

     center

     for

     education

    ,

     finance

    ,

     and

     tourism

    ,

     with

     many

     of

     France

    's

     leading

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     anticipated

     and

     is

     expected

     to

     revolution

    ize

     every

     aspect

     of

     our

     lives

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

     Aug

    mented

     and

     Virtual

     Reality

    :

     AR

     and

     VR

     will

     continue

     to

     grow

     in

     popularity

     as

     we

     continue

     to

     advance

     in

     AI

     technology

    .

     In

     AR

    ,

     AI

     will

     be

     used

     to

     create

     interactive

     and

     immersive

     experiences

     that

     are

     completely

     different

     from

     the

     physical

     world

    .

     In

     VR

    ,

     AI

     will

     enable

     us

     to

     interact

     with

     virtual

     environments

     in

     a

     more

     natural

     and

     realistic

     way

    .
    


    2

    .

     Improved

     Personal

    ization

    :

     AI

     will

     become

     even

     more

     advanced

     as

     we

     collect

     and

     analyze

     more

     data

    .

     This

     will

     enable

     us

     to

     provide

     more

     personalized

     recommendations

     to

     users

    ,

     such

     as

     recommendations

     based

     on

    



```python
llm.shutdown()
```
