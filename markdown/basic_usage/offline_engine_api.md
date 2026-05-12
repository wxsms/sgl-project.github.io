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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]


    2026-05-12 20:41:30,584 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 20:41:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.94it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.95it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.47it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.14it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.46it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.46it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.46it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.46it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.46it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.46it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.46it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.57 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.57 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.06it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.05it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.05it/s] Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.05it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.05it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=640 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=448 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=352 avail_mem=72.50 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=352 avail_mem=72.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.93it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.93it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.93it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.93it/s]Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.93it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.93it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=208 avail_mem=72.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=192 avail_mem=72.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=144 avail_mem=72.47 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=144 avail_mem=72.47 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.74it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=48 avail_mem=72.45 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.50it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.45 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=24 avail_mem=72.44 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=12 avail_mem=72.43 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.25it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 40.66it/s]


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
    Generated text:  Tameka and I'm a sophomore at The University of Texas at Austin. I'm currently a senior in French and enjoy visiting museums and visiting museums. I'm interested in French and I'd like to work on my French and I'd like to get a job when I grow up. What can you tell me about your job search process? What are some tips for getting a job as a French teacher? As a French teacher, what skills and qualifications would be most important? What advice would you give to someone who wants to become a French teacher?
    As an AI language model, I do not have personal experience or emotions. However,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. His or her job is to make sure that the government works well. Sometimes, the president has to make important decisions that may change how the country behaves. So, it's very important that the president is well educated. After all, the president is the head of the government, which can make decisions about the country. As a result, we expect that they have enough knowledge and experience to be able to make good decisions. Even if the president is not very experienced, they should be able to make good decisions, especially if they know what is going on in their country. When the president is not well educated,
    ===============================
    Prompt: The capital of France is
    Generated text:  called what?
    The capital of France is called Paris. Paris is the capital city of France and is located in the northwestern region of the country, near the Mediterranean Sea. It is known for its stunning architecture, rich history, and annual music festival, the Opéra. Paris is also famous for its fashion industry, fashion week, and the annual Eiffel Tower parade. The city is known as a melting pot of cultures and is home to many famous landmarks and attractions, including the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. Paris is often referred to as the "City of Light" due
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    AI is rapidly evolving, and its impact on society and the world we live in is shifting rapidly. Here's what we expect to see as the next decade goes by.
    
    By now, a lot of the discussion around AI has centered around its potential for commercial applications. The field is growing, and this is not necessarily good news. It's also not necessarily bad news, however. Much of the work that goes into developing AI will ultimately be about making the system more generalizable. This means that it will become more and more difficult to figure out exactly how to use the system, and it may be more difficult to figure out how


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, positive description of your favorite activity]. I'm always looking for ways to challenge myself and expand my horizons. What's your favorite book or movie? I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Quarter. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. It is also known for its cuisine, including French cuisine, and its fashion industry. The city is home to many international organizations and events, including the World Cup and the Olympics. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that is both beautiful and exciting,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread adoption in healthcare.
    
    2. AI in finance: AI is already being used in finance to improve risk management, fraud detection, and trading algorithms. As AI technology continues to evolve, we can expect to see even more widespread adoption in finance.
    
    3. AI in manufacturing: AI is already being used in manufacturing to improve efficiency, reduce costs, and increase productivity
    


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
    Generated text:  [Name], and I am a [profession] [career]. I have always been passionate about [career goal or hobby]. I enjoy [relationship or social activity] and I am always looking for new experiences and challenges to learn and grow. I am [age], and I strive to be [character trait or quality]. I am [description of self]. 
    
    I am very open-minded, creative, and driven to pursue my goals and take risks. I am a [character trait] person and I am always ready to learn and grow. I am [description of self]. I am [age], and I am [description of self].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is known for its historic landmarks, beautiful architecture, and vibrant culture, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is the largest city in the European Union and is home to the French Parliament, the President of the Republic, and other important government officials. Paris is also the birthplace of numerous famous French artists and authors, including Victor Hugo, who wrote "Les Miserables." The city is home to over 20 million people and is a major center for education, business, and entertainment in the world. Paris is often referred to as the "Parisian
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a combination of advances in computational power, data collection, and training, as well as a growing focus on ethical considerations and privacy concerns. Here are some potential trends that may emerge in the AI landscape in the coming years:
    
    1. Increased reliance on AI for routine tasks: As AI becomes more capable, it may become a more prevalent tool for automating routine tasks, such as customer service, data analysis, and routine maintenance. This could result in significant cost savings for businesses and an increased ability to provide valuable support to customers.
    
    2. Greater emphasis on AI in healthcare: AI is already making an impact in healthcare,


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

     am

     a

     computer

     scientist

    .

     I

     graduated

     from

     [

    University

    ]

     and

     worked

     as

     a

     [

    field

     of

     science

    ]

     researcher

     for

     [

    company

     name

    ]

     for

     [

    number

     of

     years

    ].

     I

     have

     been

     known

     to

     stay

     up

     late

     at

     night

     thinking

     about

     [

    idea

     or

     project

    ].

     I

     enjoy

     [

    what

     I

     enjoy

     doing

    ].

     I

     am

     always

     looking

     for

     ways

     to

     improve

     [

    idea

     or

     project

    ].

     I

     believe

     in

     using

     my

     creativity

     and

     knowledge

     to

     make

     a

     difference

     in

     the

     world

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

    .

     I

     am

     passionate

     about

     [

    field

     of

     science

    ],

     and

     I

     am

     always

     eager

     to

     learn

     more

     about

     it

    .

     I

     believe

     that

     the

     world

     needs

     more

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Ex

    plain

     why

     you

     believe

     this

     to

     be

     the

     correct

     answer

    .

     To

     answer

     this

     question

    ,

     consider

     the

     following

     points

    :


    -

     The

     question

     asks

     for

     a

     specific

     city

    's

     name

    .


    -

     The

     answer

     provided

     is

     "

    Paris

    ".


    -

     The

     question

     specifies

     that

     the

     capital

     of

     France

     is

     Paris

    .


    -

     The

     answer

     is

     a

     factual

     statement

    ,

     not

     a

     subjective

     interpretation

    .


    -

     "

    Paris

    "

     is

     the

     official

     name

     of

     the

     capital

     city

     of

     France

    .

     
    


    Therefore

    ,

     based

     on

     the

     criteria

     mentioned

     in

     the

     question

    ,

     I

     believe

     "

    Paris

    "

     to

     be

     the

     correct

     answer

     as

     it

     is

     a

     proper

     noun

     used

     to

     refer

     to

     the

     capital

     city

     of

     France

    .

     The

     other

     option

    ,

     "

    M

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     varied

    .

     Here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Improved

     Efficiency

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

     efficiency

     will

     continue

     to

     be

     a

     key

     area

     of

     focus

    .

     This

     could

     include

     tasks

     such

     as

     automation

    ,

     stream

    lining

     workflows

    ,

     and

     improving

     productivity

    .
    


    2

    .

     Increased

     Personal

    ization

    :

     AI

     will

     continue

     to

     improve

     as

     it

     learns

     from

     user

     data

     to

     provide

     more

     personalized

     experiences

    .

     This

     could

     involve

     creating

     models

     that

     tailor

     advertisements

     and

     recommendations

     to

     individual

     users

    .
    


    3

    .

     Autonomous

     Systems

    :

     Autonomous

     systems

     will

     become

     increasingly

     common

     in

     industries

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     manufacturing

    .

     These

     systems

     will

     be

     able

    



```python
llm.shutdown()
```
