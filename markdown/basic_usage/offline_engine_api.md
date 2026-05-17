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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.55it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]


    2026-05-17 05:30:18,314 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 05:30:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.55it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.32it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.54it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.62it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.62it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.62it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.62it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.62it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.62it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.62it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.11 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]Capturing num tokens (num_tokens=960 avail_mem=71.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s] Capturing num tokens (num_tokens=896 avail_mem=71.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]Capturing num tokens (num_tokens=832 avail_mem=71.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=768 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=704 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=640 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=576 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=512 avail_mem=71.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=512 avail_mem=71.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=480 avail_mem=71.03 GB):  50%|█████     | 29/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=448 avail_mem=71.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=416 avail_mem=71.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=384 avail_mem=71.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.37it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=352 avail_mem=71.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=320 avail_mem=71.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=240 avail_mem=71.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=224 avail_mem=71.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=224 avail_mem=71.00 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.31it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.31it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.31it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.31it/s]Capturing num tokens (num_tokens=160 avail_mem=70.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.31it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.31it/s]Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=112 avail_mem=70.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=96 avail_mem=70.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s] Capturing num tokens (num_tokens=80 avail_mem=70.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=64 avail_mem=70.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=64 avail_mem=70.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=32 avail_mem=70.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=28 avail_mem=70.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=24 avail_mem=70.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=20 avail_mem=70.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.47it/s] Capturing num tokens (num_tokens=4 avail_mem=70.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=4 avail_mem=70.94 GB): 100%|██████████| 58/58 [00:01<00:00, 41.95it/s]


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
    Generated text:  31-year-old female and I'm in my mid-30's. I've never been pregnant and I'm pretty sure I didn't have an IVF cycle. I went to get my period on June 17th and I came back home on June 20th. I've been on a contraceptive pill since I got my period, I haven't done anything else that could cause pregnancy yet. I'm worried I might have a small amount of pregnancy but I'm not sure. Is this possible? If I did have a small amount, what's the best way to go? There's no such thing as
    ===============================
    Prompt: The president of the United States is
    Generated text:  a head of government. The president is the leader of the executive branch of the government. This branch is responsible for making major decisions on the day-to-day running of the country. These decisions include deciding whether to have a new president, or who the next vice president of the country will be. The president is also responsible for making decisions to keep the country safe. This includes deciding how to respond to a terrorist attack, how to stop criminal activities, and keeping the country safe. The president also has the power to nominate new officials to the positions of the armed forces, as well as other positions of power. The president is also responsible for
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Nice C. London D. Rome
    
    The capital of France is Paris.
    
    So, the answer is A. Paris. 
    
    To provide additional context:
    - Paris is the capital and largest city of France.
    - Nice is a city in France.
    - London is the capital and largest city of England.
    - Rome is the capital and largest city of Italy. 
    
    None of the other options (Nice, London, or Rome) are the capital of France.
    ===============================
    Prompt: The future of AI is
    Generated text:  growing and developing at an incredible pace. We have seen vast improvements in the way that we interact with technology, from chatbots to voice assistants, and even self-driving cars. But what about AI in the future? Will it replace humans, or will it create new roles and opportunities? It's a question that we need to consider and explore. This article will explore the future of AI, and how it will shape the future of work. We'll look at how AI will interact with humans, and how it will impact society and the economy. We'll also look at potential consequences of AI, including job loss, skill gaps, and increased


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for ways to [what I'm looking for in a job]. I'm a [type of person] and I enjoy [reason for being a [type of person]]. I'm [what I'm looking for in a job] and I'm always eager to learn and grow. I'm [what I'm looking for in a job] and I'm always looking for ways
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French Parliament House, which are important symbols of the country's political and cultural institutions. Paris is a bustling city with a rich history and a diverse population, and it is a popular tourist destination for many visitors. It is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a city that is both a cultural and historical center of France, and it continues to be a major economic and political center of the country.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased emphasis on ethical considerations, such as privacy, fairness, and accountability. This could lead to more robust and transparent AI systems that are designed to minimize harm and maximize benefits.
    
    3. Increased reliance on machine learning: Machine learning is likely to
    


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
    Generated text:  [Name], and I am a [occupation or profession]. As a [occupation/occupation], I've been in the field of [occupation] for [number of years] years. I specialize in [specialty or expertise]. And I love [reason for being passionate about the industry]. What kind of character do you think I am? What do you think I should do to achieve success in this field? Remember, I am not trying to be anything; I am just an easy answer to a question.
    Hey, everyone! My name is [Name]. I'm a [occupation] with [number of years] years of experience in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the largest city in Europe. Paris is known for its iconic Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It has a rich history dating back thousands of years and is a UNESCO World Heritage Site. Paris is also home to many world-renowned museums, fashion houses, and food scenes. The city is known for its art, culture, and gastronomy.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be dominated by four trends: automation, intelligence, decentralization, and customization. Automation will continue to play a key role in AI, driving more efficient and cost-effective processes. Intelligence is expected to continue to develop, with advances in machine learning, natural language processing, and computer vision driving more advanced AI solutions. Decentralization is expected to increase the number of AI systems, making them more accessible to users and reducing costs. Finally, customization is expected to play a key role in AI, with systems becoming increasingly personal and tailored to individual needs and preferences. These trends are expected to continue as AI technologies continue to evolve and become more


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

    name

    ].

     I

    'm

     a

     [

    brief

     biography

    ]

     who

     believes

     in

     [

    your

     belief

     or

     values

    ].

     My

     passion

     for

     [

    your

     interest

    ]

     has

     driven

     me

     to

     [

    why

     you

     do

     what

     you

     do

    ].

     Whether

     it

    's

     through

     my

     work

    ,

     hobbies

    ,

     or

     personal

     life

    ,

     I

     strive

     to

     [

    how

     I

     plan

     to

     contribute

     to

     [

    target

     area

     or

     community

    ]]

    .
    


    I

     hope

     you

     find

     this

     short

     introduction

     informative

     and

     engaging

    .

     What

     about

     you

    ?

     Are

     there

     any

     particular

     aspects

     of

     your

     background

     that

     you

    'd

     like

     to

     highlight

     in

     your

     introduction

    ?

     Let

     me

     know

    ,

     and

     I

    'll

     do

     my

     best

     to

     include

     them

    .

     [

    Name

    ]

     [

    Phone

     Number

    ]

     [

    Email

     Address

    ]

     [

    LinkedIn

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     largest

     city

     in

     Europe

     and

     the

     

    1

    2

    th

     most

     populous

     city

     in

     the

     world

    .

     The

     city

     is

     located

     on

     the

     North

     West

     coast

     of

     the

     French

     Alps

    ,

     and

     includes

     the

     historic

     centre

     of

     the

     city

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

    ,

     and

     is

     famous

     for

     its

     many

     landmarks

     and

     attractions

    ,

     including

     the

     Lou

    vre

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     home

     to

     many

     renowned

     museums

    ,

     such

     as

     the

     Mus

    ée

     du

     Lou

    vre

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Mus

    ée

     d

    '

    Art

     Moder

    ne

    ,

     and

     is

     a

     popular

     destination

     for

     visitors

     from

     around

     the

     world

    .

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     it

     is

     impossible

     to

     predict

     with

     certainty

     what

     specific

     trends

     will

     develop

    .

     However

    ,

     a

     few

     possible

     trends

     that

     are

     commonly

     expected

     to

     occur

     in

     the

     coming

     years

     include

    :
    


    1

    .

     Increased

     efficiency

     and

     productivity

     in

     manufacturing

    :

     With

     AI

    ,

     manufacturing

     plants

     can

     optimize

     production

     processes

    ,

     automate

     repetitive

     tasks

    ,

     and

     improve

     accuracy

     and

     efficiency

    .

     This

     could

     result

     in

     more

     productive

    ,

     less

     costly

    ,

     and

     more

     reliable

     products

    .
    


    2

    .

     Improved

     patient

     care

     and

     treatment

     outcomes

    :

     AI

     can

     be

     used

     to

     analyze

     patient

     data

    ,

     predict

     disease

     outcomes

    ,

     and

     provide

     personalized

     treatment

     recommendations

    .

     This

     could

     lead

     to

     better

     outcomes

     for

     patients

    ,

     reduced

     healthcare

     costs

    ,

     and

     improved

     health

     outcomes

    .
    


    3

    .

     Increased

     use

     of

    



```python
llm.shutdown()
```
