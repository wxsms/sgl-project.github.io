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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]


    2026-05-18 21:54:48,705 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 21:54:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.67it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=20):  76%|███████▌  | 44/58 [00:05<00:00, 26.60it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 36.82it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 36.82it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 36.82it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 36.82it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 36.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.56 GB):   3%|▎         | 2/58 [00:00<00:03, 17.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.52 GB):   3%|▎         | 2/58 [00:00<00:03, 17.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.52 GB):   3%|▎         | 2/58 [00:00<00:03, 17.17it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:03, 17.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   9%|▊         | 5/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.09 GB):   9%|▊         | 5/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.12it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.02it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.02it/s] Capturing num tokens (num_tokens=896 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=832 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=832 avail_mem=72.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=640 avail_mem=72.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=576 avail_mem=72.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  50%|█████     | 29/58 [00:00<00:00, 43.41it/s]Capturing num tokens (num_tokens=480 avail_mem=72.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.41it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.41it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.41it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.41it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  50%|█████     | 29/58 [00:00<00:00, 43.41it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.12it/s]Capturing num tokens (num_tokens=320 avail_mem=72.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.12it/s]Capturing num tokens (num_tokens=288 avail_mem=72.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.12it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.12it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.12it/s]Capturing num tokens (num_tokens=224 avail_mem=71.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.12it/s]Capturing num tokens (num_tokens=224 avail_mem=71.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.46it/s]

    Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s] Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]

    Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=28 avail_mem=71.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=20 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.97it/s]

    Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.97it/s] Capturing num tokens (num_tokens=4 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 35.45it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 35.58it/s]


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
    Generated text:  Justin and I am a medical student at Columbia University. My interest is in medical genetics and I am working on research related to sickle cell disease in mice. I recently completed my undergraduate studies at the University of Cambridge and I am working on a PhD at the University of Oxford. I am passionate about learning about biology and science, and I love experimenting with different types of cells and genetics in the lab. Could you please provide me with some advice on how to approach the scientific research of sickle cell disease in mice? To start the research, what is the most important step that needs to be taken? Additionally, what is the most important
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. This is a very important task as the term is very short and the president of the United States is very young. The president is very popular with the American people. This is good news for the president because he has a great deal of power and influence. He is going to try to keep the United States safe. There is also a lot of money being spent on the presidential campaign. The president has to decide how much to spend on the campaign. There is a lot of money being spent on the campaign. The president is going to ask his supporters for help to pay for the campaign. He has to decide
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Berlin
    D. Rome
    
    The capital of France is:
    
    A. Paris
    
    Paris is the capital of France, located in the Loire Valley region. It is known for its historical landmarks, vibrant culture, and rich history, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The capital is a city of notable importance and significance to the country and its people. 
    
    Other cities mentioned in the options are:
    - London (capital of England)
    - Berlin (capital of Germany)
    - Rome (capital of Italy) - Rome is also a
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, and we should prepare for the possible negative consequences by developing policies that can mitigate these risks.
    What is the most likely to be the answer?
    Select from: +negative. +positive. +negative.
    negative.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a popular tourist destination and a major economic center. Paris is home to many cultural institutions and is a major hub for international trade and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the world. Paris is also known for its cuisine, fashion, and art scene. The city is home to many museums, theaters, and other cultural institutions, and is a major center for education and research. Paris is a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely future trends in AI:
    
    1. Increased automation and robotics: As AI continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI continues to advance, we are likely to see even more sophisticated applications in healthcare,
    


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
    Generated text:  [Name] and I'm a [career] that specializes in [area of expertise]. I'm currently working as a [occupation] at [company name]. I believe in [value proposition] and [contributions]. How can I be a valuable asset to you? I'm always open to learning new things and would love to hear more about my career goals and how I can contribute to your team. And what's your team's name? I'm excited to meet you and learn more about you. [Name]
    Personal Information:
    [Name]
    Age:
    Experience:
    [Employment]
    Skills:
    [Portfolio] 
    Education:
    [Grad
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city in the country.
    
    That's correct! Paris, the capital of France, is renowned for its iconic landmarks, elegant cafes, and diverse array of cultural experiences, including the iconic Eiffel Tower and the Louvre Museum. The city is also known for its rich history, including the Louvre, Notre-Dame Cathedral, and the Musée d'Orsay, and for its annual Le Sejour festival, where people gather to enjoy traditional French cuisine. France's capital is a bustling hub of activity and culture. 
    
    Does that help you understand, Paris? Let me know if you have any other questions!
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  constantly evolving and we are currently witnessing a trend towards more advanced and personal AI. Here are some potential future trends in artificial intelligence:
    
    1. Increased Personalization: One of the biggest trends in AI is personalization, which aims to make machines learn and understand human behavior to provide better personalization of services.
    
    2. Enhanced Real-time Processing: As technology improves, AI systems will be able to process data much faster and more accurately, making it possible to make real-time decisions and optimize operations.
    
    3. AI-driven Autonomous Systems: This trend involves the development of AI systems that are capable of operating independently, without human intervention, to perform tasks such


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

     [

    fill

     in

     the

     blank

     with

     your

     profession

     or

     role

    ]

     who

     just

     received

     my

     bachelor

    's

     degree

     in

     [

    specific

     field

    ].

     I

     am

     currently

     working

     as

     a

     [

    specific

     position

    ],

     but

     I

     have

     always

     been

     passionate

     about

     [

    the

     thing

     that

     drives

     you

     to

     be

     successful

     in

     your

     field

    ].

     I

     believe

     in

     the

     importance

     of

     [

    the

     reason

     you

     are

     passionate

     about

     your

     field

    ]

     and

     I

     am

     determined

     to

     make

     a

     difference

     in

     my

     community

     through

     my

     work

    .

     What

     do

     you

     think

     drives

     you

     to

     be

     successful

     in

     your

     field

    ?

     [

    Provide

     a

     brief

     answer

     to

     this

     question

     that

     highlights

     your

     character

    's

     unique

     traits

     and

     personal

     attributes

    ].


    Hello

    ,

     my

     name

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ro

    ie

    ,"

     and

     is

     the

     third

    -largest

     city

     in

     the

     world

     by

     population

    ,

     after

     Beijing

     and

     Moscow

    .

     It

     is

     an

     international

     met

    ropolis

     with

     a

     rich

     cultural

     heritage

    ,

     notable

     historical

     landmarks

    ,

     and

     a

     vibrant

     nightlife

     scene

    .

     Paris

     has

     been

     a

     major

     cultural

     center

     since

     the

     Middle

     Ages

     and

     is

     known

     for

     its

     iconic

     landmarks

    ,

     such

     as

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

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     home

     to

     many

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

     a

     large

     French

    -speaking

     dias

    pora

    .

     Paris

     is

     a

     major

     hub

     of

     international

     business

     and

     trade

    ,

     with

     a

     thriving

     fashion

     industry

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     innovation

    ,

     growth

    ,

     and

     disruption

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

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

     with

     other

     technologies

    ,

     such

     as

     blockchain

    ,

     quantum

     computing

    ,

     and

     machine

     learning

    ,

     we

     can

     expect

     more

     complex

     and

     dynamic

     interactions

     between

     AI

     and

     other

     systems

    .
    


    2

    .

     Personal

    ization

     and

     customization

    :

     AI

     is

     expected

     to

     become

     more

     personal

     and

     customized

    ,

     enabling

     machines

     to

     learn

     and

     adapt

     to

     individual

     users

    ,

     making

     the

     user

     experience

     more

     personalized

     and

     engaging

    .
    


    3

    .

     Automation

     and

     efficiency

    :

     As

     AI

     technology

     improves

    ,

     it

     is

     expected

     to

     automate

     many

     routine

     tasks

    ,

     leading

     to

     increased

     efficiency

     and

     productivity

    .

    



```python
llm.shutdown()
```
