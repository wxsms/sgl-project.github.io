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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.87it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:20,  2.49it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:20,  2.49it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:20,  2.49it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:20,  2.49it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:20,  2.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:10,  4.47it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:10,  4.47it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:10,  4.47it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:10,  4.47it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:10,  4.47it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:10,  4.47it/s]

    Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:10,  4.47it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:04,  8.29it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:02, 12.89it/s]

    Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]

    Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 26.41it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 36.15it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.93it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.38 GB):   7%|▋         | 4/58 [00:00<00:04, 13.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.14 GB):   7%|▋         | 4/58 [00:00<00:04, 13.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.15 GB):   7%|▋         | 4/58 [00:00<00:04, 13.15it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.15 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.16 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.17 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.17 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.54it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.19 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.20 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.20 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.19 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.20 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.17it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.22 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.25 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.99it/s]Capturing num tokens (num_tokens=960 avail_mem=74.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.99it/s] Capturing num tokens (num_tokens=896 avail_mem=74.25 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.99it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.25 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.99it/s]Capturing num tokens (num_tokens=768 avail_mem=74.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.99it/s]Capturing num tokens (num_tokens=768 avail_mem=74.24 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.59it/s]Capturing num tokens (num_tokens=704 avail_mem=74.23 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.59it/s]Capturing num tokens (num_tokens=640 avail_mem=74.23 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.59it/s]Capturing num tokens (num_tokens=576 avail_mem=74.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.59it/s]Capturing num tokens (num_tokens=512 avail_mem=74.21 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.59it/s]Capturing num tokens (num_tokens=512 avail_mem=74.21 GB):  50%|█████     | 29/58 [00:01<00:00, 32.47it/s]Capturing num tokens (num_tokens=480 avail_mem=74.21 GB):  50%|█████     | 29/58 [00:01<00:00, 32.47it/s]Capturing num tokens (num_tokens=448 avail_mem=74.22 GB):  50%|█████     | 29/58 [00:01<00:00, 32.47it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.22 GB):  50%|█████     | 29/58 [00:01<00:00, 32.47it/s]Capturing num tokens (num_tokens=384 avail_mem=74.21 GB):  50%|█████     | 29/58 [00:01<00:00, 32.47it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:00, 32.47it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=320 avail_mem=74.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=288 avail_mem=74.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=224 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=224 avail_mem=74.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=208 avail_mem=74.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.69it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=160 avail_mem=74.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=160 avail_mem=74.15 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=144 avail_mem=74.14 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.47it/s] Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  81%|████████  | 47/58 [00:01<00:00, 37.96it/s]Capturing num tokens (num_tokens=80 avail_mem=74.12 GB):  81%|████████  | 47/58 [00:01<00:00, 37.96it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.11 GB):  81%|████████  | 47/58 [00:01<00:00, 37.96it/s]Capturing num tokens (num_tokens=48 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:01<00:00, 37.96it/s]Capturing num tokens (num_tokens=32 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:01<00:00, 37.96it/s]Capturing num tokens (num_tokens=32 avail_mem=74.10 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=28 avail_mem=74.09 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=16 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=16 avail_mem=74.07 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.66it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.06 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.66it/s] Capturing num tokens (num_tokens=4 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=4 avail_mem=74.05 GB): 100%|██████████| 58/58 [00:01<00:00, 29.72it/s]


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
    Generated text:  Dina. I am from the north of Spain and I love to eat delicious Spanish food. There are many interesting and well-known dishes in Spain, and I always want to learn more. You can tell me about a dish that you have never eaten before. I will eat it and tell you how it tastes. You are welcome to share your favorite dish if you like.
    Sure, I would love to have a chance to taste some new Spanish dishes. What's your favorite dish? It could be anything from a traditional Spanish dish like paella to something entirely new like a tapas platter. Just let me know what you're looking
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term, and he is running against another president. They are going to have a special election to see who will be president. The first president was born in 1972, and the second president is born in 1985. How old is the president who will be running for president if he was born in 1977?
    
    To determine the age of the president who will be running for president, we need to find out how many years have passed since the first president was born and how many years have passed since the second president was born. We know that the second president was born in
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the ____.
    A. Atlantic
    B. Mediterranean
    C. Arno River
    D. Rhone River
    Answer:
    
    B
    
    Which of the following statements about the information security management system is true?
    A. An Information Security Management System (ISMS) is a set of written documents, procedures, people, and means designed to ensure the information security of an organization and its information assets
    B. An Information Security Management System (ISMS) is a set of written documents, procedures, and people designed to ensure the information security of an organization and its information assets
    C. An Information Security Management System (ISMS)
    ===============================
    Prompt: The future of AI is
    Generated text:  driven by a renewed focus on transparency, ethics, and accountability. To achieve this, researchers should focus on three key areas: privacy, security, and accountability. Privacy concerns are at the forefront of the AI era, and they are essential for ensuring that data used for AI is protected. Privacy is essential because AI has the potential to provide the same level of accuracy and efficiency that humans have in certain applications. It is important to protect individuals' data from being misused or exploited. This is especially important for individuals whose privacy is being violated and for businesses that are using AI to provide services to their customers. 
    
    Security concerns are also critical to


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm passionate about [Why You're Interested in the Industry], and I'm always looking for ways to [What You Can Do to Improve Your Career]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm passionate about [Why You're Interested in the Industry], and I'm always looking for ways to [What You Can Do to Improve Your Career].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    (Note: The statement provided is a factual statement about Paris, not a fictional one. The Eiffel Tower is a famous landmark in Paris, and Notre-Dame Cathedral is a major religious site in the city.) 
    
    The statement is concise and factual, providing a clear and accurate description of Paris's capital city. However, it could be improved by including more details about the city's history, culture, or attractions. For example, the statement could be expanded to mention that Paris is the capital of France and is home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and personalized medicine to virtual assistants and chatbots. Additionally, AI is likely to continue to be used for a wide range of applications, from healthcare and finance to transportation and entertainment. As AI becomes more integrated into our daily lives, it is likely to have a significant impact on the way we work, live, and interact with each other. However, it is also important to consider the potential risks and ethical concerns associated with AI
    


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
    Generated text:  [Name], and I'm an AI assistant. How can I help you today?
    As an AI assistant, I'm programmed to be helpful and efficient. How can I assist you? I'm here to assist you with any questions or tasks you have, whether it's to provide information, answer questions, or simply to engage in a conversation.
    If you have any specific questions or topics you'd like to discuss, feel free to ask, and I'll do my best to provide you with accurate and informative responses. And don't forget to take care of yourself and prioritize your well-being, as AI assistants are designed to be helpful, not
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the seat of the country's government and politics. It is located in the North Eastern region of the country and is the oldest capital city in Europe. The city was founded in the 6th century and is famous for its historical landmarks and vibrant culture. Paris is known for its fashion industry, culinary culture, and a wide range of cultural institutions, including the Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also home to many famous museums, including the Metropolitan Museum of Art and the Musée Rodin. Paris is a popular tourist destination and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  unpredictable and uncertain, but some possible trends that are currently being explored and studied include:
    
    1. Deep Learning: This approach involves using artificial neural networks to process and analyze large amounts of data, with the goal of improving accuracy and efficiency in various applications, including image recognition, natural language processing, and speech recognition.
    
    2. Automation and AI Ethics: With the increasing automation of many tasks, there is growing concern about the impact of AI on employment and privacy. There are also ethical issues related to AI that need to be addressed, such as the potential for AI to be used for malicious purposes.
    
    3. Quantum Computing: Quantum computers have the potential


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

    ],

     a

     [

    Your

     Profession

     or

     Job

     Title

    ]

     with

     over

     [

    Number

     of

     Years

    ]

     years

     of

     experience

     in

     [

    Industry

    /

    Field

    ].

     Throughout

     my

     career

    ,

     I

     have

     hon

    ed

     my

     skills

     and

     expertise

     in

     [

    Your

     Main

     Job

     Function

    ],

     ensuring

     that

     [

    Your

     Profession

     or

     Job

     Title

    ]

     remains

     at

     the

     forefront

     of

     [

    Industry

    /

    Field

    ]

     for

     decades

    .

     I

     am

     a

     [

    Your

     Character

     Trait

    /

    Ab

    ilities

    ]

     who

     is

     always

     [

    Your

     Character

     Trait

    /

    Ability

    ].

     Looking

     forward

     to

     exploring

     the

     world

     with

     you

    .

     [

    Your

     Name

    ]

     [

    Your

     Job

     Title

    ]

     [

    Your

     Contact

     Information

    ]

     [

    Your

     About

     Me

    ]

     [

    Your

     Values

    ]

     [

    Your

     Inter

    ests

    ]

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     renowned

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

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

     Mont

    mart

    re

    .

     Paris

     is

     also

     known

     for

     its

     rich

     cultural

     history

    ,

     including

     the

     annual

     Les

     Ch

    amps

    -

    É

    lys

    ées

     parade

    ,

     the

     Notre

     Dame

     Infer

    nal

    ,

     and

     the

     iconic

     Opera

     Garn

    ier

    .

     
    


    The

     city

     of

     Paris

     is

     a

     bustling

     met

    ropolis

     with

     a

     diverse

     population

     that

     includes

     French

    ,

     European

    ,

     and

     international

     residents

    .

     The

     city

     is

     home

     to

     some

     of

     the

     world

    ’s

     most

     famous

     museums

    ,

     including

     the

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

     Centre

     Pom

    pid

    ou

    .

     Paris

     is

     also

     a

     popular

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     diverse

    ,

     with

     numerous

     areas

     of

     development

     and

     potential

     applications

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

     Automation

    :

     AI

     is

     increasingly

     being

     used

     in

     automated

     systems

    ,

     including

     in

     manufacturing

    ,

     transportation

    ,

     healthcare

    ,

     and

     customer

     service

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     is

     expected

     to

     automate

     repetitive

     and

     mundane

     tasks

    ,

     freeing

     up

     human

     resources

     for

     more

     complex

    ,

     creative

     work

    .
    


    2

    .

     Enhanced

     Personal

    ization

    :

     AI

     is

     being

     used

     to

     personalize

     the

     user

     experience

    ,

     offering

     recommendations

     and

     insights

     based

     on

     user

     behavior

    ,

     preferences

    ,

     and

     context

    .

     This

     is

     likely

     to

     lead

     to

     more

     efficient

     and

     effective

     use

     of

     technology

    ,

     as

     well

     as

     a

     more

     personalized

     and

    



```python
llm.shutdown()
```
