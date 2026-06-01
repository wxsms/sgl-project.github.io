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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.75it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:05,  5.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:05,  5.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:05,  5.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:05,  5.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:05,  5.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:05,  7.02it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 17.20it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:06<00:01, 17.20it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:06<00:00, 23.63it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 30.49it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 30.49it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 30.49it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 30.49it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 30.49it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 30.49it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 30.49it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 30.49it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 15.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 15.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 15.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 15.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.95it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.45it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.70it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.70it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:01<00:00, 29.79it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 29.79it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.33it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.33it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.33it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.33it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.33it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.33it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.12it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.12it/s]

    Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.38it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:02<00:00, 38.38it/s]Capturing num tokens (num_tokens=12 avail_mem=74.24 GB):  90%|████████▉ | 52/58 [00:02<00:00, 38.38it/s]Capturing num tokens (num_tokens=12 avail_mem=74.24 GB):  97%|█████████▋| 56/58 [00:02<00:00, 11.62it/s]Capturing num tokens (num_tokens=8 avail_mem=74.24 GB):  97%|█████████▋| 56/58 [00:02<00:00, 11.62it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.23 GB):  97%|█████████▋| 56/58 [00:02<00:00, 11.62it/s]Capturing num tokens (num_tokens=4 avail_mem=74.23 GB): 100%|██████████| 58/58 [00:02<00:00, 21.54it/s]


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
    Generated text:  Jack. I'm a middle school student. I like football, music, and swimming. I like to exercise every day. I do morning exercises and play football after school. I have a pet dog, but I don't like to play with him. I like to get up early and go to bed late. I have breakfast with my family, and then I go to school. I go to bed at nine thirty. I have a big dinner with my family, then I go to bed. What about you? What do you like to do on Saturdays? I usually go to the park with my friends and play on the swings.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person who serves as the country's representative to the nation's highest governing body, the United States Congress, and is the highest leader in the United States. He or she is the head of the executive branch of the government and the President of the United States is a citizen of the United States and is elected to two terms.
    The current president of the United States is Donald Trump. He has been the President of the United States since January 2017.
    The president is elected by the citizens of the United States and is the head of the executive branch of the government. He or she is the highest leader in the United States
    ===============================
    Prompt: The capital of France is
    Generated text:  ( )
    
    A: Paris  
    B: Rome  
    C: Athens  
    D: Rome or Athens
    
    To determine the capital of France, we need to understand the political geography of the country. France is located in northeastern Europe, bordering the Atlantic Ocean to the west and the Mediterranean Sea to the east. The capital of France is Paris, which is the largest city in the country.
    
    Let's break down the options:
    A: Paris - This is the capital of France, which is the correct answer.
    B: Rome - This is the capital of Italy, not France.
    C: Athens - This is the capital of Greece, not France
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, according to a report published by the German Federal Ministry of Education and Research (BMBF). The report claims that the number of people who will benefit from the benefits of artificial intelligence will grow by a factor of 14 by 2045, compared to the figure of 10,000 currently in place.
    While the BMBF is the central agency responsible for defining and implementing technological policies, the report also notes that there is a strong correlation between the development of AI and innovation in technology, as well as in areas such as energy, healthcare, and urban planning. The BMBF is therefore


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, interesting fact about yourself]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, interesting fact about your hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I love [insert a short, interesting fact about your favorite book or movie]. I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other cultural institutions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the arts and literature. It is also home to many famous French artists and writers, including the painter Monet and the novelist Dumas. Paris is a popular tourist destination, with millions of visitors each year, making it one of the most visited cities in the world. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn and adapt to new situations and tasks. This could lead to more efficient and effective decision-making, as well as better human-computer interaction.
    
    2. Greater emphasis on ethical considerations: As AI systems become more complex and sophisticated, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability. This will require developers to consider the potential impact of their AI systems on society and to develop guidelines and regulations to ensure that they are used in a responsible and
    


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
    Generated text:  [Name] and I'm a [Age] year-old self-proclaimed geek. I'm always looking for new adventures and learning new things. I enjoy trying out new things and being outside. I'm a fan of all kinds of music, from rock to hip-hop, and love to read. I also love to play video games, especially those that are challenging and require a lot of teamwork. I have a strong sense of humor and enjoy making people laugh. I'm constantly learning and growing, and I'm excited to continue on this adventure. How can I become a part of this community? Join us at [Social Media Platform]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that was founded in 987 AD by Clovis I and is located on the Seine River. 
    
    Summary: France's capital city, located on the Seine River, is Paris. Founded in 987 AD by Clovis I, Paris is the second largest city in France and is known for its cultural, historical, and artistic heritage. It is also the seat of the government, the monarchy, and the most important cultural and religious centers in the country. The city's famous landmarks, including Notre-Dame Cathedral and the Louvre Museum, are some of France's most iconic tourist attractions.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be exciting and diverse, with many promising areas to explore. Here are some potential future trends in AI:
    
    1. Increased focus on ethical considerations: As more AI systems become more sophisticated, there will be a growing awareness of their potential to harm people or cause unintended consequences. This means that ethical considerations will become even more important as AI is developed and deployed.
    
    2. Advancements in AI for healthcare: AI has the potential to revolutionize healthcare by improving diagnostic accuracy, personalized treatment plans, and patient monitoring. We can expect to see significant advancements in AI-driven diagnostic tools, personalized medicine, and artificial intelligence for managing chronic diseases.
    
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

     Emily

    ,

     and

     I

    'm

     a

     [

    insert

     your

     profession

     here

    ,

     such

     as

     "

    writer

    ,"

     "

    teacher

    ,"

     "

    sales

    person

    ,"

     etc

    .

    ].

     I

    'm

     an

     [

    insert

     your

     age

    ,

     such

     as

     "

    teen

    ager

    ,"

     "

    adult

    ,"

     "

    teen

    ager

    ,

     "

     etc

    .

    ].

     I

    've

     always

     had

     a

     passion

     for

     writing

    ,

     and

     I

     love

     to

     create

     stories

     and

     share

     them

     with

     others

    .

     I

    'm

     also

     an

     avid

     reader

     and

     love

     to

     learn

     new

     things

     through

     books

     and

     online

     resources

    .

     In

     my

     free

     time

    ,

     I

     enjoy

     playing

     sports

     and

     hanging

     out

     with

     friends

    .

     Thank

     you

     for

     having

     me

    !

     Based

     on

     the

     details

     provided

    ,

     what

     is

     the

     character

    's

     profession

     and

     age

    ?

     


    
    
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

     France

     and

     the

     seat

     of

     the

     French

     government

    .

     It

     has

     been

     the

     capital

     since

     

    1

    8

    0

    4

    ,

     and

     its

     main

     landmarks

     include

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

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     architecture

    ,

     and

     is

     a

     major

     center

     of

     commerce

    ,

     culture

    ,

     and

     entertainment

     in

     Europe

    .

     The

     city

     is

     also

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     other

     attractions

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     significant

     economic

     center

     in

     France

    .

     
    


    In

     summary

    ,

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     the

     country

    ,

     known

     for

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

     with

     many

     possibilities

     and

     potential

     applications

    .

     Here

     are

     some

     of

     the

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

     machine

     learning

    :

     With

     the

     increasing

     amount

     of

     data

     and

     complex

     tasks

    ,

     machine

     learning

     is

     expected

     to

     become

     more

     sophisticated

     and

     accurate

    .

     This

     will

     lead

     to

     more

     personalized

     and

     adaptive

     AI

     systems

     that

     can

     learn

     from

     feedback

     and

     adapt

     to

     new

     situations

    .
    


    2

    .

     Improved

     natural

     language

     processing

    :

     Natural

     language

     processing

     will

     continue

     to

     improve

    ,

     allowing

     machines

     to

     understand

     and

     interpret

     human

     language

     more

     accurately

     and

     naturally

    .

     This

     will

     lead

     to

     more

     convers

    ational

     AI

    ,

     where

     machines

     can

     interact

     with

     humans

     in

     human

    -like

     ways

    .
    


    3

    .

     Cyber

    security

    :

     As

     AI

     systems

     become

    



```python
llm.shutdown()
```
