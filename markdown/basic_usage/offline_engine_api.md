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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.16it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]

    Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 19.15it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 33.17it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 33.17it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 33.17it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 33.17it/s]

    Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 33.17it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 33.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 17.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 17.49it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:03, 17.49it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:03, 17.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.46it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 27.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 27.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 27.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:01, 27.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:01, 27.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.42it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.09it/s] Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]

    Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  50%|█████     | 29/58 [00:00<00:00, 35.74it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  50%|█████     | 29/58 [00:00<00:00, 35.74it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  50%|█████     | 29/58 [00:00<00:00, 35.74it/s]Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  50%|█████     | 29/58 [00:00<00:00, 35.74it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  50%|█████     | 29/58 [00:01<00:00, 35.74it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.54it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.12it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.12it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.12it/s]Capturing num tokens (num_tokens=208 avail_mem=73.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.12it/s]

    Capturing num tokens (num_tokens=192 avail_mem=73.90 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.12it/s]Capturing num tokens (num_tokens=192 avail_mem=73.90 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=176 avail_mem=73.63 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 27.56it/s]Capturing num tokens (num_tokens=128 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 27.56it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 27.56it/s]Capturing num tokens (num_tokens=96 avail_mem=73.34 GB):  76%|███████▌  | 44/58 [00:01<00:00, 27.56it/s] Capturing num tokens (num_tokens=96 avail_mem=73.34 GB):  81%|████████  | 47/58 [00:01<00:00, 25.99it/s]Capturing num tokens (num_tokens=80 avail_mem=73.34 GB):  81%|████████  | 47/58 [00:01<00:00, 25.99it/s]Capturing num tokens (num_tokens=64 avail_mem=72.83 GB):  81%|████████  | 47/58 [00:01<00:00, 25.99it/s]Capturing num tokens (num_tokens=48 avail_mem=72.63 GB):  81%|████████  | 47/58 [00:01<00:00, 25.99it/s]Capturing num tokens (num_tokens=32 avail_mem=72.63 GB):  81%|████████  | 47/58 [00:01<00:00, 25.99it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.33it/s]Capturing num tokens (num_tokens=28 avail_mem=72.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.33it/s]Capturing num tokens (num_tokens=24 avail_mem=72.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.33it/s]Capturing num tokens (num_tokens=20 avail_mem=72.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.33it/s]Capturing num tokens (num_tokens=20 avail_mem=72.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 26.78it/s]Capturing num tokens (num_tokens=16 avail_mem=72.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 26.78it/s]Capturing num tokens (num_tokens=12 avail_mem=72.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 26.78it/s]Capturing num tokens (num_tokens=8 avail_mem=72.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 26.78it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=72.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 27.29it/s]Capturing num tokens (num_tokens=4 avail_mem=72.60 GB):  98%|█████████▊| 57/58 [00:01<00:00, 27.29it/s]Capturing num tokens (num_tokens=4 avail_mem=72.60 GB): 100%|██████████| 58/58 [00:02<00:00, 28.91it/s]


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
    Generated text:  Alex, and I'm a science fiction author. I started writing in 2007 and have been writing fiction, poetry, graphic novels and comics ever since.
    What is your writing process like? I started with a rough idea and then I would work on it. I would write the first draft of a short story and revise it. Then I would write a novel, then an epic.
    Sometimes I am in a rush to finish things. I work on a book in about 100 to 200 hours. I do not want to write about myself or people I know. I don't want to be a
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to declare a state of emergency in his state. He has a large database of all the news articles from the past 10 years that mention the topic of the state's economy. The database has a column for whether or not the article is classified as "High Priority". He wants to know how many articles are classified as "High Priority". Unfortunately, the database is broken and some articles are missing. He also wants to know how many of those missing articles are classified as "High Priority". Given that the database is missing 8 articles, what is the total number of articles that are classified as "High Priority"?
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it is the capital of the country of France. It is also the seat of government for the 26th department of the French Republic.
    Does this next sentence follow, given the above text?
    The capital of France is situated on the Atlantic Ocean.
    
    Options are:
     1). yes.
     2). no.
    2). no.
    
    The capital of France is not located on the Atlantic Ocean. The capital is located in the heart of Paris, a major city in the region of Paris, which is on the French Riviera. The Atlantic Ocean is to the west of Paris, and the capital is situated between two large
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and uncertain how to bring it into the real world
    
    The AI industry is a very fast-moving and innovating industry with many opportunities and challenges. One of the big challenges that the industry faces is that the AI industry has been revolutionized by the advent of artificial intelligence (AI) and machine learning (ML). The impact of this revolution on the real world is unpredictable.
    
    The real world is a multi-faceted field that is challenging to understand and to predict, but AI technology has the potential to have a huge impact on this field. As a result, the future of AI is uncertain, and the amount of uncertainty on the horizon


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. Let's chat about [mention a specific topic or activity you enjoy doing]. I look forward to meeting you! [Name] [Company Name] [Company Address] [Company Phone Number] [Company Email] [Company Website] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company GitHub Profile] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for business, finance, and tourism, making it a popular destination for tourists and locals alike. The city is known for its annual Eiffel Tower Festival, which attracts millions of visitors each year. Paris is a city of contrasts, with its traditional French architecture and modern fashion, and its rich cultural heritage.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As the technology continues to advance, we can expect to see even more sophisticated AI systems being used in healthcare, such as personalized medicine, disease diagnosis, and treatment planning.
    
    2. Integration of AI into everyday life: AI is already being integrated into our daily lives, from smart home devices to self-driving cars. As the technology continues to evolve, we can expect to see even more
    


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
    Generated text:  [Name] and I'm a professional software developer with over [number] years of experience. I'm a team player and have a natural ability to solve complex problems quickly and efficiently. I am a problem solver, with a drive to achieve exceptional results. I love to work on exciting projects that have a positive impact on others and the community. I am passionate about being a part of the development and innovation within the company. I am always looking to learn and grow, and I am always eager to contribute to the team's success. If you're interested in joining our team, I would love to hear from you! [Name] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is the largest city in both France and the European Union. 
    
    To add more context:
    
    1. Paris is located on the southern bank of the Seine river, which flows through the city center.
    2. It is situated on the Atlantic coast, surrounded by the Mediterranean Sea on three sides and the Rhône River on the eastern coast.
    3. The city is divided into 12 districts, each with its own unique characteristics and landmarks.
    
    Overall, Paris is a city of contrasts, with its distinct features and iconic landmarks reflecting the country's rich cultural heritage and modernity. Its status as the capital has influenced
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  currently being shaped by several trends that are expected to continue. These trends include the increasing use of AI in sectors such as healthcare, finance, and manufacturing, the growing importance of AI in creating new technologies, and the increasing reliance on AI in decision-making processes. Some of the potential trends include:
    
    1. Enhanced AI technology: As AI technology continues to improve, we can expect to see more powerful and flexible AI systems that can perform a wider range of tasks. This could lead to new opportunities in fields such as education, healthcare, and transportation.
    
    2. AI ethics and safety: As more and more AI systems are developed, there will be


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

    ]

     who

     enjoys

     [

    what

     you

     do

    ]

     and

     [

    reason

     for

     doing

     this

    ].

     I

     love

     [

    what

     you

     do

    ]

     and

     I

     believe

     in

     the

     [

    reason

     for

     doing

     this

    ].

     I

     believe

     in

     the

     [

    reason

     for

     doing

     this

    ]

     and

     I

     am

     passionate

     about

     [

    interest

    ].

     If

     you

     would

     like

     to

     know

     more

     about

     me

    ,

     you

     can

     contact

     me

     at

     [

    contact

     information

    ].

     I

    'm

     [

    age

    ]

     years

     old

    .

     I

     hope

     to

     continue

     learning

     and

     growing

     as

     a

     [

    career

     or

     hobby

    ].

     I

    'm

     always

     looking

     to

     improve

     and

     explore

     new

     ideas

     and

     perspectives

    .

     How

     can

     I

     best

     connect

     with

     you

    ?

     How

     can

     I

     best

     help

     you

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     historic

     and

     cultural

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

     and

     a

     modern

     met

    ropolis

     with

     a

     diverse

     population

    .

     It

     is

     home

     to

     many

     famous

     landmarks

     and

     attractions

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

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

     vibrant

     street

     life

    ,

     art

    ,

     cuisine

    ,

     and

     music

     scene

    .

     It

     is

     also

     home

     to

     numerous

     international

     institutions

     and

     organizations

    ,

     such

     as

     the

     French

     Academy

     of

     Fine

     Arts

    ,

     the

     Paris

     Opera

    ,

     and

     the

     European

     Union

    .

     Paris

     is

     a

     major

     hub

     for

     international

     business

    ,

     politics

    ,

     and

     culture

    ,

     and

     has

     played

     an

     important

     role

     in

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     to

     be

     one

     of

     the

     most

     exciting

     and

     transformative

     periods

     in

     history

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     AI

     accuracy

    :

     AI

     is

     getting

     better

     at

     performing

     tasks

     that

     require

     human

     intelligence

    ,

     such

     as

     image

     and

     speech

     recognition

    ,

     language

     understanding

    ,

     and

     decision

    -making

    .

     Future

     AI

     will

     continue

     to

     improve

     its

     accuracy

     and

     precision

     in

     these

     areas

    .
    


    2

    .

     AI

     integration

     with

     human

     emotion

    :

     AI

     is

     becoming

     more

     adept

     at

     sim

    ulating

     human

     emotions

    ,

     such

     as

     empathy

    ,

     passion

    ,

     and

     intuition

    .

     This

     integration

     could

     lead

     to

     AI

     systems

     that

     can

     better

     understand

     and

     respond

     to

     human

     emotions

    ,

     leading

     to

     more

     empath

    etic

     and

     compassionate

     AI

    .
    


    3

    .

     AI

     growth

     in

    



```python
llm.shutdown()
```
