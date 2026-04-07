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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.94it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.93it/s]


    2026-04-07 01:03:45,803 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 01:03:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:42,  1.29it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:42,  1.29it/s]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:42,  1.29it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:03<00:42,  1.29it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:17,  3.06it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:17,  3.06it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:17,  3.06it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:17,  3.06it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:17,  3.06it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.98it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 13.52it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:02, 13.52it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 20.33it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 20.33it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 20.33it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 20.33it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 20.33it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 20.33it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 31.50it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 31.50it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 31.50it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 31.50it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 31.50it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 31.50it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 31.50it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 36.05it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 36.05it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 36.05it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 36.05it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 36.05it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 36.05it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 36.05it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 36.05it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 42.57it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 42.57it/s]

    Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 42.57it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 42.57it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 42.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.46it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  50%|█████     | 29/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.42it/s]

    Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.87it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]

    Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.53it/s]

    Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.39it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 39.71it/s]


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
    Generated text:  Vicky. I'm a high school student in our city. I love reading. When I go out, I like to buy some books. Now I want to ask you some questions. What do you like to do after school? What do you like to do on weekends? What do you like to do in summer? What do you like to do when you are free? Can you tell me something about your school life? Write an article about yourself in about 80 words.
    Sure, I can help you write an article about yourself in about 80 words. What would you like the article to be about? This could
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize a budget for the upcoming fiscal year. In the budget, there are four items to be included: military spending, education spending, and the balance of trade. Military spending is expected to be $200 billion, education spending is expected to be $300 billion, and the balance of trade is expected to be $100 billion. If the president wants to balance the budget by $50 billion, how much is the remaining amount for the remaining items? Let's break down the budget allocation and see if we can find a way to make the budget balance by $50 billion.
    
    1. **
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which of the following can best describe Paris?
    A. The capital of France
    B. The French capital
    C. The capital city of France
    D. The city of Paris
    Answer:
    
    C
    
    Regarding the statement "only a few people in the world can understand the meaning of life," which of the following conclusions is the most reasonable?
    A. Only a few people in the world can understand the meaning of life.
    B. Only a few people in the world do not understand the meaning of life.
    C. Not all people in the world can understand the meaning of life.
    D. Some people in the world can understand
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and as technology becomes more integrated into our daily lives, it will continue to transform various aspects of society, including education. With the increasing demand for personalized learning experiences, AI is poised to revolutionize the field of education. By analyzing student data and leveraging machine learning algorithms, AI can provide personalized learning experiences that cater to individual student needs and preferences. This can lead to improved student outcomes, greater student engagement, and ultimately, a more personalized education experience for all students. As AI continues to improve, it will be essential for educators to stay updated and learn how to effectively use AI to create more effective learning experiences for students. 
    
    One


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [job title]. I'm a [job title] at [company name] and I'm always looking for ways to [describe a new skill or initiative]. I'm a [job title] at [company name] and I'm always looking for ways to [describe a new skill or initiative]. I'm a [job title] at [company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is home to iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, as well as numerous museums, theaters, and restaurants. Paris is also known for its vibrant culture, including its annual Eiffel Tower Festival and its annual fashion week. The city is a major economic and cultural center, with a strong emphasis on tourism and a growing economy. Paris is a city of contrasts, with its modern architecture and historical landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This could lead to more sophisticated and personalized AI systems that can learn from and adapt to human behavior and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations and responsible use of AI. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Increased focus on
    


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
    Generated text:  [Name], and I'm a [career or occupation] with a passion for [What you do for work]. I'm [age] years old and I love [What you do for fun]. I enjoy [What you do for hobbies]. What are some things you're known for? As an AI language model, I'm not inherently known for anything. However, my creators recognize that I can be a valuable asset in many fields, including machine learning, natural language processing, and more. Can you tell me more about your background or skills that you're known for? I'm here to help if you have any questions about my abilities
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    (Note: The statement was originally provided as "A) Paris is France's capital city." However, the question asks for a concise statement, which is a brief, accurate description rather than a single word or phrase.) 
    
    Given the instruction to ensure the provided statement remains a concise factual statement, I will rephrase it to "The capital of France is Paris." This maintains the original meaning while providing a more precise answer. 
    
    If you need a statement that conveys the same idea as "A) Paris is France's capital city," you could say "Paris is the capital of France." This remains a concise factual statement.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of trends that will shape how it is developed, deployed, and used. Here are some potential future trends in AI:
    
    1. Increased focus on ethical and social implications: As AI systems become more advanced and widespread, there will be a greater emphasis on ensuring that they are developed and used in ways that respect human values and prevent harmful consequences. This will involve examining the potential for AI to exacerbate existing social inequalities and discrimination, and developing strategies for mitigating those effects.
    
    2. Continued development of machine learning algorithms: Machine learning is a core component of AI, and its development will likely continue to evolve over


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

     Sarah

    .

     I

     am

     a

     software

     engineer

     with

     a

     passion

     for

     creating

     innovative

     solutions

     that

     solve

     complex

     problems

    .

     My

     expertise

     lies

     in

     building

     systems

     that

     optimize

     performance

    ,

     user

     experience

    ,

     and

     cost

    -effect

    iveness

    .

     I

     am

     always

     looking

     for

     ways

     to

     improve

     the

     quality

     of

     my

     work

     and

     continuously

     learn

     new

     technologies

    .

     If

     you

    're

     looking

     for

     someone

     who

     can

     help

     make

     a

     positive

     impact

    ,

     I

    'm

     excited

     to

     meet

     you

    !

     What

    's

     your

     name

     and

     what

     do

     you

     do

    ?

     I

    'm

     Sarah

    .

     What

    's

     your

     name

     and

     what

     do

     you

     do

    ?


    Sarah

     is

     a

     software

     engineer

     with

     a

     passion

     for

     creating

     innovative

     solutions

     that

     solve

     complex

     problems

    .

     Her

     expertise

     lies

     in

     building

     systems

     that

     optimize

     performance

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     central

     region

     of

     the

     country

     and

     the

     largest

     city

     and

     seat

     of

     government

     of

     France

    .


    The

     French

     capital

    ,

     Paris

    ,

     is

     the

     largest

     city

     and

     seat

     of

     government

     of

     France

    ,

     with

     a

     population

     of

     approximately

     

    2

    .

    7

     million

     people

    .

     It

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

     the

     Lou

    vre

     Museum

    ,

     and

     other

     historic

     and

     cultural

     landmarks

    .

     Paris

     is

     also

     a

     major

     transportation

     hub

    ,

     with

     the

     Lou

    vre

     Museum

     as

     the

     world

    's

     largest

     art

     museum

     and

     the

     most

     visited

     museum

     in

     the

     world

    ,

     and

     the

     E

    iff

    el

     Tower

     as

     the tallest

     structure

     in

     the

     world

    .

     The

     city

     is

     also

     renowned

     for

     its

     food

     culture

    ,

     including

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     driven

     by

     several

     key

     trends

     that

     are

     expected

     to

     shape

     the

     industry

     in

     the

     coming

     years

    :
    


    1

    .

     AI

     will

     continue

     to

     become

     more

     integrated

     into

     daily

     life

    :

     As

     more

     applications

     of

     AI

     are

     integrated

     into

     various

     industries

    ,

     from

     healthcare

     to

     finance

    ,

     we

     can

     expect

     that

     we

     will

     see

     more

     AI

    -driven

     solutions

     being

     implemented

     in

     everyday

     life

    .
    


    2

    .

     AI

     will

     become

     more

     accessible

    :

     With

     the

     rise

     of

     AI

    -powered

     devices

     such

     as

     smartphones

    ,

     smart

     homes

    ,

     and

     self

    -driving

     cars

    ,

     there

     is

     no

     doubt

     that

     AI

     will

     become

     more

     accessible

     to

     a

     wider

     range

     of

     people

    .

     This

     will

     lead

     to

     a

     more

     widespread

     adoption

     of

     AI

     solutions

     in

     various

     industries

    .
    


    3

    .

     AI

    



```python
llm.shutdown()
```
