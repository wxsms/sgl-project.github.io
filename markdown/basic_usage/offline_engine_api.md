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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.41it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.40it/s]


    2026-04-15 02:15:09,092 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 02:15:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.77it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.77it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.77it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.89it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.73it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 48.75it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 48.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.83it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:03, 14.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:03, 14.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:03, 14.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:03, 14.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.43it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.43it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.43it/s]

    Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.43it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.43it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.43it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.91it/s]

    Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 40.72it/s]

    Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.26it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.94it/s]

    Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.00it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.00it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.00it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 35.79it/s]


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
    Generated text:  Sarah and I am a college student in my final year. I have worked as a consultant for several companies and have also been a web developer for several years. I was thinking to write a book that I have already written a few books that cover a lot of topics including business management, business strategy, and business communication. I have to write a book on business ethics. I am writing my book with the book publisher in mind to have a greater amount of feedback and also a possibility of getting a bigger chance to publish the book. The book is going to be a book on business ethics. I am having a hard time deciding what the book
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, but what exactly does the President of the United States do? What exactly does the President do?  Do these two questions have the same meaning?
    Yes, the two questions have the same meaning. They are both seeking information about the role and responsibilities of the President of the United States. The first question is a more general inquiry that encompasses the full scope of the President's duties, while the second question is a more specific inquiry that focuses on the President's main responsibilities. However, both questions are essentially asking the same thing about the President's responsibilities and duties. The difference in specificity of the question does not affect the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, a city with a population of about 3 million. Many of the residents live in apartments and buildings, and there are also a number of large residential areas, such as the "Rue du Champ-de-Fond" in the centre of Paris and the "Rue des Jardins" in the East of Paris. The city is often considered a symbol of France and its culture. How many apartments and buildings are there in Paris?
    To determine the total number of apartments and buildings in Paris, we need to analyze the given information step by step.
    
    1. **Identify the population of Paris:**
       The problem states
    ===============================
    Prompt: The future of AI is
    Generated text:  high tech, but it is also low tech. From the natural world to the internet, everything around us is made of things that can be processed by AI. So, what does AI mean for us?
    That is the question that Driving Sciencetiming is exploring. We will look at the history of AI, the role it is currently playing and the prospects for its future. In this episode we are joined by Mark Tait, Chief Scientist at the National Oceanic and Atmospheric Administration (NOAA). He has over 25 years of experience in the scientific field and in the AI field in particular.
    This week, we are talking


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for fashion, art, and music, and is home to many world-renowned museums, theaters, and restaurants. The city is known for its annual festivals, such as the Eiffel Tower Parade and the Carnaval de Paris, and its status as a global cultural hub. Paris is a city of contrasts, with its modern architecture and high-tech industries, as well
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation: AI is likely to become more prevalent in many industries, with automation becoming more widespread as machines become more sophisticated and capable of performing tasks that were previously done by humans.
    
    2. Personalization: AI is likely to become more personalized as machines learn more about individual users and their preferences. This could lead to more targeted marketing and personalized recommendations, as well as more efficient use of resources.
    
    3. Ethical and responsible AI: As AI becomes more
    


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
    Generated text:  [Name]. I am a [type of job or hobby] from [Location] and have always been interested in [field of interest or passion]. I am [age], and I believe that [reason why you are a good fit for the position]. If you would like to learn more about me, please feel free to ask me any questions. Your feedback would be greatly appreciated! (End self-introduction) [End] [End] The character's background, personal details, and the specific details of the job or hobby they are a part of are included in the text. The introduction is neutral, informative, and appears to be
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Source". It is the largest city in Europe and is home to many of France's most prominent cultural, historical, and artistic institutions, including the Louvre Museum, the Notre-Dame Cathedral, and the Palace of Versailles. Paris is also known for its fashion, gastronomy, and food culture, and is home to many world-renowned restaurants, including Le Bernardin and Le Figaro. Overall, Paris is a vibrant and diverse city with a rich cultural heritage that has been continuously influenced by various cultures and civilizations throughout its history. As one of the most visited cities in the world, Paris continues
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends:
    
    1. Increased focus on ethics and privacy: As AI becomes more integrated into our daily lives, there is a growing concern about its ethical implications. Governments and organizations are exploring ways to balance the potential benefits of AI with its potential risks and consequences. This includes developing guidelines and regulations that address issues such as bias, accountability, and transparency.
    
    2. Integration with other technologies: As AI continues to advance, it is likely to become more integrated with other technologies such as machine learning, natural language processing, and the Internet of Things (IoT). This integration could lead to a more seamless and interconnected experience


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

    ]

     and

     I

    'm

     a

     self-described

     [

    short

     description

     of

     your

     character

    ].

     I

    've

     always

     had

     a

     passion

     for

     [

    mention

     a

     hobby

     or

     interest

    ],

     and

     it

    's

     something

     that

     I

     strive

     to

     pursue

     with

     dedication

     and

     passion

    .

     Whether

     it

    's

     through

     my

     work

    ,

     personal

     projects

    ,

     or

     even

     just

     social

    izing

     with

     friends

     and

     family

    ,

     I

    'm

     constantly

     looking

     for

     new

     ways

     to

     explore

     my

     creative

     side

     and

     express

     myself

     through

     art

    ,

     music

    ,

     writing

    ,

     or

     any

     other

     medium

     that

     I

    'm

     comfortable

     with

    .

     I

    'm

     an

     [

    tell

     the

     reader

     about

     your

     character

    's

     personality

    ,

     interests

    ,

     and

     hobbies

    ,

     and

     how

     they

     relate

     to

     your

     own

    ].

     I

    'm

     [

    give

     a

     brief

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     culture

    .

     It

     is

     also

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    ,

     with

     millions

     of visitors

     annually

    .

     The

     city

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     the

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

     the

     Lou

    vre

     Museum

    ,

     among

     others

    .

     Paris

     is

     also

     known

     for

     its

     culinary

     scene

    ,

     with

     many

     restaurants

     and

     food

     markets

     offering

     delicious

     cuisine

    .

     The

     city

     is

     also

     home

     to

     many

     international

     institutions

    ,

     including

     the

     French

     Parliament

    ,

     the

     Academy

     of

     Sciences

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Overall

    , Paris

     is

     a

     city

     of

     art

    ,

     culture

    ,

     and

     historical

     significance

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

    ,

     as

     it

     will

     continue

     to

     evolve

     and

     expand

     our

     understanding

     of

     the

     world

     around

     us

    .

     Here

     are

     some

     potential

     trends

     that

     may

     come

     to

     play

     a

     significant

     role

     in

     shaping

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Enhanced

     intelligence

     and

     decision

    -making

    :

     AI

     is

     advancing

     rapidly

    ,

     and

     researchers

     are

     working

     to

     create

     machines

     that

     can

     make

     more

     accurate

     and

     informed

     decisions

    .

     This

     includes

     using

     machine

     learning

     algorithms

     to

     analyze

     large

     amounts

     of

     data

    ,

     identify

     patterns

    ,

     and

     make

     predictions

     on

     a

     wide

     range

     of

     topics

    .

     AI

     will

     continue

     to

     become

     more

     intelligent

    ,

     able

     to

     learn

     from

     experience

    ,

     and

     able

     to

     adapt

     to

     changing

     circumstances

    .
    


    2

    .

     Autonomous

     vehicles

    :

     AI

     is

     already

     being

     used

     in

    



```python
llm.shutdown()
```
