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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.00it/s]


    2026-05-05 14:33:38,377 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 14:33:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.83it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 33.41it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  31%|███       | 18/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=960 avail_mem=71.99 GB):  31%|███       | 18/58 [00:00<00:01, 33.41it/s] Capturing num tokens (num_tokens=960 avail_mem=71.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.25it/s]Capturing num tokens (num_tokens=896 avail_mem=71.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.25it/s]Capturing num tokens (num_tokens=832 avail_mem=71.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.25it/s]Capturing num tokens (num_tokens=768 avail_mem=71.98 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.25it/s]Capturing num tokens (num_tokens=704 avail_mem=71.98 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.25it/s]Capturing num tokens (num_tokens=640 avail_mem=71.98 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.25it/s]Capturing num tokens (num_tokens=640 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=576 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=512 avail_mem=71.96 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.08it/s]

    Capturing num tokens (num_tokens=480 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=448 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=416 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=416 avail_mem=71.98 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.46it/s]Capturing num tokens (num_tokens=384 avail_mem=71.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.46it/s]Capturing num tokens (num_tokens=352 avail_mem=71.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.46it/s]Capturing num tokens (num_tokens=320 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.46it/s]Capturing num tokens (num_tokens=288 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.46it/s]Capturing num tokens (num_tokens=256 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=256 avail_mem=71.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=240 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.01it/s]

    Capturing num tokens (num_tokens=224 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=208 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=192 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=176 avail_mem=71.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=176 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=160 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=144 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=128 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=112 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.33it/s]

    Capturing num tokens (num_tokens=112 avail_mem=71.93 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=96 avail_mem=71.93 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.23it/s] Capturing num tokens (num_tokens=80 avail_mem=71.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=64 avail_mem=71.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=48 avail_mem=71.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=32 avail_mem=71.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=32 avail_mem=71.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=28 avail_mem=71.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=24 avail_mem=71.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=20 avail_mem=71.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=16 avail_mem=71.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.02it/s]

    Capturing num tokens (num_tokens=12 avail_mem=71.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=12 avail_mem=71.90 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.50it/s]Capturing num tokens (num_tokens=8 avail_mem=71.90 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.50it/s] Capturing num tokens (num_tokens=4 avail_mem=71.89 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.50it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB): 100%|██████████| 58/58 [00:01<00:00, 36.34it/s]


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
    Generated text:  Tricia, and I'm a (I'll tell you my job) for the city of Seattle. My job is to manage the city's social media accounts, including Twitter, Facebook, Instagram, and YouTube. As part of my role, I will answer all questions from our community about the city, share interesting updates on Seattle, and promote our community events, including races, festivals, and cultural events. If you have any questions, or if you have an idea for a blog or podcast, please reach out to me! To start, I need you to tell me about the type of media you currently manage on Twitter. 
    I
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a country. The country's president told the president that he is visiting the United States only if the president of the United States is visiting the country. Therefore, the president of the United States must go to the country.
    
    This reasoning is flawed in that it:
    
    A. Provides a one-sided argument
    B. Argues for the truth of a universal statement
    C. Makes a false statement
    D. Argues from a necessary condition to a sufficient condition To determine why the reasoning is flawed, let's carefully analyze the structure of the argument step by step.
    
    The argument is as follows:
    1. The president of the United States is
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    
    The capital of France is Paris. 
    
    The options are:
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    
    Paris is the capital city of France. The capital of France is Paris. 
    
    Therefore, the correct answer is A. Paris. 
    
    Since the task requires choosing only one answer, and Paris is the only option that matches the criteria of being the capital of France, the answer is:
    
    A. Paris
    You are an AI assistant that helps you understand and revise texts. Read the sentence carefully and choose the
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly changing. With the rapid advancements in technology and AI becoming more accessible, the field of AI is continually evolving. One area of AI that is gaining significant attention and impacting businesses today is the use of natural language processing (NLP). NLP has the potential to transform the way we interact with technology and improve the efficiency of our daily lives.
    
    One of the key benefits of NLP is that it allows machines to understand and process human language in a way that is similar to human language. This makes it possible to build more sophisticated and accurate chatbots, language translation software, and other AI-powered tools.
    
    In addition to its ability to improve


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm excited to learn and grow with you. What's your name, and what's your job title? [Name] [Job Title] [Company Name] [Company Address] [Company Phone Number] [Company Email] [Company Website] [Name] [Job Title] [Company Name] [Company Address] [Company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling city with a rich history and a diverse population, making it a popular tourist destination. The city is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. The city is also home to many international organizations and events, making it a hub for global affairs. Overall, Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust and transparent AI systems that are designed to minimize harm and maximize safety.
    
    3. Increased
    


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
    Generated text:  [Name] and I'm a [occupation] who have [number of years] years of experience in [industry]. I bring a unique combination of [benefits] to my work. As a [job title], I'm [introduction], and I'm here to [description of their work] with [description of their experience or education]. I'm always looking for opportunities to [describe a problem], and [introduce a solution] to help us solve it. I believe in [value proposition], and I strive to do [something]. I'm always committed to [commitment to something] and I'm always looking to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its beautiful architecture, rich culture, and iconic landmarks such as the Eiffel Tower and Louvre Museum. The city is also renowned for its food, such as croissants and pastries, as well as its wine and nightlife. Paris has a diverse and vibrant culture, and it is home to numerous museums, theaters, and other cultural institutions. It is a popular tourist destination and attracts millions of visitors every year. Paris is a major hub of intellectual and artistic life and is considered one of the most important cities in the world. The city is also home to many world-renowned museums and art galleries. Additionally
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but several trends are likely to shape the field in the coming years:
    
    1. Increased focus on ethical AI: As more individuals and organizations are recognizing the need for responsible AI, there will be increased focus on creating AI that is not only effective but also aligned with ethical principles.
    
    2. AI will become more integrated with other technologies: AI will become more integrated with other technologies such as blockchain, machine learning, and cloud computing, allowing for more complex and interconnected AI systems.
    
    3. AI will become more accessible: As AI technology becomes more advanced, there will be more opportunities for AI to be used in the real world, from


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

    .

     I

    'm

     a

     

    2

    5

    -year

    -old

     software

     engineer

     who

     loves

     exploring

     new

     technologies

     and

     learning

     new

     things

    .

     I

    'm

     currently

     working

     on

     a

     project

     for

     a

     startup

     that

     focuses

     on

     developing

     AI

    -powered

     recommendation

     systems

     for

     mobile

     devices

    .

     I

    'm

     passionate

     about

     using

     technology

     to

     improve

     people

    's

     lives

    ,

     and

     I

    'm

     always

     up

     for

     learning

     and

     trying

     new

     things

    .

     I

     love

     being

     a

     social

     media

     influ

    encer

    ,

     so

     I

    'm

     always

     on

     the

     lookout

     for

     new

     ideas

     and

     trends

     to

     share

     with

     my

     followers

    .

     I

     also

     love

     spending

     time

     outdoors

     and

     traveling

     to

     new

     places

    .

     Thank

     you

     for

     having

     me

    !

     

    🌍

    ✨

     #

    Software

    Engine

    er

     #

    In

    flu

    encer

     #

    Travel

    er

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     most

     populous

     city

     in

     the

     country

    ,

     and

     its

     name

     comes

     from

     the

     Latin

     word

     "

    par

    vis

    ,"

     meaning

     "

    front

     door

    ."


    Paris

     is

     the

     capital

     of

     France

     and

     is

     the

     most

     populous

     city

     in

     the

     country

    ,

     with

     a

     population

     of

     approximately

     

    2

    .

    2

     million

     people

     as

     of

     

    2

    0

    2

    1

    .

     The

     city

     is

     known

     for

     its

     stunning

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     economic

     center

     in

     Europe

    .

     It

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

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

     also

     famous

     for

     its

     wine

     and

     cheese

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     new

     technologies

     and

     applications

     being

     developed

     that

     promise

     to

     transform

     how

     we

     live

    ,

     work

    ,

     and

     interact

     with

     the

     world

     around

     us

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

     advanced

    ,

     it

     will

     be

     integrated

     more

     closely

     with

     other

     technologies

    ,

     such

     as

     machine

     learning

    ,

     computer

     vision

    ,

     and

     natural

     language

     processing

    .

     This

     will

     enable

     AI

     to

     better

     understand

     and

     interpret

     complex

     data

    ,

     which

     will

     lead

     to

     new

     applications

     in

     areas

     like

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sophisticated

    ,

     there

     will

     be

     a

     need

     to

     address

     issues

     related

     to

    



```python
llm.shutdown()
```
