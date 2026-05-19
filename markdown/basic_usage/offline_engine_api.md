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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.44it/s]


    2026-05-19 22:18:14,026 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-19 22:18:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:14,  3.45it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:14,  3.45it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:14,  3.45it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:14,  3.45it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:14,  3.45it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:14,  3.45it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:14,  3.45it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:14,  3.45it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:04<00:05,  7.50it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:02, 13.26it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:02, 13.26it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:02, 13.26it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:02, 13.26it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:02, 13.26it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:02, 13.26it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:02, 13.26it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:04<00:02, 13.26it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 19.07it/s]

    Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 26.78it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s] 

    Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:04<00:00, 34.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 47.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.96 GB):   3%|▎         | 2/58 [00:00<00:03, 17.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.95 GB):   3%|▎         | 2/58 [00:00<00:03, 17.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.95 GB):   3%|▎         | 2/58 [00:00<00:03, 17.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=68.95 GB):   3%|▎         | 2/58 [00:00<00:03, 17.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.95 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.95 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.94 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.93 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.93 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.93 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=68.93 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=68.92 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.92 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.92 GB):  21%|██        | 12/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.92 GB):  21%|██        | 12/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.92 GB):  21%|██        | 12/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.91 GB):  21%|██        | 12/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.91 GB):  21%|██        | 12/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.90 GB):  21%|██        | 12/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.90 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.90 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.90 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.29it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=68.90 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.88 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.29it/s]Capturing num tokens (num_tokens=960 avail_mem=68.89 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.29it/s] Capturing num tokens (num_tokens=960 avail_mem=68.89 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=896 avail_mem=68.89 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=832 avail_mem=68.89 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=768 avail_mem=68.88 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=704 avail_mem=68.88 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=640 avail_mem=68.88 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=640 avail_mem=68.88 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=576 avail_mem=68.88 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.83it/s]

    Capturing num tokens (num_tokens=512 avail_mem=68.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=480 avail_mem=68.88 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=448 avail_mem=68.87 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=416 avail_mem=68.87 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=416 avail_mem=68.87 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=384 avail_mem=68.87 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=352 avail_mem=68.86 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=320 avail_mem=68.86 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=288 avail_mem=68.86 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=256 avail_mem=68.85 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=256 avail_mem=68.85 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.44it/s]Capturing num tokens (num_tokens=240 avail_mem=68.85 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.44it/s]

    Capturing num tokens (num_tokens=224 avail_mem=68.85 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.44it/s]Capturing num tokens (num_tokens=208 avail_mem=68.84 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.44it/s]Capturing num tokens (num_tokens=192 avail_mem=68.84 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.44it/s]Capturing num tokens (num_tokens=176 avail_mem=68.84 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.44it/s]Capturing num tokens (num_tokens=176 avail_mem=68.84 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=160 avail_mem=68.84 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=144 avail_mem=68.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=128 avail_mem=68.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=112 avail_mem=68.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=96 avail_mem=68.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.64it/s] Capturing num tokens (num_tokens=96 avail_mem=68.82 GB):  81%|████████  | 47/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=80 avail_mem=68.82 GB):  81%|████████  | 47/58 [00:01<00:00, 44.11it/s]

    Capturing num tokens (num_tokens=64 avail_mem=68.82 GB):  81%|████████  | 47/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=48 avail_mem=68.81 GB):  81%|████████  | 47/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=32 avail_mem=68.81 GB):  81%|████████  | 47/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=28 avail_mem=68.81 GB):  81%|████████  | 47/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=28 avail_mem=68.81 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=24 avail_mem=68.80 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=20 avail_mem=68.80 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=16 avail_mem=68.80 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=12 avail_mem=68.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=8 avail_mem=68.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.82it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=68.79 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=4 avail_mem=68.79 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=4 avail_mem=68.79 GB): 100%|██████████| 58/58 [00:01<00:00, 38.17it/s]


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
    Generated text:  Valerie and I am a student of Mathematics. I am at age 19 and I am new to my school. I am from a very poor family and I was always fed up with the life. I want to get out of my life and get a better life. I decided to go to college and I want to be a doctor. I have always been interested in medicine and I want to learn how to cure diseases. I would like to know what is the most important thing to get a doctor and why? 
    
    What do you think about getting a doctor? 
    
    What's your opinion on getting a doctor?
    Valerie's question
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. Who is this person? His name is George Washington.
    George Washington was the first president of the United States. He was born on April 17, 1732, in Mount Vernon, Virginia. He was the son of a wealthy planter. He spent many years in the service of his father, growing up in Virginia and eventually becoming a wealthy planter himself.
    When he was 17 years old, he won the vote of the yeoman farmers of Virginia by promising to help them raise their families in their own homes and educate them for a better life. The victory of George Washington at
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is a giant city with about 1 million inhabitants. This city has an ancient history. It was founded by the Romans. It has had many important cities over the centuries. In 1792, when the French Revolution broke out, the capital became the Bastille, a symbol of the revolution. In 1804, Napoleon took over the city. He made it his new capital, and it became known as Paris. Today, Paris is a great city. It has many of the old buildings and streets of the past. There are also new hotels and restaurants. It is also very popular. It
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of us all. With the rapid advancements in the field of machine learning, we are seeing the emergence of new technologies that are more advanced and capable than ever before. One such technology is the use of artificial neural networks, which are a type of machine learning algorithm that is used to identify patterns and make predictions based on large amounts of data.
    
    One of the main benefits of using artificial neural networks is their ability to learn from data and improve over time. This means that as the data changes and grows larger, the model can adapt and improve its accuracy. This is particularly important in areas such as image recognition, natural language processing,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for French culture, cuisine, and fashion. Paris is a bustling metropolis with a rich history and a diverse population. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. It is also a popular tourist destination, with millions of visitors annually. Paris is a cultural and economic hub of France and a major international city. Its status as the capital of France has made it a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Enhanced capabilities in natural language processing: AI is likely to continue to improve its ability to understand and respond to human language, including spoken and written language. This could lead to more sophisticated language models that can better understand context, tone, and intent.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence,
    


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
    Generated text:  [Your Name], and I'm a [career field] who has been dedicated to [Your occupation] for [number] years. I'm always up for learning and always looking for new experiences to add value to the industry. And in the future, I hope to expand my skill set and achieve [career goal]. Welcome to my personal page! 🌟
    
    ---
    
    Please note that you can replace the placeholders with your own personal information and experiences. The style of the introduction should reflect a neutral and professional tone. Let me know if you need any modifications to this example. 
    
    ---
    
    ### Example of a Neutral Self-Introduction
    
    **
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement is factually correct and provides the core information about the capital city's name and location within the country. It succinctly captures the essence of Paris' status as the largest and most influential city in France, offering readers a clear understanding of its place within the broader context of the nation's political and cultural landscape. 
    
    The factual aspect of the statement has been verified, ensuring its accuracy for use in various contexts, including educational materials, travel guides, and news articles. The statement is concise yet informative, making it suitable for a wide audience interested in learning about Paris and the nation's capital. It effectively communicates the capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some potential trends that are emerging are:
    
    1. Increased reliance on automation: As AI systems become more complex and integrated with other technologies, automation will become more prevalent in various fields, from manufacturing to healthcare to customer service.
    
    2. AI will become more integrated with human decision-making: With AI systems becoming more capable of understanding and interpreting natural language, it is likely that they will become more integrated with human decision-making processes. This could lead to more efficient and effective decision-making, but it also raises concerns about the potential for AI to take over certain roles and decision-making processes.
    
    3. AI will become more reliant on data


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

     am

     a

     [

    job

     title

    ].

     I

     have

     been

     in

     the

     [

    industry

    ]

     industry

     for

     [

    number

    ]

     years

    .

     My

     passion

     for

     [

    industry

    ]

     is

     [

    reason

     for

     interest

    ].

     I

     am

     always

     looking

     for

     [

    what

     to

     achieve

     in

     my

     career

    ],

     and

     my

     [

    career

     goal

    ]

     is

     [

    description

     of

     what

     I

     want

     to

     achieve

    ].

     I

     am

     always

     eager

     to

     learn

     new

     things

     and

     try

     new

     things

    ,

     and

     I

     am

     determined

     to

     prove

     myself

     to

     myself

     and

     others

    .

     My

     [

    personal

     characteristic

    ]

     is

     [

    describe

     a

     characteristic

    ],

     and

     I

     am

     proud

     of

     my

     [

    reason

     for

     the

     characteristic

    ].

     I

     am

     always

     respectful

    ,

     helpful

    ,

     and

     consider

    ate

     to

     others

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Does

     this

     next

     sentence

     follow

    ,

     given

     the

     preceding

     text

    ?
    


    Paris

     is

     the

     only

     capital

     city

     of

     France

    .

     
    


    Select

     from

    :

     (

    i

    )

     yes

     (

    ii

    )

     no

    
    


    (ii

    )

     No

    


    This

     sentence

     does

     not

     follow

     from

     the

     given

     text

    .

     The

     initial

     statement

     mentions

     that

     Paris

     is

     the

     capital

     of

     France

    ,

     but

     it

     does

     not

     provide

     information

     about

     whether

     it

     is

     the

     only

     capital

     city

     of

     France

    .

     The

     text

     only

     states

     that

     Paris

     is

     the

     capital

    ,

     but

     it

     does

     not

     indicate

     that

     it

     is

     the

     only

     capital

     city

    .

     Therefore

    ,

     the

     correct

     answer

     is

     (

    ii

    )

     No

    .

     The

     information

     provided

     does

     not

     confirm

     or

     deny

     the

     existence

     of

     other

     capital

     cities

     in

     France

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     one

     of

     rapid

     advancement

     and

     innovation

    ,

     with

     several

     potential

     trends

     shaping

     the

     technology

    's

     trajectory

    .
    


    1

    .

     Increased

     Integration

     with

     Human

     Inter

    actions

    :

     AI

     is

     already

     becoming

     more

     integrated

     with

     human

     interactions

    ,

     from

     personal

     assistants

     like

     Siri

     and

     Alexa

     to

     virtual

     assistants

     like

     IBM

    's

     Watson

    .

     It

    's

     expected

     that

     this

     integration

     will

     continue

    ,

     with

     more

     people

     relying

     on

     AI

    -driven

     assistants

     for

     tasks

     like

     scheduling

     appointments

    ,

     managing

     finances

    ,

     and

     even

     tele

    con

    fer

    encing

    .
    


    2

    .

     Enhanced

     Data

     Processing

     and

     Analytics

    :

     AI

     algorithms

     will

     continue

     to

     become

     more

     sophisticated

     and

     powerful

    ,

     enabling

     them

     to

     process

     and

     analyze

     large

     amounts

     of

     data

     more

     efficiently

    .

     This

     will

     lead

     to

     improved

     prediction

     and

     decision

    -making

    



```python
llm.shutdown()
```
