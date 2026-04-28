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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.90it/s]


    2026-04-28 03:52:10,151 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 03:52:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.80it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.35 GB):   3%|▎         | 2/58 [00:00<00:03, 17.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.35 GB):   3%|▎         | 2/58 [00:00<00:03, 17.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.35 GB):   3%|▎         | 2/58 [00:00<00:03, 17.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.35 GB):   3%|▎         | 2/58 [00:00<00:03, 17.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=119.35 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.33 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.33 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.33 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=119.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=119.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=119.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.23it/s]Capturing num tokens (num_tokens=1024 avail_mem=119.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.23it/s]

    Capturing num tokens (num_tokens=960 avail_mem=119.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.23it/s] Capturing num tokens (num_tokens=896 avail_mem=119.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.23it/s]Capturing num tokens (num_tokens=832 avail_mem=119.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.23it/s]Capturing num tokens (num_tokens=832 avail_mem=119.28 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=768 avail_mem=119.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=704 avail_mem=119.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=640 avail_mem=119.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=576 avail_mem=119.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=512 avail_mem=119.25 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=512 avail_mem=119.25 GB):  50%|█████     | 29/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=480 avail_mem=119.27 GB):  50%|█████     | 29/58 [00:00<00:00, 40.46it/s]

    Capturing num tokens (num_tokens=448 avail_mem=119.27 GB):  50%|█████     | 29/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=416 avail_mem=119.27 GB):  50%|█████     | 29/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=384 avail_mem=119.26 GB):  50%|█████     | 29/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=352 avail_mem=119.26 GB):  50%|█████     | 29/58 [00:00<00:00, 40.46it/s]Capturing num tokens (num_tokens=352 avail_mem=119.26 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=320 avail_mem=119.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=288 avail_mem=119.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=256 avail_mem=119.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=240 avail_mem=119.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=224 avail_mem=119.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.63it/s]

    Capturing num tokens (num_tokens=224 avail_mem=119.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=208 avail_mem=119.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=192 avail_mem=119.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=176 avail_mem=119.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=160 avail_mem=119.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=144 avail_mem=119.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=144 avail_mem=119.23 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=128 avail_mem=119.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=112 avail_mem=119.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=96 avail_mem=119.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.26it/s] Capturing num tokens (num_tokens=80 avail_mem=119.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.26it/s]

    Capturing num tokens (num_tokens=64 avail_mem=119.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=64 avail_mem=119.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=48 avail_mem=119.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=32 avail_mem=119.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=28 avail_mem=119.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=24 avail_mem=119.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=20 avail_mem=119.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=20 avail_mem=119.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=16 avail_mem=119.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=12 avail_mem=119.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=8 avail_mem=119.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.44it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=119.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=4 avail_mem=119.18 GB): 100%|██████████| 58/58 [00:01<00:00, 39.19it/s]


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
    Generated text:  Mei and I am from England. I am so excited to have found this great place to learn about the life of a wildlife photographer. To begin, I would like to ask you some basic questions about your life. Where did you grow up? What's your favorite sport?
    Hello and welcome to my website. Thank you for visiting my website. My name is Mei and I'm from England. I was born in Huddersfield. My parents work in manufacturing and teaching. So they are always looking out for me and want me to do well in school.
    I'm also an English citizen and so my parents have been proud of me
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. In fact, the president is the leader of the government of the United States. Most people in the United States support the president. Why? Because they believe that the president is the head of the government, the one who makes decisions about all the important things in the country. In the past, the president was elected by the people in a popular election. But now, most people in the United States think that it is more important to have a person who has a lot of experience in the government. People tend to vote for a person who they think can help solve the problems in the country. In this way,
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    
    A: Paris  
    B: Paris (already mentioned in the options)  
    C: London  
    D: Rome
    To determine the capital of France, let's review the options given:
    
    A: Paris  
    B: Paris (already mentioned in the options)  
    C: London  
    D: Rome
    
    Since the question asks for the capital of France, the correct answer should be one of these options. However, the options do not include any capital cities of France. Therefore, the capital of France is not listed among the options.
    
    Given the options, the correct choice is:
    
    \boxed{D} (The capital of France is
    ===============================
    Prompt: The future of AI is
    Generated text:  far from certain, but one thing is certain: with the rapid advancement of artificial intelligence, the education system will face a significant change. The technology is projected to be transformed in the near future, transforming the education system in a profound way. The potential of artificial intelligence in the education system has been hailed as a significant development. As a result, the education system is changing from traditional to a more personalized and efficient educational system. It is a new era of education where technology is playing a significant role in the way we learn.
    One of the aspects of the education system in the future is the use of AI in the classroom. AI has the


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Quarter. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination, with many famous landmarks and attractions. Paris is known for its cuisine, fashion, and art, and is a popular destination for tourists and locals alike. It is a city that has a unique blend of old-world charm and modernity,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for workers in fields like manufacturing and logistics.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be an increased need for measures to protect user data and prevent cyber attacks. This could lead to new regulations and standards
    


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
    Generated text:  [Character Name] and I'm a [Brief Description of Your Profession or Character]. I'm a [Specific Skill/Ability] that has been [Number of Years] years of experience in this field. I'm a [Personality Type] and have always [Reason for Choosing This Path] and have always been [Reason for Choosing This Path]. I'm [Favorite Subject/Major/Interest] and I'm always looking for [What I'm Looking for in My Next Adventure]. I'm [Personality Type] and I'm always [Reason for Your Personality Type]. I'm a [Personality Type] and I'm always [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historical and cultural city renowned for its stunning architecture, world-renowned museums, and diverse array of cuisine.
    You are to summarize the aforementioned text by selecting only upon the essential components. Avoid adding any information that does not affect the summary.
    
    Paris, the capital of France, is a historical and cultural city renowned for its stunning architecture, world-renowned museums, and diverse array of cuisine. Its historic centers, known as the Quartiers, feature picturesque canals, quaint streets, and well-preserved 16th- and 17th-century houses that reflect the city's rich history. The city is also home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly changing, with many possibilities and trends emerging. Some potential trends in AI include:
    
    1. Personalization: AI will be able to learn from user data and personalization will become more advanced. AI will be able to provide personalized recommendations, products, and services based on user preferences and behavior.
    
    2. Artificial intelligence ethics: AI will need to be developed in a way that is ethically sound, considering the potential negative consequences of AI technology.
    
    3. Autonomous robots: Autonomous robots will become more advanced, capable of performing complex tasks and making decisions based on data.
    
    4. Improved machine learning: Machine learning will be able to improve its performance


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

    ]

     and

     I

     am

     a

     [

    insert

     your

     profession

     or

     field

     of

     work

    ]

     with

     [

    insert

     your

     degree

     or

     educational

     background

    ].

     I

     am

     passionate

     about

     [

    insert

     your

     main

     interest

     or

     hobby

    ].

     I

     enjoy

     [

    insert

     how

     you

     enjoy

     your

     hobby

    ],

     and

     I

     strive

     to

     be

     a

     [

    insert

     what

     you

     hope

     to

     achieve

     in

     the

     coming

     year

    ]

     by

     [

    insert

     your

     profession

     in

     the

     next

     few

     years

    ].

     I

     am

     always

     eager

     to

     learn

     and

     stay

     up

    -to

    -date

     on

     the

     latest

     trends

     and

     developments

     in

     my

     field

    .

     What

     is

     your

     experience

     with

     [

    insert

     your

     field

     of

     work

     or

     education

    ],

     and

     what

     exc

    ites

     you

     the

     most

     about

     it

    ?


    [

    Your

     Name

    ]

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     Lou

    v

    ain

    ,

     and

     it

     is

     the

     largest

     city

     and

     the

     largest

     metropolitan

     area

     in

     Europe

    .

     Paris

     is

     known

     for

     its

     stunning

     architecture

    ,

     music

    ,

     and

     festivals

    ,

     and

     it

     is

     an

     important

     cultural

     and

     economic

     center

    .

     It

     is

     located

     on

     the

     River

     Se

    ine

     and

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

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     has

     a

     rich

     and

     diverse

     history

     dating

     back

     to

     ancient

     times

    ,

     making

     it

     a

     fascinating

     city

     to

     explore

    .

     The

     city

     is

     also

     a

     major

     transportation

     hub

    ,

     with

     many

     famous

     landmarks

     located

     in

     the

     city

     center

    .

     Paris

     is

     home

     to

     the

     French

     Parliament

     and

     many

     museums

     and

     cultural

     institutions

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     promising

    ,

     with

     numerous

     trends

     and

     innovations

     poised

     to

     shape

     the

     technology

     and

     its

     applications

     in

     the

     years

     ahead

    .

     Here

     are

     some

     of

     the

     most

     promising

     trends

    :
    


    1

    .

     **

    Deep

     Learning

     and

     Neural

     Networks

    **:

     Deep

     learning

     and

     neural

     networks

     are

     becoming

     more

     and

     more

     advanced

    ,

     with

     models

     that

     can

     perform

     tasks

     previously

     considered

     impossible

    .

     This

     includes

     machine

     learning

     algorithms

     that

     can

     recognize

     patterns

     in

     complex

     data

    ,

     understand

     language

    ,

     and

     even

     play

     games

     like

     Go

     and

     chess

    .
    


    2

    .

     **

    Natural

     Language

     Processing

     (

    N

    LP

    )**

    :

     N

    LP

     is

     transforming

     how

     we

     interact

     with

     computers

    .

     AI

    -driven

     chat

    bots

     and

     virtual

     assistants

     like

     Siri

    ,

     Alexa

    ,

     and

     Google

     Assistant

     are

     becoming

     more

     sophisticated

    ,

    



```python
llm.shutdown()
```
