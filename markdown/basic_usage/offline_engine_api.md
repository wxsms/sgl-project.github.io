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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.06 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.06 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.06 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.06 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.06 GB):   9%|▊         | 5/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.05 GB):   9%|▊         | 5/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.04 GB):   9%|▊         | 5/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.04 GB):   9%|▊         | 5/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.04 GB):   9%|▊         | 5/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=56.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.02 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.01 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.01 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.01 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.00 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.00 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.60it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.98 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.60it/s]

    Capturing num tokens (num_tokens=960 avail_mem=55.99 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.60it/s] Capturing num tokens (num_tokens=896 avail_mem=55.99 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.60it/s]Capturing num tokens (num_tokens=832 avail_mem=55.99 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.60it/s]Capturing num tokens (num_tokens=832 avail_mem=55.99 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=768 avail_mem=55.98 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=704 avail_mem=55.98 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=640 avail_mem=55.98 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=576 avail_mem=55.98 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=512 avail_mem=55.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=512 avail_mem=55.96 GB):  50%|█████     | 29/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=480 avail_mem=55.98 GB):  50%|█████     | 29/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=448 avail_mem=55.98 GB):  50%|█████     | 29/58 [00:00<00:00, 42.21it/s]

    Capturing num tokens (num_tokens=416 avail_mem=55.98 GB):  50%|█████     | 29/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=384 avail_mem=55.97 GB):  50%|█████     | 29/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=352 avail_mem=55.97 GB):  50%|█████     | 29/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=352 avail_mem=55.97 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=320 avail_mem=55.96 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=288 avail_mem=55.96 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=256 avail_mem=55.96 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=240 avail_mem=55.95 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=224 avail_mem=55.95 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.78it/s]Capturing num tokens (num_tokens=224 avail_mem=55.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=208 avail_mem=55.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=192 avail_mem=55.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.16it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=160 avail_mem=55.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=144 avail_mem=55.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=144 avail_mem=55.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.56it/s]Capturing num tokens (num_tokens=128 avail_mem=55.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.56it/s]Capturing num tokens (num_tokens=112 avail_mem=55.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.56it/s]Capturing num tokens (num_tokens=96 avail_mem=55.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.56it/s] Capturing num tokens (num_tokens=80 avail_mem=55.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.56it/s]Capturing num tokens (num_tokens=64 avail_mem=55.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.56it/s]Capturing num tokens (num_tokens=64 avail_mem=55.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=48 avail_mem=55.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=32 avail_mem=55.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.16it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=24 avail_mem=55.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=20 avail_mem=55.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=20 avail_mem=55.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=16 avail_mem=55.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=12 avail_mem=55.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=8 avail_mem=55.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.23it/s] Capturing num tokens (num_tokens=4 avail_mem=55.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=4 avail_mem=55.89 GB): 100%|██████████| 58/58 [00:01<00:00, 40.23it/s]


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
    Generated text:  Faye and I'm a Marketing Manager at Etsy. My job is to help our team create, sell, and distribute amazing handmade crafts and gifts. From the day we first hired me as a Marketing Manager, I've had the opportunity to work with many great people to grow my team and accomplish our goals.
    My process for creating and selling my craft products has always been to promote and drive sales through social media, email, and direct mail. My team began with simple strategies, but now it's a dynamic, innovative approach to crafting an incredible brand.
    I've learned a lot about the online world and how to use it to sell,
    ===============================
    Prompt: The president of the United States is
    Generated text:  to be elected in the following manner: There is a 30% chance of a Republican candidate winning, a 50% chance of a Democrat winning, and a 20% chance of a third-party candidate winning. What is the probability that the president is won by a Democrat, given that the winner is Republican? To determine the probability that the president is won by a Democrat given that the winner is Republican, we need to use the concept of conditional probability. The conditional probability \( P(A|B) \) is the probability of event \( A \) occurring given that event \( B \) has occurred. It
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. Moscow
    D. Tokyo
    Answer: A
    
    The first international airport in China is located in ____.
    A. Hangzhou
    B. Nanjing
    C. Shanghai
    D. Beijing
    Answer: C
    
    The most fundamental view held by the proletariat is ____
    A. Capitalism is the root of all evil
    B. Capitalism is the primary stage of human society
    C. Capitalism is an evil state
    D. Capitalism is beneficial and correct
    Answer: A
    
    Male, 45 years old. Admitted to the hospital due to acute exacerb
    ===============================
    Prompt: The future of AI is
    Generated text:  that it will not be a single, unified system but will be a complex network of many diverse AI technologies that are interconnected and can work together to solve complex problems. As we advance in the field of AI, we will also see the rise of new AI technologies that will make it more difficult for existing systems to be defeated.
    
    The impact of AI on society will be enormous, as it will enable new forms of productivity and innovation that were not previously possible. It will also lead to more efficient and effective ways of decision-making and problem-solving, which could improve the quality of life for everyone.
    
    However, there are also concerns about the potential misuse


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your personality or skills]. And what can you tell me about your company? I'm excited to learn more about your company and how it fits into the larger picture of [insert a company name]. What can you tell me about your work? I'm excited to learn more about your work and how it contributes to the success of [insert a company name]. And what can you tell me about your hobbies? I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is also home to many world-renowned museums, including the Musée d'Orsay and the Musée d'Orsay. Paris is a popular tourist destination and a major financial center in Europe. It is also known for its fashion industry, with many famous designers and boutiques located in the city. The city is known for its cuisine, with many famous restaurants and cafes serving traditional French dishes. Overall, Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence. This could lead to more sophisticated forms of AI that can learn from and adapt to human behavior and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Greater use of AI in healthcare: AI is
    


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
    Generated text:  [name] and I'm a/an [job title] who has been working at [company name] for [number] years. I've always been passionate about [career objective]. I'm always looking for new challenges and opportunities to make a difference. I'm confident in my ability to adapt to new situations and work under pressure. I have a strong work ethic and a great sense of humor. I'm friendly, outgoing, and always willing to help others. I love to learn and always want to grow and improve myself. I'm a team player and enjoy working with others to achieve common goals. I'm always up for a challenge
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Conjugating the verb "to be" in French means to be in possession of, to be in charge, to be present. 
    
    The verb "to be" is conjugated in the present tense for the first person singular, the second person singular, and the third person singular.
    
    The verb "to be" is also used to show possession and to show identity. 
    
    The verb "to be" is conjugated in French for the past tense: "was," "were."
    
    The verb "to be" is used to express states of mind, which include both certainty and uncertainty.
    
    The verb "to be"
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends:
    
    1. Increased focus on ethics and privacy: As AI systems become more complex and capable of interacting with humans, they will continue to raise ethical and privacy concerns. Governments and industry leaders will continue to focus on developing systems that prioritize ethical considerations and data privacy, while also ensuring transparency and accountability.
    
    2. Advancements in machine learning: Machine learning will continue to become more advanced, enabling AI systems to learn from data and adapt to new situations. This will require significant investments in research and development, as well as the development of new algorithms and techniques.
    
    3. Integration with other technologies: AI systems will


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

    'm

     a

     [

    job

     title

     or

     hobby

    ]

     who

     has

     always

     loved

     [

    something

     that

     makes

     me

     happy

     or

     an

     aspect

     of

     my

     identity

    ].

     I

     started

     my

     career

     in

     [

    industry

     or

     field

    ],

     and

     I

    've

     always

     been

     passionate

     about

     [

    a

     particular

     cause

     or

     issue

    ].

     I

    'm

     determined

     to

     continue

     growing

     as

     a

     [

    role

     or

     skill

    ],

     and

     I

     strive

     to

     make

     a

     positive

     impact

     on

     the

     world

    .

     So

    ,

     what

    's

     your

     name

    ,

     and

     what

     do

     you

     do

    ?

     [

    Add

     your

     name

     and

     hobbies

     and

     any

     other

     relevant

     information

     if

     you

     have

     it

    ].


    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    job

     title

     or

     hobby

    ]

     who

     has

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

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

     Notre

    -D

    ame

     Cathedral

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     dynamic

     and

     uncertain

    ,

     and

     it

     is

     difficult

     to

     predict

     exactly

     what

     will

     happen

     in

     the

     coming

     years

    .

     However

    ,

     there

     are

     some

     general

     trends

     that

     are

     likely

     to

     continue

     in

     the

     AI

     industry

    :
    


    1

    .

     Increased

     sophistication

     of

     AI

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     so

     too

     will

     its

     capabilities

    .

     AI

     systems

     will

     become

     better

     at

     recognizing

     patterns

     in

     large

     datasets

    ,

     learning

     from

     their

     mistakes

    ,

     and

     adapting

     to

     new

     situations

    .

     This

     will

     require

     more

     sophisticated

     algorithms

     and

     better

     training

     data

     to

     achieve

     these

     goals

    .
    


    2

    .

     Deep

     learning

    :

     Deep

     learning

     is

     a

     type

     of

     AI

     that

     uses

     neural

     networks

     with

     many

     layers

     to

     learn

     complex

     patterns

     in

     data

    .

     As

     neural

     networks

     become

    



```python
llm.shutdown()
```
