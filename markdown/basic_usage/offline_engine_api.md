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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.35it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.53it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.53it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=60.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.55it/s]Capturing num tokens (num_tokens=960 avail_mem=60.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.55it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=60.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.55it/s]Capturing num tokens (num_tokens=832 avail_mem=60.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.55it/s]Capturing num tokens (num_tokens=832 avail_mem=60.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=768 avail_mem=60.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=704 avail_mem=60.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=640 avail_mem=60.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=576 avail_mem=60.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=512 avail_mem=60.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=512 avail_mem=60.32 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=480 avail_mem=60.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=448 avail_mem=60.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=416 avail_mem=60.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]

    Capturing num tokens (num_tokens=384 avail_mem=60.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=352 avail_mem=60.32 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=352 avail_mem=60.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=320 avail_mem=60.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=288 avail_mem=60.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=256 avail_mem=60.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=240 avail_mem=60.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=224 avail_mem=60.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=224 avail_mem=60.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=208 avail_mem=60.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=192 avail_mem=60.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=176 avail_mem=60.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.70it/s]

    Capturing num tokens (num_tokens=160 avail_mem=60.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=144 avail_mem=60.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=144 avail_mem=60.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=128 avail_mem=60.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=112 avail_mem=60.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=96 avail_mem=60.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.39it/s] Capturing num tokens (num_tokens=80 avail_mem=60.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=64 avail_mem=60.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=64 avail_mem=60.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=48 avail_mem=60.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=32 avail_mem=60.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=28 avail_mem=60.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.22it/s]

    Capturing num tokens (num_tokens=24 avail_mem=60.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=20 avail_mem=60.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=20 avail_mem=60.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=16 avail_mem=60.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=12 avail_mem=60.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=8 avail_mem=60.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.62it/s] Capturing num tokens (num_tokens=4 avail_mem=60.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=4 avail_mem=60.24 GB): 100%|██████████| 58/58 [00:01<00:00, 40.49it/s]


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
    Generated text:  Livia. This is a natural language processing (NLP) problem. This problem is to construct a sentiment analysis model for an essay. I have 16 features that I would like to add to the model. How do I write a code to do so? I am a beginner in NLP and machine learning. Please let me know on how to write the code to add the features to the model. If you can provide some examples, it would be very helpful. To start, I want to add the following features:
    
    1. Punctuation
    2. Emoji
    3. Emoji Use
    4. Word length
    5.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a dictator who makes decisions for the entire country. The president's role is to govern the country, not to represent the country. Do you think it would be better to have an elected president or a dictator? As an AI language model, I cannot make definitive statements about whether it would be better to have an elected president or a dictator, as this decision depends on various factors such as the specific circumstances and political context of the country.
    
    However, it's worth noting that both the presidency and the role of the president as a representative of the country can have various implications and challenges. While the role of a president is to govern the country,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a city of a thousand years of history. When you take a walk through Paris, you can see the memory of all the time of French history. Here, we will tell you the most famous and important buildings in Paris. From the Notre Dame Cathedral to the Louvre Museum, every building has a history, a story, and a legend. If you have visited Paris, you are sure to be amazed by its beauty and history. We hope you can enjoy the beauty of Paris and make new friends there. The best title of the passage is __________
    A. Historical Memorabilia in Paris
    B. The Most
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s coming through the lens of gaming. With the introduction of AI into the gaming industry, it’s become more common to see an AI capable of solving complex problems in real-time, and even changing the game itself. When a game developer brings an AI to life, they must also think about the ethical implications of their AI, and ensure that it is not using human-like behaviors that might lead to negative consequences for players. This is why we are examining the ethical implications of AI in gaming in this post.
    
    AI in Gaming: The Game-Changer
    
    The introduction of AI into the gaming industry is a significant shift for the


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Character] who has always been [What motivates you to be who you are]. I'm passionate about [What you enjoy doing], and I'm always looking for ways to [What you hope to achieve]. I'm always looking for ways to [What you hope to achieve], and I'm always eager to [What you hope to achieve]. I'm always looking for ways to [What you hope to achieve], and I'm always eager to [What you hope to achieve]. I'm always looking for ways to [What you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is the largest city in France and the second-largest city in the European Union. Paris is also known for its fashion industry, art scene, and its role in the French Revolution and the French Revolution. It is a popular tourist destination and a major economic center in Europe. The city is home to many famous landmarks and museums, including the Louvre and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of culture, art, and history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in manufacturing, transportation, and other industries. As machines become more capable of performing tasks that were previously done by humans, there will be an increase in automation, which will lead to more efficient and cost-effective operations.
    
    2. AI ethics: As AI becomes more advanced, there will be a need for ethical guidelines to govern its use. This will include issues such as bias, privacy, and accountability. AI developers will need
    


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
    Generated text:  [insert name]. I am a [insert profession or area of expertise] with a deep understanding of [insert what the character does or knows about]. My goal is to [insert specific goal or objective]. I am always looking to learn and grow in my field and to stay up to date with the latest trends and technologies. I am a [insert any unique traits or qualities] that make me a valuable member of our team. What's your name, and what do you do or know about? Here's to a great year ahead! [Insert name] is a [insert profession or area of expertise] with a deep understanding of [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historical and cultural center with its own unique blend of Gothic architecture, art, and cuisine. The city is known for its rich history and is a popular tourist destination. Paris is also home to the Eiffel Tower, one of the most recognizable landmarks in the world, and offers a diverse range of cultural attractions and dining options. The city's climate is Mediterranean, with hot summers and mild winters, making it a popular destination for outdoor activities and relaxation. Overall, Paris is a vibrant and dynamic city that continues to be a global cultural and economic center. Paris, France. The capital of France is Paris, a historical and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly complex and uncertain, but some of the key trends that are likely to shape it include:
    
    1. Integration with human beings: AI will become more integrated with humans, enabling them to work alongside AI in various fields like healthcare, education, and manufacturing. This will lead to more efficient and effective use of resources.
    
    2. Personalization: AI will become more personalized, allowing machines to learn from the data they receive and make better-informed decisions. This will lead to more targeted and effective marketing and customer service.
    
    3. Ethics and accountability: As AI becomes more prevalent, there will be a growing emphasis on ethical considerations and accountability. This


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

     [

    Age

    ]

     years

     old

    .

     I

     am

     a

     [

    job

    ]

     with

     [

    company

    ].

     I

     have

     always

     been

     [

    what

     your

     pet

     or

     pet

    -d

    og

    -like

     character

     does

    ]

     for

     [

    reason

    s

    ]

     and

     have

     been

     [

    your

     pet

    's

     pet

     name

     for

     a

     long

     time

    ].

     I

     like

     to

     [

    what

     you

     like

     to

     do

     for

     fun

    ].

     I

    'm

     currently

     [

    present

    ing

     a

     recent

     image

     of

     yourself

     in

     front

     of

     a

     simple

    ,

     neutral

     image

     board

    ,

     such

     as

     a

     kitchen

     counter

     or

     coffee

     table

     with

     just

     a

     few

     neutral

     colors

    ].

     Would

     you

     like

     to

     meet

     me

    ?

     [

    Would

     you

     like

     to

     meet

     me

     and

     have

     a

     brief

     chat

    ?]

     [

    Name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     and

     vibrant

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

    .

     It

     is

     known

     for

     its

     grand

     bou

    lev

    ards

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     its

     iconic

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     has

     a

     diverse

     population

     and

     is

     home

     to

     many

     cultural

     institutions

     and

     events

     throughout

     the

     year

    ,

     including

     the

     World

     Cup

     and

     the

     Summer

     Olympics

    .

     The

     French

     capital

     is

     also

     known

     for

     its

     cuisine

    ,

     art

    ,

     and

     nightlife

    .

     Paris

     is

     a

     city

     that

     has

     a

     strong

     sense

     of

     community

     and

     offers

     visitors

     a

     unique

     and

     unforgettable

     experience

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     changing

     and

     it

    's

     difficult

     to

     predict

     exactly

     what

     will

     happen

    .

     However

    ,

     some

     possible

     trends

     that

     are

     currently

     being

     explored

     and

     explored

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     with

     other

     technologies

    :

     The

     integration

     of

     AI

     with

     other

     technologies

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

     is

     likely

     to

     become

     more

     common

     in

     the

     coming

     years

    .

     This

     will

     allow

     AI

     to

     be

     more

     effective

     in

     a

     variety

     of

     applications

    ,

     from

     healthcare

     to

     transportation

     to

     finance

    .
    


    2

    .

     AI

     in

     more

     domains

    :

     AI

     is

     already

     being

     used

     in

     many

     different

     domains

    ,

     such

     as

     autonomous

     vehicles

    ,

     smart

     homes

    ,

     and

     healthcare

    .

     It

    's

     likely

     that

     we

     will

     see

     even

     more

     AI

     applications

     in

    



```python
llm.shutdown()
```
