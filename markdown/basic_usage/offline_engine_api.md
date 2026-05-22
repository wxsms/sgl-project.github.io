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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 23.01it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 32.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s] Capturing num tokens (num_tokens=896 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]Capturing num tokens (num_tokens=832 avail_mem=72.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=640 avail_mem=72.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=576 avail_mem=72.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=480 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=320 avail_mem=72.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=288 avail_mem=71.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=224 avail_mem=71.98 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.97it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.65it/s] Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.65it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.16it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.16it/s]Capturing num tokens (num_tokens=28 avail_mem=71.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.16it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.16it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.16it/s]Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.16it/s]Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.15it/s] Capturing num tokens (num_tokens=4 avail_mem=71.93 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 42.15it/s]


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
    Generated text:  Amir and I am a high school senior. I am from a single family with two parents, both of whom are mothers. I have always been a very small child. I do not have many friends, but I do have a good teacher. She is very kind and always has a great sense of humor. She does everything I ask of her and makes me feel like a special person. I have a lot of energy but also a very difficult time during times of stress. I am also very athletic, and my mom and dad both have a great interest in basketball and it makes my life very enjoyable. I have an extremely strict and
    ===============================
    Prompt: The president of the United States is
    Generated text:  42 years older than the president of Mexico.  In two years, the sum of their ages will be 71.  At the same time, the president of the United States will be twice as old as the president of Mexico.  How old will the president of the United States be in two years?
    Let's define the variables for the current ages of the presidents. Let the current age of the president of the United States be \( U \) and the current age of the president of Mexico be \( M \).
    
    From the problem, we know two key pieces of information:
    1. The president of the United States
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the capital of France.
    
    What are the first letters of each word in this sentence? The first letters of each word in the sentence "The capital of France is Paris. " are "T, H, A, R, C, A, P, F, R, I, A, G, H."
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s the future of healthcare.
    
    For the most part, AI has been a phenomenon of recent years, appearing in the news and in the headlines of innovation and advancement. It’s been quietly, but quietly, making waves in various fields.
    
    But the field that’s made the most significant breakthrough to date has been in healthcare. As healthcare has become more sophisticated, its algorithms have been augmented, and AI has become a major force in treating medical problems.
    
    What’s the future of AI in healthcare? In a recent blog post, an AI researcher discussed what we can expect from the future of AI in healthcare. The post contains some


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill/Ability] who has been [Number of Years] years in the industry. I'm passionate about [What I Love to Do] and I'm always looking for new challenges and opportunities to grow and succeed. I'm a [Favorite Hobby] and I enjoy [What I Like to Do]. I'm [What I Do for a Living] and I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. I'm [What I Believe in] and I believe in [What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to ancient times. Paris is a popular tourist destination, attracting millions of visitors each year. The city is known for its diverse cuisine, including French cuisine, and its vibrant nightlife. It is a major hub for international business and diplomacy, and is home to many of the world's most famous museums and landmarks. Paris is a city of contrasts, with its historic architecture and modern art, and its rich cultural heritage. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries. This will lead to increased efficiency, productivity, and cost savings for businesses and individuals alike.
    
    2. AI-powered healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field. AI-powered healthcare systems will be able to analyze large amounts of medical data, identify patterns and trends, and provide
    


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
    Generated text:  [insert your name here], and I am a [insert profession here]. I am [insert your age here], and I am [insert your height here], and I am [insert your weight here]. I am [insert your personality traits here]. I love to [insert your hobby here], [insert any other hobbies you have here]. I am a [insert your occupation here], and I have been working in this field for [insert the number of years here]. I have always been passionate about [insert one or two things that you are particularly passionate about here]. I enjoy [insert one or two things you enjoy doing here]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is also known as the City of Light, and is considered one of the world’s most livable cities. The city is located on the Île de France, the largest island in the Mediterranean Sea. It is situated on the Seine River, which flows through the city, and has numerous landmarks, including the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Sacré-Cœur Basilica. Paris is a global city with a diverse population of over 2 million people, and it is the fifth-largest city in the world by population. The city is also home to the Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  unpredictable and can take many different forms. Some possible trends that are currently being explored and discussed include:
    
    1. Autonomous robots: Self-driving cars, drones, and other unmanned vehicles will become more common in the future. These autonomous robots will be able to navigate complex environments, make decisions, and respond to human commands.
    
    2. Artificial intelligence in healthcare: AI will be used to improve the accuracy and speed of diagnosis, treatment, and patient care. AI systems will be able to analyze large amounts of medical data and provide valuable insights that can help doctors make more informed decisions.
    
    3. Facial recognition: AI will be used to enhance security and privacy


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

    name

    ]

     and

     I

    'm

     a

     [

    background

     information

     about

     your

     character

    ].

     I

    'm

     [

    age

    ]

     years

     old

     and

     I

     have

     [

    occupation

     or

     profession

    ]

     experience

    .

     I

    've

     always

     loved

     to

     learn

     new

     things

     and

     have

     always

     tried

     to

     improve

     myself

    .

     I

    'm

     very

     respectful

     and

     have

     a

     friendly

     demeanor

    ,

     and

     I

     enjoy

     meeting

     new

     people

     and

     making

     new

     friends

    .

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     myself

     and

     keep

     learning

    .

     Thank

     you

     for

     asking

     about

     me

    !

     How

     can

     I

     help

     you

    ?

     [

    name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     and

     the

     seat

     of

     government

     of

     the

     nation

    .
    


    Does

     the

     fact

     that

     it

     is

     Paris

     mean

     that

     it

     is

     the

     only

     capital

    ?

     The

     answer

     is

     No

    .

     The

     capital

     of

     France

     is

     also

     Lyon

    ,

     another

     city

     located

     in

     the

     country

    .

     The

     information

     provided

     in

     the

     statement

     is

     correct

    ;

     Paris

     is

     the

     largest

     city

     and

     the

     capital

     of

     France

    .

     However

    ,

     it

    's

     important

     to

     note

     that

     Lyon

     is

     indeed

     the

     second

     largest

     city

     and

     the

     capital

     of

     France

    .

     Lyon

     is

     known

     for

     its

     historic

     architecture

    ,

     French

     cuisine

    ,

     and

     vibrant

     cultural

     scene

    ,

     making

     it

     a

     significant

     city

     in

     French

     politics

     and

     economy

    .

     Lyon

     is

     part

     of

     the

     Paris

     metropolitan

     area

    ,

     which

     includes

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     range

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     advances

     in

     natural

     language

     processing

    ,

     advances

     in

     machine

     learning

     algorithms

    ,

     and

     the

     emergence

     of

     new

     technologies

     and

     industries

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

     integration

     with

     humans

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     may

     become

     more

     integrated

     with

     humans

    ,

     providing

     assistance

     and

     insight

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     education

    .
    


    2

    .

     AI

     for

     personal

    ization

    :

     AI

     algorithms

     will

     be

     able

     to

     analyze

     large

     amounts

     of

     data

     to

     provide

     personalized

     recommendations

     and

     solutions

     to

     users

    .
    


    3

    .

     AI

     for

     automation

    :

     AI

     will

     be

     used

     for

     a

     wide

     range

     of

     tasks

    ,

     from

    



```python
llm.shutdown()
```
