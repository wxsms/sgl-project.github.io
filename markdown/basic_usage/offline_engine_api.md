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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.60it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.58it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.93 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.92 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.87 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.05 GB):   9%|▊         | 5/58 [00:00<00:02, 20.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.88 GB):   9%|▊         | 5/58 [00:00<00:02, 20.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.87 GB):   9%|▊         | 5/58 [00:00<00:02, 20.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.86 GB):   9%|▊         | 5/58 [00:00<00:02, 20.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.86 GB):   9%|▊         | 5/58 [00:00<00:02, 20.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.85 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.85 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.85 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.85 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.85 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.84 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.84 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.83 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.83 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.83 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.83 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.83 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.81 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]

    Capturing num tokens (num_tokens=960 avail_mem=57.82 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s] Capturing num tokens (num_tokens=896 avail_mem=57.82 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]Capturing num tokens (num_tokens=832 avail_mem=57.82 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]Capturing num tokens (num_tokens=832 avail_mem=57.82 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=768 avail_mem=57.81 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=704 avail_mem=57.81 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=640 avail_mem=57.81 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=576 avail_mem=57.81 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=512 avail_mem=57.79 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=512 avail_mem=57.79 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=480 avail_mem=57.81 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=448 avail_mem=57.80 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]

    Capturing num tokens (num_tokens=416 avail_mem=57.80 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=384 avail_mem=57.80 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=352 avail_mem=57.79 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=352 avail_mem=57.79 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=320 avail_mem=57.79 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=288 avail_mem=57.79 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=256 avail_mem=56.97 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=240 avail_mem=56.96 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=224 avail_mem=56.96 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=224 avail_mem=56.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=208 avail_mem=56.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=192 avail_mem=56.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]

    Capturing num tokens (num_tokens=176 avail_mem=56.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=160 avail_mem=56.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=144 avail_mem=56.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=144 avail_mem=56.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=128 avail_mem=56.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=112 avail_mem=56.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=96 avail_mem=56.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.43it/s] Capturing num tokens (num_tokens=80 avail_mem=56.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=64 avail_mem=56.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=64 avail_mem=56.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.39it/s]Capturing num tokens (num_tokens=48 avail_mem=56.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.39it/s]Capturing num tokens (num_tokens=32 avail_mem=56.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.39it/s]

    Capturing num tokens (num_tokens=28 avail_mem=56.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.39it/s]Capturing num tokens (num_tokens=24 avail_mem=56.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.39it/s]Capturing num tokens (num_tokens=20 avail_mem=56.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.39it/s]Capturing num tokens (num_tokens=20 avail_mem=56.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=16 avail_mem=56.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=12 avail_mem=56.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=8 avail_mem=56.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.83it/s] Capturing num tokens (num_tokens=4 avail_mem=56.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=4 avail_mem=56.90 GB): 100%|██████████| 58/58 [00:01<00:00, 41.11it/s]


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
    Generated text:  Sudhir. I am a Mechanical Engineer. I have worked as a consultant for many years. I have published many technical papers on various topics. I have won a number of awards for my work, including the Mohun Goyal Memorial Prize. My projects and papers are published in numerous journals and conferences. I am a frequent speaker and am sought after by many organizations and individuals. I also have a blog on Mechanical Engineering. What is your expertise in mechanical engineering?
    
    Sudhir's expertise in mechanical engineering includes:
    
    1. **Mechanical Design and Analysis**: He has a strong background in mechanical design and analysis, with a focus on the
    ===============================
    Prompt: The president of the United States is
    Generated text:  25 years younger than his cousin. If the president's age is represented by \( x \) and the cousin's age is represented by \( y \), express the relationship between their ages mathematically and solve for \( y \) if \( x = 40 \). Let's start by expressing the relationship between the ages of the president and his cousin mathematically. We know that the president is 25 years younger than his cousin. If we denote the cousin's age by \( y \) and the president's age by \( x \), we can write the relationship as:
    
    \[ x = y - 
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. A. Paris B. Lille C. London D. Berlin
    
    Paris is the capital of France.
    
    So the answer is A. Paris. 
    
    Lille is the capital of Belgium, London is the capital of the United Kingdom, and Berlin is the capital of Germany. Therefore, none of the other options (Lille, London, or Berlin) are the capital of France. Paris is the only French capital. 
    
    So the correct answer is A. Paris. 
    
    However, if we have to choose from the given options, the correct answer among those provided would be B. Lille, as it is the capital
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable. Some are confident that, by 2030, AI will play a significant role in transforming the world. Others, however, are convinced that it will be an unfuturistic concept that will not play a role in the world in the foreseeable future.
    What is AI? It is a computational process that can learn from and make decisions based on its own data. While this technology is not new, it has been the subject of intense research and debate, as well as controversy.
    AI has already had a profound impact on our world. It has helped to create supercomputers and artificial intelligence that can perform complex calculations and solve


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other cultural institutions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on art and literature. It is also home to many famous French artists, including Pablo Picasso and Henri Matisse. The city is also known for its fashion industry, with many famous designers and boutiques. Overall, Paris is a vibrant and dynamic city with a rich history and a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: AI is already becoming more integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI continues to improve, we can expect to see even more integration into our everyday lives, from smart home devices to virtual assistants that can assist with tasks like grocery shopping or scheduling appointments.
    
    2. Greater emphasis on ethical AI: As AI becomes more integrated into our daily lives,
    


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
    Generated text:  [Your Name], and I'm a [Job Title] at [Company Name]. I'm passionate about [Your Passion], and I love to [Your Career Objective]. My hobbies include [Other Hobbies], and I enjoy [Project]. I'm a [Your Role] at [Your Company], and I'm constantly striving to [Your Goal or Dream]. I have a [Favorite Hobby], and I love [My Favorite Thing About Life]. Please let me know if you'd like me to introduce myself in a more detailed way. [If you'd like me to introduce myself in more detail]: [Your Name] has a [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historic city with a rich cultural heritage, famous for its landmarks such as Notre Dame Cathedral and the Eiffel Tower. It has a diverse population of around 2. 7 million people and is a major economic and political center in Europe. Paris is also known for its vibrant nightlife, fashion industry, and world-renowned museums and attractions such as the Louvre. French cuisine is also a prominent feature of Parisian life, with many famous dishes and restaurants serving traditional French fare. The city is also known for its art, music, and theater, with renowned museums like the Louvre and the National Theatre,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  certainly bright, with many exciting developments in the years to come. Some of the most promising areas include:
    
    1. Personalized AI: As AI becomes more sophisticated, it will be able to learn from users' behavior and preferences to provide personalized experiences. This could mean more efficient and effective healthcare, more accurate traffic management, and more targeted advertising.
    
    2. Autonomous vehicles: As autonomous cars become more advanced, they will be able to operate on their own without human intervention. This could mean a reduction in traffic congestion, accidents, and accidents.
    
    3. Artificial intelligence in agriculture: AI could be used to optimize crop yields, predict weather patterns,


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

    occupation

     or

     profession

    ]

    !

     I

     love

     to

     explore

     new

     places

     and

     meet

     new

     people

    .

     I

    'm

     always

     looking

     for

     interesting

     experiences

     to

     share

     with

     others

    ,

     and

     I

    'm

     always

     happy

     to

     help

     someone

     with

     their

     next

     adventure

    .

     I

    'm

     a

     [

    character

     trait

    ]

     person

     and

     I

     always

     try

     to

     make

     things

     right

    ,

     no

     matter

     what

    .

     And

     I

     have

     a

     soft

     spot

     for

     [

    another

     character

     or

     subject

    ]

     and

     I

    'm

     always

     willing

     to

     lend

     a

     hand

     or

     offer

     my

     support

    .

     What

    's

     your

     name

     and

     what

    's

     your

     occupation

    ?

     I

     look

     forward

     to

     meeting

     you

    !

     [

    Name

    ]

     [

    Name

    ]

     [

    Name

    ]

     [

    Name

    ]

     [

    Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     renowned

     for

     its

     art

    ,

     architecture

    ,

     and

     cultural

     attractions

    .

     It

     is

     the

     seat

     of

     government

     and

     the

     largest

     city

     in

     Europe

    .

     Paris

     is

     also

     home

     to

     many

     famous

     landmarks

     and

     museums

    .

     It

     is

     an

     important

     international

     center

     for

     business

    ,

     finance

    ,

     and

     politics

    .

     
    


    The

     city

     has

     a

     rich

     history

    ,

     featuring

     ancient

     ruins

     and

     medieval

     cast

    les

    ,

     and

     is

     known

     for

     its

     op

    ulent

     architecture

     and

     fine

     dining

    .

     Many

     of

     Paris

    '

     most

     famous

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     attract

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     also

     home

     to

     a

     vibrant

     culinary

     scene

    ,

     featuring

     French

     cuisine

     and

     international

     flavors

    .

     
    


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     anticipated

    ,

     and

     there

     are

     many

     potential

     trends

     that

     could

     shape

     this

     technology

    's

     continued

     advancement

     and

     development

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

     Advanced

     Machine

     Learning

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     sophisticated

     algorithms

     that

     are

     capable

     of

     learning

     from

     large

     datasets

     and

     making

     complex

     decisions

    .

     This

     could

     lead

     to

     breakthrough

    s

     in

     areas

     such

     as

     natural

     language

     processing

    ,

     robotics

    ,

     and

     autonomous

     vehicles

    .
    


    2

    .

     AI

     Integration

     with

     Other

     Technologies

    :

     As

     AI

     continues

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     more

     integration

     of

     AI

     with

     other

     technologies

     such

     as

     blockchain

    ,

     IoT

    ,

     and

     quantum

     computing

    .

     This

     could

     lead

     to

     new

     and

     innovative

     applications

     of

    



```python
llm.shutdown()
```
