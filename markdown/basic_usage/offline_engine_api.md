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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.06it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 24.06it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.04 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.28it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.28it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.83it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.83it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.83it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.83it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.83it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.83it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.61it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.59it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.49it/s]Capturing num tokens (num_tokens=112 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.49it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.49it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.49it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.49it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.49it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.25it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.46it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 41.77it/s]


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
    Generated text:  Sarah, I'm a college student. I'm in the US, but I live in Spain. I've been in Spain for a month and I'm planning to return home. My current living situation is a two-bedroom apartment in Barcelona. I have my keys for the apartment and I've been living there for a while now. When I first arrived, I only had keys to my car, and now I have keys to my apartment. Also, I've been taking care of my dog and my cat. My dog has been here for about 2 years, and my cat has been here for about 2 years. My roommate got
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. (Judge true or false) The president of the United States is a person. (Judge true or false)
    To determine whether the statement "The president of the United States is a person" is true or false, we need to consider the nature of the position of the president of the United States.
    
    1. **Identify the authority and importance of the president of the United States:**
       - The president of the United States serves as the head of state and the commander-in-chief of the armed forces.
       - The position of president is considered the highest office in the federal government of the United States.
       - The
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. London
    
    B. Paris
    
    C. Berlin
    
    D. Geneva
    
    To determine the capital of France, let's follow these steps:
    
    1. **Identify the correct option**: The capital of France is Paris, which is option B.
    
    2. **Verify the capital**: 
       - London is the capital of the United Kingdom.
       - Paris is the capital of France.
       - Berlin is the capital of Germany.
       - Geneva is the capital of Switzerland.
    
    Given that the capital of France is Paris, and we have identified this correctly among the options, we can conclude that the correct answer is:
    
    \boxed{
    ===============================
    Prompt: The future of AI is
    Generated text:  often portrayed as a narrative of hope and triumph. While there are certainly some impressive advancements in AI technologies today, it is clear that there are also significant challenges and risks associated with these technologies. This paper aims to explore the potential risks and challenges associated with AI in the future, highlighting key areas that need to be addressed and discussed.
    
    One of the most pressing risks associated with AI is the potential for misuse and abuse. As AI systems become more sophisticated and complex, there is a greater risk that they may be used to commit crimes or engage in harmful behavior. For example, an AI system trained to identify and target extremist groups could be used to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few details about yourself, such as your age, gender, occupation, and any other relevant information]. And what's your favorite hobby or activity? I enjoy [insert a few details about your favorite hobby or activity, such as your favorite food, a favorite book, or a favorite sport]. And what's your favorite book or movie? I love [insert a few details about your favorite book or movie, such as the author or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Middle Ages, and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major cultural and economic center, with a diverse population and a vibrant nightlife. The city is home to many world-renowned museums, art galleries, and restaurants, and is a popular tourist destination. It is also known for its cuisine, with dishes such as croissants, escargot, and charcuterie being popular among locals and tourists alike. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: As AI becomes more advanced, it is likely to automate more tasks, freeing up human workers to focus on more complex and creative work. This could lead to increased efficiency and productivity, as well as the creation of new jobs in areas such as data analysis, machine learning, and software development.
    
    2. AI-driven healthcare: AI is already being used to improve the accuracy and speed of medical diagnosis and treatment. As AI becomes more advanced, it is likely to be used in more complex medical procedures, such as personalized medicine and virtual surgery. This could lead to more
    


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
    Generated text:  [Name], and I'm an [Occupation] who has a passion for [Favorite Hobby or Activity]. I'm currently [Age], and I have [Number of Years in Profession/Experience], and [Any Achievements or Contributions]. I enjoy [Describe your interests or hobbies]. How can someone best start their day at [Your Profession]? As an [Occupation], I would say [Describe the best way to start my day]. If someone were to ask me what I do for a living, what would I say? As a [Occupation], I would say [Describe my job responsibilities]. As a [Occupation], I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. This is the most populous city in Europe, and it is known for its rich history, beautiful architecture, and vibrant cultural scene. Paris is also the home of many world-renowned museums, such as the Louvre and the Musée d'Orsay, and the Eiffel Tower, which is one of the most recognizable landmarks in the world. The city is also known for its gastronomy, with its many famous food and drink destinations, including the Eiffel Tower, the Marne River, and the Croissants of Montmartre. With its beautiful architecture and rich cultural heritage, Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by the development of new technologies and the increasing sophistication of AI algorithms. Some potential trends that could shape the future of AI include:
    
    1. Increased focus on ethical considerations: There is growing concern about the potential ethical implications of AI, and as a result, there is likely to be more focus on developing AI that is ethical and transparent.
    
    2. More advanced natural language processing: As AI systems become more sophisticated, they will likely be able to process and understand natural language in more complex and nuanced ways. This could lead to new applications and opportunities for AI, such as natural language generation, speech recognition, and chatbots.
    
    


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

     [

    job

     title

    ]

     [

    Job

     Title

    ].

     I

    'm

     passionate

     about

     [

    What

     motiv

    ates

     you

    ]

     and

     have

     [

    How

     long

    ]

     on

     this

     role

    ,

     [

    Number

    ]

     years

     experience

     in

     this

     field

    ,

     and

     have

     always

     been

     a

     [

    What

     exc

    els

     in

     me

    ],

     [

    What

     does

     my

     best

     work

     look

     like

    ].

     I

    'm

     always

     looking

     for

     opportunities

     to

     [

    What

     I

    'm

     trying

     to

     achieve

    ]

     and

     I

    'm

     [

    What

     I

    'll

     be

     doing

     next

    ].

     I

     believe

     in

     [

    What

     I

     believe

     in

    ],

     and

     I

    'm

     confident

     in

     my

     ability

     to

     succeed

     in

     this

     role

    .

     So

    ,

     if

     you

    're

     interested

     in

     learning

     more

     about

     me

    ,

     I

    'm

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Its

     historical

     landmarks

     include

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

    .

     Paris

     is

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     annual

     French

     Quarter

     food

     and

     wine

     festivals

    .

     It

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     medieval

     architecture

     in

     the

    城

     centre

     and

     its

     modern

    ,

     industrial

     heart

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     museums

    ,

     and

     numerous

     museums

    ,

     and

     has

     an

     extensive

     network

     of

     transportation

     options

    ,

     including

     underground

     train

    ,

     tram

    ,

     and

     bus

     services

    .

     Paris

     is

     also

     known

     for

     its

     fashion

    ,

     art

    ,

     and

     music

     scenes

    ,

     making

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

     city

     with

     a

     rich

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     unpredictable

    ,

     with

     new

     technologies

     and

     advancements

     constantly

     emerging

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Autonomous

     vehicles

    :

     In

     the

     next

     few

     years

    ,

     we

     may

     see

     the

     widespread

     adoption

     of

     autonomous

     vehicles

    ,

     which

     will

     allow

     drivers

     to

     focus

     on

     other

     tasks

    ,

     such

     as

     navigating

     streets

    ,

     talking

     to

     passengers

    ,

     or

     handling

     emergencies

    .
    


    2

    .

     Smart

     homes

    :

     With

     the

     rise

     of

     AI

    ,

     we

     will

     see

     more

     and

     more

     home

     appliances

     and

     devices

     that

     can

     learn

     and

     adapt

     to

     our

     needs

    ,

     such

     as

     smart

     ther

    most

    ats

    ,

     smart

     lighting

    ,

     and

     smart

     security

     systems

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     diagnose

    



```python
llm.shutdown()
```
