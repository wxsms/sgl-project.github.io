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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.25it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.40it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.40it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.40it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.46it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.37it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.37it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.37it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.82it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.82it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.82it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.82it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.82it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.82it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=256 avail_mem=75.73 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=240 avail_mem=75.73 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=224 avail_mem=75.70 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=224 avail_mem=75.70 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.47it/s]Capturing num tokens (num_tokens=208 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.47it/s]Capturing num tokens (num_tokens=192 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.47it/s]

    Capturing num tokens (num_tokens=144 avail_mem=75.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.66it/s] Capturing num tokens (num_tokens=80 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=32 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=24 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.89it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.13it/s]Capturing num tokens (num_tokens=12 avail_mem=74.97 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.13it/s]Capturing num tokens (num_tokens=8 avail_mem=74.96 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.13it/s] Capturing num tokens (num_tokens=4 avail_mem=74.96 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.13it/s]Capturing num tokens (num_tokens=4 avail_mem=74.96 GB): 100%|██████████| 58/58 [00:01<00:00, 42.39it/s]


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
    Generated text:  Vinny and I'm a retired engineer. I have a passion for talking to people. I'm a huge fan of music and travel. I started talking to people in January and have been interviewing and writing for most of 2017. I've interviewed some of the biggest names in business, music, sports, and travel. I'm open to writing stories that relate to the things you are interested in.
    I'm a freelance journalist with a 20+ year background in writing and writing about writing. My writing has been published in many outlets, including the Wall Street Journal, The Washington Post, The New York Times,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to estimate the average weight of a specific type of bird in the US. The bird is known to have a normal distribution. The president has a sample of 50 birds, and the average weight of the birds in the sample is 120 grams. The standard deviation of the weights in the sample is 15 grams. Using the t-distribution, what is the estimated standard error of the mean weight of the birds in the US? Round your answer to three decimal places. To estimate the standard error of the mean weight of the birds in the US, we need to use the formula for the standard error of the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris  
    B: Versailles  
    C: Rome  
    D: Berlin
    
    To determine the capital of France, we need to recall the capital cities of the countries mentioned in the options. Let's break it down step by step:
    
    1. **Paris** is the capital of France.
    2. **Versailles** is the capital of France.
    3. **Rome** is the capital of Italy.
    4. **Berlin** is the capital of Germany.
    
    Since the problem asks for the capital of France, and we have already established that Paris is the capital of France, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it's also a threat. A study by the National Academy of Sciences suggests that AI is beginning to reach a point where it will become a tool for people, not machines. This is certainly a concern. But isn't it true that we are responsible for how we make decisions about the AI we use? In other words, isn't the "bias" of AI algorithms an issue that we must address?
    I believe that the answer is yes, but that we must take a closer look at the issues that arise when we put algorithms into use.
    One of the biggest issues is that when we make a decision about the AI we


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and [job title]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France and is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and is known for its rich history, art, and cuisine. The city is also home to the French Parliament and the French Parliament building. 
    
    B. False is incorrect because Paris is indeed the capital of France, and it is a major cultural and economic center. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and responsible development. This could lead to more stringent regulations and guidelines for AI development and deployment
    


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
    Generated text:  [Your Name] and I'm a [Your Profession] with [Your Occupation] experience. I'm passionate about [Your Hobby/Interest] and have a deep appreciation for [Your Hobby/Interest]. I'm committed to [Your Career Goal or Passion] and always ready to help others learn and grow. What excites you most about my career, and how do you aim to achieve it? Let's get started!
    [Your Name] Professional Summary:
    As [Your Profession], I am a [Your Occupation] with [Your Profession] experience. I am passionate about [Your Hobby/Interest] and have a deep appreciation for [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and one of the most populous cities in the world. The city has a rich cultural history and is home to many famous landmarks, such as the Eiffel Tower and Notre-Dame Cathedral. It is also known for its cuisine, art, and music. 
    
    Paris is a bustling city with a lively nightlife and is a popular tourist destination for many people worldwide. It is home to the Louvre Museum and the Champ de Mars, and many other important attractions and landmarks. The city is also home to a diverse population of people from all over the world. 
    
    In conclusion, Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of possibilities, and it is hard to predict exactly what the path will take. However, some potential trends that we might see in the near and medium term include:
    
    1. AI becoming more integrated into our daily lives: As AI becomes more advanced, it will be able to perform tasks that are currently the domain of humans, such as language translation, decision-making, and problem-solving. This integration will likely lead to a more personalized and convenient way of interacting with technology.
    
    2. AI becoming more human-like: As AI becomes more capable, it could begin to show more of a human-like quality. For example, AI could be able


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

    Age

    ]

     year

     old

     [

    Occup

    ation

    ].

     I

     have

     a

     love

     for

     [

    Your

     Major

    /

    Interest

    ].

     I

     am

     [

    Your

     Profession

    ]

     because

     [

    Your

     Background

     or

     What

     Makes

     You

     Unique

    ].

     I

     believe

     in

     [

    Your

     Values

     or

     Why

     You

     Do

     What

     You

     Do

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     learn

     and

     grow

    ,

     so

     I

     am

     always

     eager

     to

     learn

     from

     others

     and

     try

     new

     things

    .

     I

     have

     a

     strong

     work

     ethic

    ,

     which

     I

     believe

     will

     help

     me

     succeed

     in

     my

     career

    .

     I

     am

     proud

     to

     be

     [

    Your

     Profession

    ]

     and

     I

     am

     excited

     to

     have

     the

     opportunity

     to

     work

     with

     [

    Your

     Company

    /

    Group

    /

    Place

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     is

     accurate

    .

     Paris

     is

     the

     capital

     and

     largest

     city

     of

     France

    ,

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culture

    .

     The

     city

     is

     also

     home

     to

     the

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

     the

     Notre

    -D

    ame

     Cathedral

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

    ,

     supporting

     the

     country

    's

     economy

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     had

     a

     population

     of

     around

     

    2

    .

     

    2

     million

     people

    .

     The

     city

     is

     home

     to

     many

     notable

     French

     institutions

    ,

     including

     the

     Lou

    vre

     Museum

    ,

     the

     National

     Museum

     of

     Modern

     Art

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     The

     French

     government

     and

     culture

     are

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     varied

    ,

     with

     new

     areas

     of

     research

     and

     development

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

    :
    


    1

    .

     Deep

     learning

    :

     As

     technology

     continues

     to

     advance

    ,

     deep

     learning

     will

     be

     an

     increasingly

     important

     component

     of

     AI

    .

     This

     involves

     building

     algorithms

     that

     can

     learn

     and

     recognize

     complex

     patterns

     in

     data

    ,

     which

     will

     be

     used

     to

     improve

     computer

     vision

     and

     natural

     language

     processing

    .
    


    2

    .

     Quantum

     computing

    :

     With

     the

     rise

     of

     quantum

     computing

    ,

     AI

     is

     set

     to

     experience

     a

     major

     breakthrough

    .

     Quantum

     computers

     have

     the

     potential

     to

     solve

     problems

     that

     are

     currently

     in

    tract

    able

     for

     classical

     computers

    ,

     such

     as

     breaking

     encryption

     codes

     or

     sim

    ulating

     complex

     physical

     systems

    .
    


    3

    .

     Natural

     language

     processing

    :

    



```python
llm.shutdown()
```
