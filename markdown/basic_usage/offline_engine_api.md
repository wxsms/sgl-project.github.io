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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.38it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.98it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.98it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.98it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.98it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.98it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.98it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.98it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.98it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.98it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.98it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.98it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.09it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.09it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.09it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.09it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.09it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.09it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.09it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.91 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.88 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.88 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.88 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.88 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.88 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.88 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.87 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.86 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.86 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.86 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.85 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.85 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.85 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.84 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=57.84 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.84 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.84 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.83 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.83 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.83 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.82 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.82 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.82 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.80 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.66it/s]Capturing num tokens (num_tokens=960 avail_mem=57.82 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.66it/s] Capturing num tokens (num_tokens=896 avail_mem=57.82 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.66it/s]

    Capturing num tokens (num_tokens=832 avail_mem=57.81 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.66it/s]Capturing num tokens (num_tokens=768 avail_mem=57.81 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.66it/s]Capturing num tokens (num_tokens=768 avail_mem=57.81 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=704 avail_mem=57.81 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=640 avail_mem=57.80 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=576 avail_mem=57.80 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=512 avail_mem=57.79 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=480 avail_mem=57.80 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=448 avail_mem=57.80 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=448 avail_mem=57.80 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=416 avail_mem=57.80 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=384 avail_mem=57.80 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=352 avail_mem=57.79 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.55it/s]

    Capturing num tokens (num_tokens=320 avail_mem=57.78 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=288 avail_mem=57.78 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=256 avail_mem=57.78 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=256 avail_mem=57.78 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.18it/s]Capturing num tokens (num_tokens=240 avail_mem=57.78 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.18it/s]Capturing num tokens (num_tokens=224 avail_mem=57.77 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.18it/s]Capturing num tokens (num_tokens=208 avail_mem=57.77 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.18it/s]Capturing num tokens (num_tokens=192 avail_mem=57.77 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.18it/s]Capturing num tokens (num_tokens=176 avail_mem=57.76 GB):  64%|██████▍   | 37/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=160 avail_mem=57.76 GB):  64%|██████▍   | 37/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=160 avail_mem=57.76 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.27it/s]Capturing num tokens (num_tokens=144 avail_mem=57.76 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.27it/s]Capturing num tokens (num_tokens=128 avail_mem=57.76 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.27it/s]

    Capturing num tokens (num_tokens=112 avail_mem=57.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.27it/s]Capturing num tokens (num_tokens=96 avail_mem=57.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.27it/s] Capturing num tokens (num_tokens=80 avail_mem=57.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.27it/s]Capturing num tokens (num_tokens=80 avail_mem=57.75 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.45it/s]Capturing num tokens (num_tokens=64 avail_mem=57.74 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.45it/s]Capturing num tokens (num_tokens=48 avail_mem=57.74 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.45it/s]Capturing num tokens (num_tokens=32 avail_mem=57.74 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.45it/s]Capturing num tokens (num_tokens=28 avail_mem=57.73 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.45it/s]Capturing num tokens (num_tokens=24 avail_mem=57.73 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.45it/s]Capturing num tokens (num_tokens=24 avail_mem=57.73 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=20 avail_mem=57.73 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=16 avail_mem=57.73 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.72it/s]

    Capturing num tokens (num_tokens=12 avail_mem=57.72 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=8 avail_mem=57.72 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.72it/s] Capturing num tokens (num_tokens=4 avail_mem=57.71 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=4 avail_mem=57.71 GB): 100%|██████████| 58/58 [00:01<00:00, 43.19it/s]


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
    Generated text:  Ashley, I'm 30 years old, my parents are both in their 70s. My brother and I have a baby together. What do you think of the idea of adopting? 
    
    My brother and I are not very smart, and we are very much into the carefree lifestyle. The adoptive parents would like to have a nice home for him/her to stay. We are very happy with the baby we are bringing up. I know adoption is a very expensive and it is for the best of his/her health. Please help me get the most accurate answer from your side as fast as possible. Thank you.
    
    ---
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to initiate a new program to encourage American students to pursue higher education. The program would provide a stipend to students who graduated from high schools in the top 10% of their state or region. The president is concerned that a large number of students who do not meet the eligibility criteria could leave the program. 
    
    To test this hypothesis, the president wants to use a statistical test. He decides to use the t-distribution to determine whether the observed number of students leaving the program is significantly different from what would be expected if the program were effective and all eligible students were retained.
    
    The president finds that the observed number
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Lyon
    C. Nice
    D. London
    
    To determine the capital of France, we need to identify the capital city of the country that has the name "France." The capital of France is Paris.
    
    Let's break it down step by step:
    
    1. **Identify the country**: France is a country in Western Europe.
    2. **Identify the capital**: The capital of France is Paris.
    
    Based on the above information, the capital of France is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, and like any other technology, it is currently challenging to predict how it will evolve. The field of AI is a rapidly growing and constantly evolving one, and it is crucial to understand its current and potential future trends and developments. In this article, we will look at the current trends of AI and how it will impact the future of our lives.
    One of the most significant trends in AI is the increasing integration of AI in various sectors of our lives. For example, AI is being used in healthcare, finance, transportation, and more. This integration is driving the development of new technologies and innovations that can improve our lives and increase efficiency


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located in the south of the country and is the seat of government, administration, and culture in France. Paris is known for its rich history, art, and cuisine, and is a major tourist destination. It is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its fashion industry, with many famous designers and boutiques located in the city. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, productivity, and cost savings for businesses.
    
    2. Enhanced privacy and security: As AI technology becomes more sophisticated, we can expect to see increased concerns about privacy and security. This will require advancements in AI that are designed to protect
    


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
    Generated text:  [Name], and I'm a [Type of Person]. I'm excited to meet you here and look forward to interacting with you. Let's connect! [Name]. [Name], please. 📱💼✨ You're from [City]! 📱💼✨ This is my first time here and I am here to meet some interesting people! [Name]. How are you? 📱💼✨ I'm always looking for new connections and hope to make some friends. What about you? 📱💼✨ Are you looking for a good time at the moment? 📱💼✨ It would be great to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris, the city of lights and towering buildings, is France's political, economic, and cultural capital. It is also home to the world's largest shopping mall, the Eiffel Tower, and a plethora of art and cultural institutions. The city is known for its stunning architecture, famous landmarks, and the annual Marseillaise Festival, which showcases the city's music and dance traditions. Paris is also renowned for its culinary scene, with numerous international restaurants and bistros that offer an exquisite dining experience. Its climate, which can vary from warm to cold, has made it a popular destination for tourism and outdoor activities. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  one of innovation, change, and growth. Here are some possible trends that could unfold in the next few decades:
    
    1. Self-driving cars: Self-driving cars are expected to become more advanced and reliable in the coming years. AI is already being used in autonomous vehicles, and it's likely that we'll see even more advancements in this area in the future.
    
    2. Personalized AI: As AI technology advances, we'll see more personalized and efficient solutions to problems. Personalized AI will allow machines to learn and adapt to specific situations, making them more effective and efficient.
    
    3. AI in healthcare: AI is already being used in medical


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

     am

     a

     [

    role

    ]

     from

     [

    company

     name

    ].

     I

     bring

     a

     unique

     blend

     of

     my

     own

     experiences

    ,

     skills

    ,

     and

     passions

     to

     this

     role

    ,

     allowing

     me

     to

     contribute

     to

     the

     team

     in

     a

     truly

     unique

     way

    .


    As

     someone

     who

     has

     always

     been

     curious

     and

     curious

    ,

     I

     am

     always

     eager

     to

     learn

     and

     grow

    .

     Whether

     it

    's

     a

     new

     technology

    ,

     a

     new

     idea

    ,

     or

     a

     new

     skill

    ,

     I

     am

     always

     willing

     to

     try

     and

     explore

     new

     opportunities

    .


    In

     my

     free

     time

    ,

     I

     enjoy

     spending

     time

     with

     my

     family

     and

     friends

    ,

     reading

     books

    ,

     and

     pursuing

     my

     love

     of

     photography

    .

     I

     am

     constantly

     striving

     to

     push

     the

     boundaries

     of

     what

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     a

     UNESCO

     World

     Heritage

     site

    .

     It

     is

     also

     the

     oldest

     continuously

     inhabited

     city

     in

     the

     world

    .

     Paris

     has

     a

     rich

     history

    ,

     with

     many

     famous

     landmarks

     and

     architecture

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     known

     for

     its

     fashion

    ,

     art

    ,

     and

     wine

     culture

    .

     The

     French

     language

     and

     French

     cuisine

     are

     also

     very

     popular

     in

     Paris

    .

     
    


    Translate

     the

     following

     English

     sentence

     into

     Spanish

    .

     "

    The

     latest

     news

     on

     the

     company

    's

     stock

     was

     announced

     on

     the

     first

     day

     of

     the

     month

    ."
    


    Spanish

     translation

    :

     "

    La

     not

    icia

     más

     rec

    iente

     sobre

     la

     compañía

     se

     anunci

    ó

     el

     día

     

    1

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

    ,

     and

     it

    's

     uncertain

     which

     trends

     will

     dominate

     and

     which

     ones

     will

     fade

     away

    .

     However

    ,

     here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     AI

     will

     become

     more

     personalized

    :

     As

     AI

     becomes

     more

     accessible

     and

     accessible

     to

     the

     general

     population

    ,

     we

     can

     expect

     more

     personalized

     and

     context

    ually

    -ad

    aptive

     AI

    .

     AI

     that

     can

     learn

     and

     adapt

     to

     different

     users

     and

     their

     environments

     will

     be

     more

     effective

     at

     providing

     tailored

     solutions

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     with

     other

     technologies

    :

     AI

     is

     becoming

     increasingly

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

     and

     cameras

    ,

     to

     improve

     its

     ability

     to

     perform

     tasks

     more

     accurately

     and

     efficiently

    .

     AI

     that

     can

    



```python
llm.shutdown()
```
