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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.35it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.72it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.79it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.79it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.79it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.79it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.79it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.79it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.79it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.79it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.79it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.79it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.79it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.81it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.81it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.81it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.81it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.81it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.81it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.81it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 14.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 14.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.12 GB):   3%|▎         | 2/58 [00:00<00:03, 14.74it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=75.12 GB):   7%|▋         | 4/58 [00:00<00:03, 16.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.12 GB):   7%|▋         | 4/58 [00:00<00:03, 16.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):   7%|▋         | 4/58 [00:00<00:03, 16.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.88 GB):   7%|▋         | 4/58 [00:00<00:03, 16.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.88 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.48it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.62it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.46it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.77it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.28it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 43.31it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.40it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.40it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.57it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 37.04it/s]


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
    Generated text:  Denny. I am a student of History, and I have been learning about the events that happened in my country during the Civil War. My question is: Was the Union the only one that participated in the Civil War?
    Answer with confidence. Yes, the Union was the main participant in the Civil War. The Union was the largest and most influential of the several major regions and factions that participated in the war. The Confederate States of America, which included the states in the Southern states of the United States, fought a separate war against the Union and the North. This created a complex interplay of various forces, including the Union's own
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy organizing the 2008 presidential campaign. During the first week, he worked for 5 hours. In the second week, he worked half the hours he worked in the first week. In the third week, he worked 2 additional hours compared to the second week. How many hours did the president work during the entire campaign?
    
    To determine the total number of hours the president worked during the entire campaign, we need to calculate the hours worked each week and then sum them up.
    
    1. **First Week:**
       The president worked for 5 hours in the first week.
       \[
       \text{Hours
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it has a population of 2.8 million. The population of the capital of a country is 60% of the population of the capital of France. What is the population of the capital of a country?
    
    To determine the population of the capital of a country, we start with the given information that the population of Paris, the capital of France, is 2.8 million. We are also told that the population of the capital of a country is 60% of the population of the capital of France. Therefore, we need to calculate 60% of 2.8 million.
    
    First,
    ===============================
    Prompt: The future of AI is
    Generated text:  about to hit a major milestone, and it's that of a supercomputer that has been built to do what?
    Options:
    A) Launch a new generation of smartphones
    B) Help people perform complex calculations faster
    C) Create a large number of new species
    D) Have a heart attack
    
    The problem requires you to find the right answer based on the given context. The context is about AI and what it can do in the future. The AI that has been built is capable of supercomputing and performing complex calculations faster, which implies that it is capable of performing calculations faster than any current AI. This means the AI is capable


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


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major transportation hub, with many major highways and airports connecting the city to other parts of France and the world. Paris is a popular tourist destination, with millions of visitors annually. It is also a cultural center, with many museums, theaters, and other cultural institutions. The city is known for its cuisine, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This will require us to develop new technologies and practices to protect
    


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
    Generated text:  __________, a(n) _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an _____________. I'm a/an __________
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To elaborate, Paris is the largest city in Europe by land area, as well as the 15th-largest in the world by population. It is a cultural and political center, home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, the Notre Dame Cathedral, and the Champs-Élysées. The city is also known for its rich history and diverse culture, which is reflected in its various museums, art galleries, and festivals. Despite its size, Paris is also known for its charming and picturesque neighborhoods, which offer a unique experience for tourists and locals alike. The city has
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be marked by rapid advancements in areas such as machine learning, natural language processing, computer vision, and robotics. Here are some potential trends that could shape the AI landscape in the coming years:
    
    1. Greater emphasis on ethical considerations: As AI systems become more sophisticated, there will be a growing emphasis on ethical considerations and responsible use of AI. This will include ensuring that AI systems are fair and unbiased, and that they are not used to perpetuate or exacerbate social or economic inequalities.
    
    2. Increased focus on privacy and security: As more data is collected and processed by AI systems, there will be increased scrutiny of how this data


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

    Job

     Title

    ]

     at

     [

    Company

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     start

     a

     new

     adventure

     with

     you

    .

     
    


    I

    'm

     [

    Age

    ],

     [

    Gender

    ]

     and

     I

    'm

     [

    Occup

    ation

    ].

     I

    'm

     very

     experienced

     in

     [

    field

    ],

     with

     over

     [

    number

    ]

     years

     of

     experience

     in

     [

    job

     title

    ].

     I

    'm

     a

     [

    character

    istic

    ]

     individual

     who

    's

     always

     [

    positive

     or

     negative

    ].

     I

     enjoy

     [

    att

    itude

    ]

     tasks

    ,

     [

    skill

    ]

     tasks

    ,

     or

     [

    skill

    ]

     tasks

    ,

     and

     I

    'm

     always

     [

    positive

     or

     negative

    ].

     I

    'm

     always

     [

    att

    itude

    ]

     to

     learning

     and

     always

     want

     to

     grow

     and

     develop

     [

    skill

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     bustling

     city

     known

     for

     its

     history

    ,

     art

    ,

     and

     cuisine

    .

     The

     city

     is

     also

     home

     to

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     home

     to

     many

     international

     institutions

     and

     cultural

     organizations

    .

     The

     city

     is

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     The

     French

     government

     and

     many

     French

     citizens

     are

     known

     for

     their

     love

     of

     the

     city

     and

     its

     culture

    .

     Paris

     is

     considered

     one

     of

     the

     most

     important

     cities

     in

     the

     world

     and

     is

     often

     referred

     to

     as

     the

     "

    met

    ropolis

     of

     the

     world

    ".

     Overall

    ,

     Paris

     is

     a

     fascinating

     and

     vibrant

     city

     that

     is

     a

     must

    -

    visit

     for

     anyone

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     rapidly

     changing

    ,

     but

     here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     technology

    's

     development

    :
    


    1

    .

     Increased

     automation

     and

     automation

     of

     repetitive

     tasks

    :

     With

     the

     advancements

     in

     machine

     learning

     and

     artificial

     intelligence

    ,

     automation

     will

     become

     more

     prevalent

     in

     many

     industries

    .

     This

     means

     that

     robots

     and

     other

     AI

    -powered

     systems

     will

     be

     able

     to

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

    ,

     such

     as

     data

     entry

    ,

     customer

     service

    ,

     and

     assembly

     line

     work

    .
    


    2

    .

     Deep

     learning

     and

     neural

     networks

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     it

    's

     likely

     that

     we

    'll

     see

     more

     focus

     on

     deep

     learning

     and

     neural

     networks

    .

     These

     technologies

     will

     be

     able

     to

     learn

     from

     large

     datasets

    ,

     making

     it

     possible

    



```python
llm.shutdown()
```
