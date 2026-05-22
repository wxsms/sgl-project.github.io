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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.19it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.42it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.46it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.34it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.24it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 43.04it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.69it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.69it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.69it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.69it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.69it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.69it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=192 avail_mem=74.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.80it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.80it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 40.76it/s]


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
    Generated text:  Michal and I will be studying at the University of the Philippines today. I'm quite excited about my time there, as I have a few options to choose from and I'm eager to learn and grow. I'm particularly interested in pursuing a master's degree in educational psychology, and I have been working as a teaching assistant for several years. However, I'm not sure how I can prepare for this academic pursuit, so I was wondering if you could provide me with some advice on how to start this path of education.
    Certainly! Starting a master's degree in educational psychology is a great step towards your career goals, and the preparation you
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she helps to decide major decisions for the country. In this lesson, we're going to learn more about the president. We're going to learn some interesting facts about the president, and we're going to watch a video that shows us how a president is elected.
    The president of the United States is chosen by the people through a process called the electoral college. It's a way for the people to elect the person who will be the leader of the country.
    The president's job is to make important decisions for the country. They have to make tough choices and deal with difficult situations.
    One of the things
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris B: Brussels C: Nice D: Lyon
    To determine the capital of France, we need to recall that the capital of France is Paris. The capital of France is located in the south-central region of the country, near the Mediterranean Sea.
    
    Therefore, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is not without its challenges. While it promises to revolutionize many industries, it also presents significant challenges that need to be addressed.
    
    ### 1. Safety Concerns
    
    One of the primary concerns when it comes to AI systems is the potential for misuse or abuse. As AI technologies become more sophisticated, it is possible that they could be used to manipulate or exploit individuals or organizations, leading to significant harm or damage. This could range from hacking into personal information to causing physical harm or even causing financial damage.
    
    To mitigate these risks, it is important for AI developers to implement robust safety features and regulations. This includes measures such


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a few key points about your background, education, or skills]. I'm looking forward to meeting you and discussing how I can contribute to your team. What can you tell me about your background? I'm a [insert a few key points about your background, education, or skills]. I'm looking forward to meeting you and discussing how I can contribute to your team. What can you tell me about your background? I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Parliament House. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The French capital is a vibrant and diverse city with a rich history and a strong sense of French identity. It is a city that is constantly evolving and is a popular destination for tourists and locals alike
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well
    


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
    Generated text:  [name], and I'm a [age] year old [job title] who works at [company name]. I have a passion for [occupation] and enjoy [reason for job]. I have a strong work ethic, and I'm always looking for ways to improve my skills and knowledge. I'm always up for learning new things, and I'm eager to expand my horizons. What's your name, and what's your occupation? [name]. Hello, my name is [name], and I'm a [age] year old [job title] who works at [company name]. I have a passion for [occupation]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital of France and is known for its historical landmarks, fashion district, and its role in the French literary, artistic, and cultural scene. The city is also home to the Eiffel Tower and the Louvre Museum. Paris is one of the most visited cities in the world and a cultural and political hub, hosting numerous world fairs, music festivals, and parades. It's also the largest city in the European Union by land area. Paris is the world's most expensive city, with the cost of living being the highest in the world. It is also the second largest economy in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic and involves a complex interplay of technological advancements, ethical considerations, and societal implications. Here are some possible future trends in AI:
    
    1. Personalized AI: As AI becomes more capable of understanding and learning from user behavior and preferences, it is expected to become more personalized. AI-powered systems will be able to tailor their responses to individual users, offering more relevant and personalized experiences.
    
    2. Autonomous vehicles: Autonomous vehicles are already a reality, but they are expected to become even more integrated into our daily lives. AI-powered autonomous vehicles will be able to navigate roads and intersections, respond to pedestrians, and handle accidents on the roads.
    
    


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

    role

    ]

     for

     [

    Company

    /

    Ent

    ertainment

     Industry

    /

    Other

    ].

     I

    've

     always

     been

     a

     [

    positive

     trait

    ]

     person

     and

     am

     passionate

     about

     [

    why

     you

    're

     passionate

    ].

     I

     thrive

     on

     [

    reason

    s

     for

     passion

    ],

     and

     I

     believe

     in

     [

    why

     you

     believe

    ].

     I

     love

     [

    career

     goal

    ]

     and

     I

    'm

     willing

     to

     work

     [

    how

     much

     you

    'd

     like

     to

     work

     for

     me

    ].

     I

    'm

     [

    age

     range

    ]

     years

     old

    ,

     and

     I

     grew

     up

     in

     [

    city

    ],

     where

     [

    something

     special

     about

     your

     upbringing

    ]

     shaped

     my

     personality

    .

     I

    've

     always

     been

     [

    about

     yourself

    ],

     and

     I

    'm

     always

     looking

     for

     [

    opport

    unities

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     also

     known

     as

     the

     "

    City

     of

     Light

    ."

     
    


    Answer

     this

     question

     by

     providing

     the

     shortest

     version

     possible

    :

     What

     is

     the

     name

     of

     the

     capital

     city

     of

     France

    ?

     Paris

    .

     The

     answer

     is

    :

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     that

     bright

     future

     is

     one

     of

     its

     capabilities

     to

     understand

     natural

     language

    ,

     translate

     text

    ,

     and

     more

    .

     It

     can

     learn

     from

     data

     and

     improve

     as

     it

     receives

     more

     information

    .
    


    In

     the

     coming

     years

    ,

     we

     will

     see

     even

     greater

     developments

     in

     AI

    .

     In

     the

     future

    ,

     AI

     will

     be

     more

     relevant

     in

     areas

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     security

    .

     It

     will

     be

     able

     to

     identify

     patterns

     and

     make

     predictions

    ,

     which

     will

     allow

     it

     to

     predict

     and

     respond

     to

     natural

     disasters

    ,

     and

     it

     will

     be

     able

     to

     do

     much

     of

     the

     work

     that

     is

     currently

     done

     by

     humans

    .
    


    In

     addition

    ,

     AI

     will

     become

     more

     integrated

     with

     human

     lives

    .

     For

     example

    ,

     AI

     will

     be

     used

    



```python
llm.shutdown()
```
