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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.91it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 24.13it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.10 GB):   3%|▎         | 2/58 [00:00<00:04, 12.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.01 GB):   3%|▎         | 2/58 [00:00<00:04, 12.63it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.44 GB):   3%|▎         | 2/58 [00:00<00:04, 12.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.44 GB):   3%|▎         | 2/58 [00:00<00:04, 12.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.44 GB):   9%|▊         | 5/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.43 GB):   9%|▊         | 5/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.42 GB):   9%|▊         | 5/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.42 GB):   9%|▊         | 5/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.42 GB):   9%|▊         | 5/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.42 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.42 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.68it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=57.41 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.41 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.40 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.40 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.40 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.39 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.39 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.39 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.39 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.39 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.59it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=57.37 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.59it/s]Capturing num tokens (num_tokens=960 avail_mem=57.38 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.59it/s] Capturing num tokens (num_tokens=896 avail_mem=57.38 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.59it/s]Capturing num tokens (num_tokens=832 avail_mem=57.37 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.59it/s]Capturing num tokens (num_tokens=832 avail_mem=57.37 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=768 avail_mem=57.37 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=704 avail_mem=57.37 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=640 avail_mem=57.36 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=576 avail_mem=57.36 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=512 avail_mem=57.35 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=512 avail_mem=57.35 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=480 avail_mem=57.36 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]

    Capturing num tokens (num_tokens=448 avail_mem=57.36 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=416 avail_mem=57.36 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=384 avail_mem=57.36 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=352 avail_mem=57.35 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=352 avail_mem=57.35 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=320 avail_mem=57.35 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=288 avail_mem=57.34 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=256 avail_mem=57.34 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=240 avail_mem=57.34 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=224 avail_mem=57.33 GB):  59%|█████▊    | 34/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=224 avail_mem=57.33 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=208 avail_mem=57.33 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.21it/s]

    Capturing num tokens (num_tokens=192 avail_mem=57.33 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=176 avail_mem=57.33 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=160 avail_mem=57.32 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=144 avail_mem=57.32 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=144 avail_mem=57.32 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=128 avail_mem=57.32 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=112 avail_mem=57.32 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=96 avail_mem=57.31 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.02it/s] Capturing num tokens (num_tokens=80 avail_mem=57.31 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=64 avail_mem=57.31 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=64 avail_mem=57.31 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=48 avail_mem=57.30 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.03it/s]

    Capturing num tokens (num_tokens=32 avail_mem=57.30 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=28 avail_mem=57.29 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=24 avail_mem=57.29 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=20 avail_mem=57.29 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=20 avail_mem=57.29 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=16 avail_mem=57.29 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=12 avail_mem=57.28 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=8 avail_mem=57.28 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.38it/s] Capturing num tokens (num_tokens=4 avail_mem=57.28 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=4 avail_mem=57.28 GB): 100%|██████████| 58/58 [00:01<00:00, 40.57it/s]


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
    Generated text:  Ming. I'm an Englishman. I am in a middle school. I have a good friend. His name is Bill. His parents are teachers. They always try to teach Bill to speak English. Bill always wants to learn English. He thinks it's very interesting to speak English. Now, Bill can speak English very well. He reads English books every day. Bill learns English from the help of his teacher. Bill can speak English. He thinks it's very helpful. He can say words correctly in English. Now Bill has learned English well. He can learn another language. He is very happy. He can understand other people and
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by a bicameral legislature, with the Senate having 10 senators and the House of Representatives having 435 representatives. In which year will the last representative be seated for the 100th time?
    To determine the year in which the last representative will be seated for the 100th time, we need to track the seating of the representatives in both the Senate and the House of Representatives.
    
    1. **Senate:**
       - The Senate has 100 representatives.
       - The first representative is seated on January 1, 2023.
       - Each year, 1
    ===============================
    Prompt: The capital of France is
    Generated text:  ( ).
    A. Paris
    B. London
    C. Rome
    D. Berlin
    Answer:
    A
    
    According to the 'Guidelines for the Construction of a Global Marine Oil Pollution Emergency Plan', which of the following is NOT an element of the 'Four Stages' of the oil spill response plan?
    A. Initial stage
    B. Containment stage
    C. Recovery stage
    D. Decontamination stage
    Answer:
    D
    
    For a linear list of size n, the number of comparisons required to find the first occurrence of an element x is ____.
    A. O(n)
    B. O(2n)
    C.
    ===============================
    Prompt: The future of AI is
    Generated text:  secure if we use it wisely and properly. Use AI responsibly, understand its impact on society and know how to use it effectively. This course will give you an overview of the future of AI in the form of a big picture. 1. Chapter 1: Overview of AI 2. Chapter 2: The Future of AI 3. Chapter 3: AI in Healthcare 4. Chapter 4: AI in Finance 5. Chapter 5: AI in Education 6. Chapter 6: AI in Media and Entertainment 7. Chapter 7: AI in Transportation and Logistics 8. Chapter 8:


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


    Generated text:  Paris, also known as "La Ville de Paris" and the "City of Light". It is the largest city in France and the third largest in the world. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Paris is a popular tourist destination and a major economic center in France. It is the seat of the French government, the French Parliament, and the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Expansion of AI applications: AI is likely to be used in a wider range of applications, including healthcare, finance, transportation, and manufacturing, as well as in areas such as education and entertainment.
    
    4. Development of new AI technologies
    


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
    Generated text:  [Name]. I am a [Occupation] who have been [Number of Years] in the field of [Field of Interest]. I am passionate about [What You Are Passionated About]. My interest in [Field of Interest] has taken me to [Where You Have Been] and I have been working hard to [What You Do In Your Work] to achieve [Your Goal]. I am always striving to improve my skills and knowledge in order to be [What You Want To Be In Your Field Of Interest]. I believe that my journey towards my goal is shaped by the [Inspiring Factor]. I am a [Choose Your Answer
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an ancient city famous for its Notre-Dame Cathedral and iconic landmarks such as the Eiffel Tower and the Louvre Museum. It is the third largest city in Europe and has a rich history dating back over 1,000 years. The city has a multicultural population and is home to many universities, theaters, and museums. Paris is known for its lively nightlife, cultural events, and access to top-notch restaurants and food outlets. It is a major hub for business, finance, and art, and is considered the cultural and economic center of the European Union. The city has been a stronghold of the French Revolution,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and complex, but here are some possible trends that are currently being studied and discussed:
    
    1. Increased emphasis on ethical considerations: As AI becomes more advanced, it is likely to be used for more ethical purposes. For example, AI could be used to assist in decision-making processes that could benefit society, rather than just being used to perpetuate inequality.
    
    2. Development of more advanced machine learning algorithms: As AI technology continues to improve, new algorithms will be developed that can handle more complex tasks and generate more accurate results.
    
    3. Integration of AI with human expertise: As AI becomes more advanced, it is likely to be used alongside human


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

    ].

     I

    'm

     a

     [

    work

     experience

    ]

     [

    title

    ]

     and

     I

     specialize

     in

     [

    special

    ization

    ].

     I

    've

     always

     had

     a

     strong

     passion

     for

     [

    interest

     or

     hobby

    ],

     and

     I

     strive

     to

     [

    positive

     action

     or

     goal

    ].

     Whether

     it

    's

     just

     for

     fun

     or

     to

     help

     others

    ,

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     develop

    .

     I

    'm

     a

     [

    professional

    ]

     and

     I

    'm

     here

     to

     make

     a

     difference

     in

     the

     world

    .

     What

    's

     your

     name

    ,

     and

     what

    's

     your

     profession

    ?

     I

    'm

     [

    Name

    ].

     I

    'm

     a

     [

    work

     experience

    ]

     [

    title

    ]

     and

     I

     specialize

     in

     [

    special

    ization

    ].

     I

    've

     always

     had

     a

     strong

     passion

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     is

     the

     largest

     and

     most

     influential

     political

    ,

     economic

    ,

     and

     cultural

     center

     in

     the

     country

    .

     It

     is

     known

     for

     its

     history

    ,

     art

    ,

     and

     architecture

    ,

     and

     is

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

     The

     city

     is

     also

     the

     center

     of

     many

     important

     cultural

     institutions

     and

     events

    ,

     including

     the

     World

     Trade

     Center

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Mus

    ée

     Rod

    in

    .

     Additionally

    ,

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     city

     in

     its

     own

     right

    ,

     known

     for

     its

     art

     and

     culture

     and

     its

     food

     and

     wine

    .

     
    


    (Note

    :

     This

     answer

     is

     based

     on

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     will

     likely

     evolve

     at

     a

     rapid

     pace

    ,

     driven

     by

     new

     developments

     in

     computing

    ,

     data

     science

    ,

     and

     machine

     learning

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

    -powered

     diagnostics

     and

     treatment

     planning

     will

     become

     more

     advanced

    ,

     leading

     to

     better

     outcomes

     for

     patients

    .

     AI

     will

     also

     be

     used

     to

     assist

     doctors

     in

     identifying

     patterns

     and

     predicting

     patient

     outcomes

    .
    


    2

    .

     More

     personalized

     healthcare

    :

     AI

     will

     be

     used

     to

     analyze

     patient

     data

     to

     identify

     personalized

     treatment

     plans

    ,

     leading

     to

     better

     outcomes

     and

     reduced

     healthcare

     costs

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     finance

    :

     AI

     will

     be

     used

     to

     analyze

     market

     data

     and

     detect

     trends

    ,

     leading

     to

     better

    



```python
llm.shutdown()
```
