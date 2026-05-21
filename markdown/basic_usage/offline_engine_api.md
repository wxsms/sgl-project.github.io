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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:00,  4.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:00,  4.23s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:00,  4.23s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:00,  4.23s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:00,  4.23s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.57it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]

    Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 34.80it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 34.80it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 34.80it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 34.80it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 34.80it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 34.80it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 34.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.51it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.97it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.16it/s]Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.16it/s] Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.16it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.16it/s]

    Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.16it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.16it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.43it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.43it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.43it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.43it/s]

    Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.43it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.43it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.43it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.87it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.87it/s]Capturing num tokens (num_tokens=208 avail_mem=73.91 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.87it/s]Capturing num tokens (num_tokens=192 avail_mem=73.91 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.87it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.87it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.87it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.87it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.83it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.83it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.83it/s]

    Capturing num tokens (num_tokens=96 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.83it/s] Capturing num tokens (num_tokens=80 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.83it/s]Capturing num tokens (num_tokens=64 avail_mem=73.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.83it/s]Capturing num tokens (num_tokens=64 avail_mem=73.52 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.67it/s]Capturing num tokens (num_tokens=48 avail_mem=72.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.67it/s]Capturing num tokens (num_tokens=32 avail_mem=72.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.67it/s]Capturing num tokens (num_tokens=28 avail_mem=72.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.67it/s]Capturing num tokens (num_tokens=24 avail_mem=72.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.67it/s]Capturing num tokens (num_tokens=20 avail_mem=72.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.67it/s]Capturing num tokens (num_tokens=20 avail_mem=72.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.76it/s]Capturing num tokens (num_tokens=16 avail_mem=72.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.76it/s]Capturing num tokens (num_tokens=12 avail_mem=72.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.76it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.76it/s] Capturing num tokens (num_tokens=4 avail_mem=72.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.76it/s]Capturing num tokens (num_tokens=4 avail_mem=72.88 GB): 100%|██████████| 58/58 [00:01<00:00, 42.87it/s]


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
    Generated text:  Shaina. I am a 17 year old female, 5'3", and 130 lbs. I was diagnosed with a T417A mutation in my DNA. I was told this mutation affects my ability to use mitochondria, meaning that my mitochondria (the main energy source) will not produce energy in the same way as the body's other cells. I have been told to take oral hemicranium. I have been told that I need to have my liver removed. I am very conflicted, because my body is very good at using my mitochondria. I can't even write it
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to hire a candidate from a specific political party. The probability that the president will hire a candidate from a particular party is 0.5. What is the probability that the president will hire a candidate from either party, given that the probability of a candidate from one party winning is 0.6 and the probability of a candidate from the other party winning is 0.4?
    To determine the probability that the president will hire a candidate from either party, we need to consider two scenarios:
    
    1. The president hires a candidate from the party that has a winning probability of 0.6.
    2. The president hires a candidate
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which of the following is a correct statement about Paris? (　　)  
    A. The westernmost city of France  
    B. The southernmost city of France  
    C. The easternmost city of France  
    D. The northernmost city of France To determine the correct statement about Paris, we need to understand the geographical location of the capital of France. Paris is located in the heart of France, near the Mediterranean Sea. It is situated on the Atlantic coast, in the south-central part of France.
    
    Let's analyze each option:
    
    A. The westernmost city of France
    - Paris is actually the westernmost city of
    ===============================
    Prompt: The future of AI is
    Generated text:  very much dependent on the technology that is in use today, and it's imperative to ensure that the technologies that are utilized by the industry are of the highest standard to meet the evolving needs of the industry.
    The internet of things (IoT) is a new technology that is being adopted in many industries, and it's a part of the AI revolution. The IoT is a collection of electronic devices that can communicate with each other, and it connects every device on a network. The IoT is utilized in a variety of ways, including smart homes, smart cities, and smart manufacturing. The IoT is a highly scalable technology, and it can be


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new experiences and adventures. What's your favorite book or movie? I love [insert a short description of your favorite book or movie here]. I'm always looking for new ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. Its history dates back to the Roman Empire and has been a major center of French culture and politics for centuries. The city is also known for its fashion industry and is home to many famous fashion houses. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some possible future trends in AI:
    
    1. Increased automation: As AI becomes more advanced, it is likely to automate many tasks that are currently performed by humans, such as data analysis, decision-making, and routine maintenance. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. Improved privacy and security: As AI becomes more sophisticated, there will be an increased need for privacy and security measures to protect the data that is generated and processed by AI
    


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
    Generated text:  [Name], and I'm a [Role]. My favorite hobby is [My hobby]. I enjoy [Something I enjoy doing]. I love [My favorite thing to do]. I have always been [What you like to do best]. My favorite color is [My favorite color]. I have always been [What you like to do best] because [reason for liking it]. I am the [Your relationship with the character] of [Name], and we have [Number of years of shared experiences] years of friendship. I look forward to [Future plans]. I have a lot of [Number of hobbies you enjoy, if any], but
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. The city is the largest metropolitan area in the world and is home to the headquarters of many of the world’s most famous brands and corporations. It is also the center of France's economy, culture, and society. The city is known for its rich history, stunning architecture, and vibrant cultural scene. Paris is the third largest city in the world by population and is also one of the most popular tourist destinations in the world. The city has been a major center of government and politics for centuries and continues to be a center of global affairs. Its numerous monuments, including the Eiffel Tower
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  complex and multifaceted, but there are several potential trends that could shape the development of this technology in the years to come. Here are some potential areas of focus and expected developments:
    
    1. Increased integration with physical and biological systems: AI will likely become more integrated with various physical and biological systems, including cars, robots, and even individuals themselves. This could lead to more advanced self-driving cars that are capable of interpreting human emotions and surroundings, and more advanced prosthetics that are capable of interacting with the environment and performing tasks with greater accuracy and precision.
    
    2. Enhanced natural language processing: AI will continue to improve natural language processing, making


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

    career

    ]

     with

     over

     [

    number

    ]

     years

     of

     experience

    .

     I

     am

     passionate

     about

     [

    career

     goal

    ],

     and

     I

     am

     excited

     to

     help

     you

     achieve

     your

     goals

    .

     I

     have

     a

    [

    insert

     how

     many

     years

     of

     experience

     you

     have

    ]

     track

     record

     of

     [

    what

     you

     do

     for

     a

     living

    ].

     Let

    's

     work

     together

     to

     make

     [

    career

     goal

    ]

     a

     reality

    .

     [

    Add

     a

     question

     to

     start

     the

     conversation

    ]:

     What

     exc

    ites

     you

     about

     your

     current

     role

    ?

     [

    Add

     a

     statement

     to

     follow

    ]:

     I

     love

     [

    an

     emotion

    ]

     and

     am

     always

     learning

     new

     things

    .

     I

    'm

     also

     very

     organized

     and

     can

     manage

     multiple

     projects

     simultaneously

    .

     I

     have

     a

    
    
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

    ,

     and

     is

     home

     to

     numerous

     historic

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     known

     for

     its

     rich

     culture

     and

     vibrant

     population

    ,

     and

     is

     a

     cultural

     and

     artistic

     center

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     history

     and

     its

     role

     as

     the

     gateway

     to

     Europe

    .

     It

     has

     a

     diverse

     population

     of

     over

     

    7

     million

     people

    ,

     and

     has

     been

     the

     seat

     of

     government

     and

     leadership

     for

     over

     a

     millennium

    .

     The

     city

     is

     renowned

     for

     its

     fashion

    ,

     art

    ,

     and

     cuisine

    ,

     and

     continues

     to

     be

     an

     influential

     part

     of

     the

     French

     cultural

     identity

    .

     Paris

     is

     a

     vibrant

     and

     dynamic

     city

     with

     a

     rich

     cultural

     heritage

    .

     Its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     certainly

     going

     to

     be

     an

     exciting

     one

    .

     As

     AI

     continues

     to

     evolve

    ,

     there

     are

     many

     different

     trends

     and

     possibilities

     that

     will

     shape

     the

     technology

     and

     applications

     that

     we

     will

     see

     in

     the

     future

    .
    


    One

     of

     the

     most

     significant

     trends

     that

     we

     can

     expect

     to

     see

     is

     the

     increasing

     integration

     of

     AI

     into

     everyday

     life

    .

     AI

     is

     already

     being

     used

     in

     a

     wide

     range

     of

     applications

    ,

     such

     as

     facial

     recognition

    ,

     voice

     recognition

    ,

     and

     autonomous

     vehicles

    .

     As

     more

     and

     more

     technology

     integrates

     with

     AI

    ,

     we

     can

     expect

     to

     see

     even

     more

     sophisticated

     applications

     emerge

    .
    


    Another

     trend

     that

     is

     likely

     to

     become

     increasingly

     important

     in

     the

     future

     is

     the

     use

     of

     AI

     for

     self

    -driving

     cars

     and

     trucks

    .

     This

     is

    



```python
llm.shutdown()
```
