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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.92it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.07 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.06 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.06 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.06 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.05 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.05 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=61.05 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.05 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.05 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.04 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.04 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.04 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.03it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.03it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.03it/s]Capturing num tokens (num_tokens=960 avail_mem=61.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.03it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=61.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=896 avail_mem=61.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=832 avail_mem=61.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=768 avail_mem=61.01 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=704 avail_mem=61.01 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=640 avail_mem=61.01 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=640 avail_mem=61.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=576 avail_mem=61.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=512 avail_mem=60.99 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=480 avail_mem=61.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=448 avail_mem=61.00 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=416 avail_mem=61.00 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.67it/s]

    Capturing num tokens (num_tokens=416 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=384 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=352 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=320 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=288 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=256 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=256 avail_mem=60.99 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=240 avail_mem=60.98 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=224 avail_mem=60.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=208 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=192 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=176 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.23it/s]

    Capturing num tokens (num_tokens=176 avail_mem=60.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=160 avail_mem=60.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=144 avail_mem=60.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=128 avail_mem=60.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=112 avail_mem=60.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=96 avail_mem=60.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.00it/s] Capturing num tokens (num_tokens=96 avail_mem=60.96 GB):  81%|████████  | 47/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=80 avail_mem=60.95 GB):  81%|████████  | 47/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=64 avail_mem=60.95 GB):  81%|████████  | 47/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=48 avail_mem=60.94 GB):  81%|████████  | 47/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=32 avail_mem=60.94 GB):  81%|████████  | 47/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=28 avail_mem=60.94 GB):  81%|████████  | 47/58 [00:01<00:00, 45.18it/s]

    Capturing num tokens (num_tokens=28 avail_mem=60.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=24 avail_mem=60.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=20 avail_mem=60.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=16 avail_mem=60.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=12 avail_mem=60.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=8 avail_mem=60.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.23it/s] Capturing num tokens (num_tokens=8 avail_mem=60.92 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=4 avail_mem=60.92 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=4 avail_mem=60.92 GB): 100%|██████████| 58/58 [00:01<00:00, 40.22it/s]


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
    Generated text:  Sam, I'm 25 years old, and I live in the USA. I can't sleep at night. I don't have enough sleep, and I have trouble focusing. I want to find a good medicine to help me sleep, but I'm worried that I might have a problem with too much caffeine. 
    
    Can you suggest a medicine that will help me get better sleep? I am concerned that too much caffeine might be causing me sleep problems. 
    
    I also need to find a treatment for my problem with caffeine. Here are some options:
    1. Caffeine replacement therapy
    2. Caffeine withdrawal
    3.
    ===============================
    Prompt: The president of the United States is
    Generated text:  retiring. As part of his retirement plan, he has decided to invest a certain amount of money in a bank account. The bank offers a compound interest rate that doubles the amount invested after one year and triples the amount invested after two years. If the president decides to invest $10,000 now, what will be the total amount of money in the bank account after 3 years?
    To determine the total amount of money in the bank account after 3 years, we need to consider the compound interest rate for each year. The bank offers two different interest rates: one that doubles the amount after one year and another that triples
    ===============================
    Prompt: The capital of France is
    Generated text: : Paris
    You are a world class trivia AI - so don't worry to answer this question between me and a random magician. My sequence of questions:
    
    1. What is the smallest country in Europe by area?
    2. What is the capital of France?
    3. What is the largest country in Europe by area?
    4. What is the name of the capital of the Netherlands?
    5. What is the name of the capital of Belgium?
    6. Which country is the smallest by area?
    7. What is the capital of Switzerland?
    8. What is the name of the capital of Italy?
    9. What is the capital of Lie
    ===============================
    Prompt: The future of AI is
    Generated text:  shaping the landscape of how we understand and improve the quality of our lives. AI is increasingly being used to enhance decision-making processes, improve healthcare outcomes, and enable advanced technologies like autonomous vehicles. As we look towards the future of AI, there are some emerging trends that could have significant implications for the field and its applications.
    One of the most significant trends in AI is the increasing use of deep learning and neural networks. Deep learning is a type of machine learning that involves modeling complex data structures using multiple layers of neural networks. This allows AI models to learn and improve from vast amounts of data, making them more accurate and efficient at solving complex problems


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests]. I like to read, watch movies, and listen to music. I'm always looking for new experiences and adventures. What's your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is a bustling metropolis with a diverse population and a rich cultural heritage. The city is home to many famous landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. Paris is also known for its fashion industry, with many famous designers and boutiques. The city is a popular tourist destination and a major economic center in France. It is a city of contrasts, with its modern skyscrapers and historic architecture blending seamlessly. Paris is a city of art, culture, and history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to play an increasingly important role in areas such as healthcare, finance, and government, as it can help to automate and streamline processes, improve accuracy and efficiency, and provide insights that are difficult to obtain through traditional methods. However, there are also potential risks and challenges associated with the use of AI, such as the potential for job displacement and ethical concerns around data privacy and bias.
    


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
    Generated text:  [Name] and I'm a [Title] with [Number] years of experience in the industry. I specialize in [Area of Expertise], and I have [Number] years of experience in this field.
    
    [Name], I'm an [X] year old, [X] year old [X] year old, and [X] year old [X] year old. I'm a full-time [Hourly Rate] or [Salary] employee, with [X] years of experience in [Industry]. My experience includes [Number of Projects or Roles]. I've had the pleasure of working with a diverse range of clients
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Paris is known for its rich history, iconic landmarks, and vibrant culture, making it a popular tourist destination. The city has a rich cultural heritage dating back to ancient times, and many French people live there with their families. It's also a major transportation hub, with several airports serving as major hubs for the nation. Paris has a diverse population, with many people from around the world living and working in the city. 
    
    Paris is also home to many world-renowned museums, including the Louvre and the Musée d'Orsay. It's also a popular destination for fashion and art lovers, with several fashion houses
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a variety of trends, including the continued growth and integration of machine learning and deep learning algorithms. There may also be an increased emphasis on ethical considerations and accountability in AI development and deployment. Additionally, there may be a focus on developing more advanced natural language processing and computer vision technologies to help with tasks such as language translation, image recognition, and autonomous driving. Finally, there may be a continued shift towards more distributed and distributed processing architectures to improve the efficiency and scalability of AI systems. Overall, the potential for AI to bring about significant changes to society and the economy is likely to continue to evolve and evolve in the years ahead.


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

    Role

    ]

     with

     [

    Experience

    ]

     years

     of

     experience

     in

     [

    Field

    ].

     I

    'm

     [

    Age

    ]

     years

     old

     and

     [

    Occup

    ation

    ]

     and

     have

     a

     [

    N

    ost

    alg

    ic

     or

     Modern

     appearance

    ].

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    ,

     and

     my

     goal

     is

     to

     [

    Goals

    ].

     I

     have

     a

     [

    Skill

    set

    ]

     and

     am

     [

    L

    oyal

     or

     Casual

    ]

     with

     [

    People

    ].

     Please

     feel

     free

     to

     ask

     me

     anything

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     the

     information

     you

    're

     looking

     for

    .

     Good

    night

    !

     

    🌟

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Not

    ably

    ,

     Paris

     is

     the

     world

    ’s

     most

     populous

     city

    ,

     with

     an

     estimated

     population

     of

     

    2

    .

    2

     million

     as

     of

     

    2

    0

    2

    1

    ,

     making

     it

     the

     second

    -largest

     city

     in

     the

     world

    ,

     after

     Mumbai

    .

     The

     city

     has

     a

     population

     of

     over

     

    2

    2

     million

     people

    ,

     and

     the

     population

     density

     is

     about

     

    3

    3

    4

     people

     per

     square

     kil

    ometer

    .

     It

     is

     also

     the

     third

    -largest

     city

     by

     area

    ,

     with

     an

     area

     of

     

    7

    8

    8

    .

    2

     square

     kilometers

    ,

     which

     is

     more

     than

     two

     times

     the

     size

     of

     the

     United

     States

    .

     Paris

     has

     a

     population

     of

     over

     

    2

    2

     million

     people

    ,

     and

     the

     population

     density

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     progress

     in

     several

     key

     areas

    ,

     including

    :
    


    1

    .

     Machine

     Learning

    :

     As

     the

     power

     of

     machine

     learning

     continues

     to

     grow

    ,

     more

     complex

     algorithms

     will

     be

     developed

    ,

     leading

     to

     more

     accurate

     and

     sophisticated

     AI

     systems

    .
    


    2

    .

     Cyber

    security

    :

     With

     the

     increasing

     amount

     of

     data

     being

     generated

     and

     used

     in

     AI

     applications

    ,

     there

     is

     a

     growing

     need

     for

     advanced

     cybersecurity

     measures

     to

     protect

     against

     potential

     threats

    .
    


    3

    .

     Eth

    ical

     considerations

    :

     As

     AI

     continues

     to

     develop

     and

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increasing

     scrutiny

     around

     its

     ethical

     implications

    ,

     such

     as

     privacy

    ,

     bias

    ,

     and

     accountability

    .
    


    4

    .

     Human

    -

    robot

     interaction

    :

     The

     integration

     of

    



```python
llm.shutdown()
```
