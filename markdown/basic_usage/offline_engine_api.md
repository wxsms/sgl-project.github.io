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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.73it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.45it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.12it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:03, 14.66it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.61it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 17.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 17.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 17.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:02, 17.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:02, 17.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.54it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.08it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.08it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.08it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.08it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.08it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.08it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.93it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.93it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.93it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.93it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.93it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.93it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=288 avail_mem=74.31 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.76it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=224 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=160 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.52it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.54it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=32 avail_mem=74.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=32 avail_mem=74.46 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=28 avail_mem=73.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=24 avail_mem=74.23 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.19it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.22 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=16 avail_mem=74.22 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=12 avail_mem=74.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=12 avail_mem=74.21 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.23it/s]Capturing num tokens (num_tokens=8 avail_mem=74.02 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.23it/s] Capturing num tokens (num_tokens=4 avail_mem=74.02 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.23it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.02 GB): 100%|██████████| 58/58 [00:01<00:00, 29.03it/s]


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
    Generated text:  John Smith and I am a software developer specializing in web development. I have been working as a freelancer for the past year and have developed several successful projects. I enjoy helping others learn and improving my skills and techniques. Can you tell me about your current projects and the skills you are proficient in? 
    
    Additionally, could you provide some guidance on how to improve my coding skills and how to approach software development in general? 
    
    Lastly, can you recommend any online courses or resources that I can use to improve my skills and knowledge in this field? 
    
    Thank you in advance for your help and guidance.
    
    Sure, I'd be happy to help with
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting two different countries. In the first country, he is in the capital city for 5 days and in the second country, he is in the capital city for 7 days. If he spends 3 days in each country, how many more days will he have left to visit countries in the United States?
    To determine how many more days the president of the United States will have left to visit countries in the United States, we need to calculate the total number of days he spends in the United States before he returns to visit another country.
    
    First, let's calculate the total number of days the president spends in the United States:
    -
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In order to move between Paris and London, you should use an airplane. On this airplane, you will be taken from Paris to London. What is the capital of France? To determine the capital of France, let's break down the information given:
    
    1. The capital of France is Paris.
    2. To move between Paris and London, you need to use an airplane.
    3. An airplane is an air transport vehicle used to transport people and goods.
    
    Given these points, the capital of France is Paris. Therefore, the answer is:
    
    \boxed{Paris}
    ===============================
    Prompt: The future of AI is
    Generated text:  the future of data, and the use of AI in data sets is one of the most important areas. Many companies are putting AI in data sets to gain insights into customer behavior, trends, and preferences. The data in AI can be thought of as a complex network that contains the data in a structured form. AI can analyze the data in this network to provide insights about customer behavior and preferences. This makes AI a powerful tool in the hands of businesses, as it can help them to make better decisions and make more informed decisions.
    
    One of the most important things about AI is that it can analyze complex data in a structured form. This makes


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I'm [height] inches tall and [weight] pounds. I have [number] years of experience in [industry]. I'm [interests and hobbies] and [personal interests]. I'm [what you like to do for fun]. I'm [what you like to do for work]. I'm [what you like to do for relaxation]. I'm [what you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and art scene. Paris is a vibrant and dynamic city with a diverse population and a strong sense of community. It is also home to many international organizations and institutions. The city is known for its annual Eiff
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This could lead to more sophisticated forms of AI that can learn from and adapt to human behavior and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use of AI.
    
    3. Increased use of AI in healthcare
    


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
    Generated text:  John. I'm a 30-year-old software engineer with a passion for technology and a keen interest in the latest trends in coding and design. My work has taken me to various cities around the world, honing my skills and learning from each new experience. I enjoy sharing my knowledge and insights with others and mentoring aspiring tech professionals. I believe in the power of technology to solve problems and make a positive impact on the world. Looking to the future, I'm excited to continue exploring new technologies and learning more about how they can be used to improve people's lives. Thanks for having me! How about you? Let's talk about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest city in France and the oldest continuously occupied city in the world.
    
    Note: I do not engage in amplification or elaboration. The statement is concise and informative about the capital of France. 
    
    The capital city of France is Paris, which is the largest city in France and the oldest continuously occupied city in the world. 
    
    The statement is a factual and concise overview of the capital city's location and historical significance, not intended for amplification or elaboration. The information provided is accurate and relevant to the general understanding of French urban geography. 
    
    Please let me know if you need any clarification or have additional questions
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by rapid growth, innovation, and integration of AI into various industries and applications. Some of the possible future trends in AI include:
    
    1. Increased automation and automation of tasks: AI is expected to continue to be applied in tasks that are repetitive, mundane, or low-level. The development of automation technologies, such as robotics, machine learning, and deep learning, is expected to increase the efficiency of many jobs and reduce the need for human labor.
    
    2. AI for healthcare: With the growing importance of AI in healthcare, we are likely to see more advanced AI solutions in areas like personalized medicine, diagnostics, and prevention.


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

     freelance

     writer

     and

     graphic

     designer

    .

     My

     work

     spans

     everything

     from

     blog

     posts

     to

     magazine

     covers

    ,

     and

     I

     have

     a

     knack

     for

     crafting

     compelling

     content

     that

     reson

    ates

     with

     readers

    .

     I

     enjoy

     working

     with

     clients

     to

     bring

     their

     ideas

     to

     life

    ,

     and

     I

    'm

     always

     looking

     for

     new

     and

     exciting

     projects

     to

     contribute

     to

     the

     creative

     industry

    .

     If

     you

    're

     looking

     to

     connect

     with

     someone

     with

     a

     creative

     mind

    ,

     I

    'd

     love

     to

     talk

     to

     you

     about

     your

     projects

     and

     aspirations

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ].

     (

    Note

    :

     This

     is

     a

     neutral

     self

    -int

    roduction

    ,

     but

     feel

     free

     to

     customize

     it

     to

     fit

     your

     persona

     or

     your

     intended

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     and

     it

     is

     a

     historic

     and

     culturally

     rich

     city

     that

     has

     a

     long

     and

     stor

    ied

     history

     dating

     back

     over

     

    5

    0

    0

     years

    .

     The

     city

     is

     known

     for

     its

     beautiful

     architecture

    ,

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     and

     its

     vibrant

     street

     life

    .

     Paris

     is

     also

     famous

     for

     its

     food

    ,

     particularly

     its

     famous

     French

     fries

    ,

     and

     its

     annual

     vibrant

     cultural

     events

     such

     as

     the

     Spring

     Festival

    ,

     the

     Mar

    ais

     Festival

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Overall

    ,

     Paris

     is

     a

     city

     that

     is

     a

     true

     marvel

     of

     human

     creativity

     and

     innovation

    ,

     and

     is

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     very

     different

     from

     its

     current

     state

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Improved

     Transparency

    :

     AI

     systems

     are

     becoming

     more

     transparent

     by

     default

    .

     This

     means

     that

     users

     can

     understand

     how

     an

     AI

     model

     arrived

     at

     a

     particular

     decision

     or

     recommendation

    .

     This

     level

     of

     transparency

     will

     help

     build

     trust

     between

     users

     and

     AI

     systems

    .
    


    2

    .

     Personal

    ization

    :

     AI

     will

     become

     more

     personalized

    ,

     so

     that

     users

     receive

     recommendations

     based

     on

     their

     past

     behavior

     and

     preferences

    .

     This

     will

     allow

     users

     to

     make

     more

     informed

     decisions

     and

     take

     advantage

     of

     personalized

     services

    .
    


    3

    .

     Autonomous

    :

     AI

     will

     become

     more

     autonomous

    ,

     with

     the

     ability

     to

     make

     decisions

     on

     its

     own

     without

     human

     intervention

    .

     This

     will

     allow

    



```python
llm.shutdown()
```
