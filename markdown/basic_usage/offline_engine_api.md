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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.07it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.57it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.21it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.69it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.84it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.84it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.84it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.84it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.84it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.84it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.84it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.77 GB):   3%|▎         | 2/58 [00:00<00:04, 11.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.77 GB):   3%|▎         | 2/58 [00:00<00:04, 11.20it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.77 GB):   3%|▎         | 2/58 [00:00<00:04, 11.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.05it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.76 GB):   7%|▋         | 4/58 [00:00<00:03, 14.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.75 GB):   7%|▋         | 4/58 [00:00<00:03, 14.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.74 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=72.74 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.74 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.73 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.73 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.73 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.73 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.67it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=72.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.62 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  31%|███       | 18/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=960 avail_mem=72.27 GB):  31%|███       | 18/58 [00:00<00:01, 29.57it/s] Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  31%|███       | 18/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.42it/s]

    Capturing num tokens (num_tokens=768 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=416 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.03it/s]

    Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=224 avail_mem=72.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=208 avail_mem=71.94 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=192 avail_mem=72.19 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.05it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.19 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=160 avail_mem=72.18 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=160 avail_mem=72.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.61it/s]Capturing num tokens (num_tokens=144 avail_mem=72.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.61it/s]Capturing num tokens (num_tokens=128 avail_mem=72.16 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.61it/s]Capturing num tokens (num_tokens=112 avail_mem=72.16 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.61it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.15 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.61it/s] Capturing num tokens (num_tokens=96 avail_mem=72.15 GB):  81%|████████  | 47/58 [00:01<00:00, 30.68it/s]Capturing num tokens (num_tokens=80 avail_mem=72.15 GB):  81%|████████  | 47/58 [00:01<00:00, 30.68it/s]Capturing num tokens (num_tokens=64 avail_mem=72.14 GB):  81%|████████  | 47/58 [00:01<00:00, 30.68it/s]Capturing num tokens (num_tokens=48 avail_mem=72.13 GB):  81%|████████  | 47/58 [00:01<00:00, 30.68it/s]Capturing num tokens (num_tokens=32 avail_mem=72.13 GB):  81%|████████  | 47/58 [00:01<00:00, 30.68it/s]Capturing num tokens (num_tokens=32 avail_mem=72.13 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=28 avail_mem=72.12 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=24 avail_mem=72.11 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.71it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.10 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=16 avail_mem=72.10 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=16 avail_mem=72.10 GB):  95%|█████████▍| 55/58 [00:01<00:00, 31.95it/s]Capturing num tokens (num_tokens=12 avail_mem=72.09 GB):  95%|█████████▍| 55/58 [00:01<00:00, 31.95it/s]Capturing num tokens (num_tokens=8 avail_mem=72.08 GB):  95%|█████████▍| 55/58 [00:01<00:00, 31.95it/s] Capturing num tokens (num_tokens=4 avail_mem=72.08 GB):  95%|█████████▍| 55/58 [00:01<00:00, 31.95it/s]Capturing num tokens (num_tokens=4 avail_mem=72.08 GB): 100%|██████████| 58/58 [00:01<00:00, 30.57it/s]


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
    Generated text:  August. I am a computer science graduate student and I like to create software for my friends. I have a computer at home and sometimes I get bored of using my computer. Please suggest me some interesting ways to use my computer.
    As an AI language model, I don't have personal preferences or a computer at home, but I can suggest some interesting ways to use your computer based on my knowledge:
    
    1. Create educational materials: Use your computer to create documents, presentations, and other educational materials for your friends or family.
    
    2. Conduct webinars: Use your computer to host webinars or live video meetings with other people. You can
    ===============================
    Prompt: The president of the United States is
    Generated text:  a type of _____ in our country.
    A. Government official
    B. Corporate leader
    C. Military officer
    D. College teacher
    Answer: A
    
    The reason why a company cannot pay the fixed asset acquisition costs directly is ____.
    A. It's too much money
    B. The company doesn't have sufficient funds
    C. There's an insufficient balance in the company's bank account
    D. The company doesn't have a fixed asset acquisition plan
    Answer: B
    
    When the US dollar strengthens against the renminbi, it will lead to ____.
    A. An increase in the demand for US dollars, a decrease in
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Madrid
    C. London
    D. Athens
    Answer:
    
    A
    
    In Traditional Chinese Medicine, what is the primary treatment method for a disease of spleen and stomach qi deficiency?
    A. Promoting Qi movement
    B. Warming and tonifying the spleen
    C. Activating blood circulation
    D. Strengthening the spleen
    E. Activating yang
    Answer:
    
    B
    
    In Traditional Chinese Medicine diagnosis, what does the term "Qi deficiency" refer to?
    A. Qi fails to circulate and move
    B. Qi is unable to control yin and yang
    C.
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain. There are many angles to consider when evaluating an AI technology, which can help you make informed decisions. Here, we’ll discuss 5 key factors to consider before making a decision on whether to invest in AI for your organization. Most importantly, remember that there are no definitive answers to these questions. You should always look at the above factors and consider other options before making a final decision.
    
    1. Risks and Benefits of AI
    
    The risks and benefits of AI can be complex to understand and assess. Let’s take a closer look at these factors and their potential effects on your organization.
    
    AI has the potential to revolutionize how


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] with a passion for [Interest]. I'm always looking for [Challenge] and I'm always eager to learn new things. I'm [Personality] and I'm always ready to help others. I'm [Hobby] and I love [Favorite Activity]. I'm [Favorite Food] and I enjoy [Favorite Movie]. I'm [Favorite Book] and I love [Favorite Music]. I'm [Favorite Sport] and I'm a [Favorite Sportswoman]. I'm [Favorite Animal] and I love [Favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. Paris is also a popular tourist destination and is home to many world-renowned museums, art galleries, and restaurants. The city is known for its fashion industry and is home to many famous fashion houses such as Chanel, Dior, and Louis Vuitton. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is a popular destination for tourists and locals alike, and is a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. AI in finance: AI is already being used in finance to improve fraud detection, risk management, and investment decision
    


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
    Generated text:  [insert name]. I am a [insert occupation or profession] who has been in the industry for [insert number of years] years. My [insert one or two things that you are passionate about or enjoy] is [insert something interesting about yourself]. I am constantly learning new things, growing and becoming stronger with every passing day. I am confident and have a strong work ethic. I believe that passion, hard work, and perseverance are the keys to success. So, my [insert your name] is excited to be here today and contribute to the industry in any way that I can. Goodbye. [insert your name].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of lights and the symbol of the French nation. Located in the south of the country, it is known for its historic landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Arc de Triomphe. Paris is a cultural hub with a vibrant nightlife, numerous museums, and a rich history dating back over 2,500 years. It is a major tourist destination and a popular destination for business and leisure. Paris is also home to the Eiffel Tower, the Louvre, and the Notre-Dame Cathedral. According to the 2019 census, Paris had a population
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with many exciting developments on the horizon. Here are some potential trends that could shape the future of AI:
    
    1. Increasingly personalized AI: As AI becomes more capable, it will be able to learn from and improve on personalization. AI will be able to analyze vast amounts of data and make informed decisions that are tailored to individual users. This could lead to highly personalized experiences, such as speech-to-text and language translation services.
    
    2. Enhanced autonomy and interaction with users: With the development of stronger AI, users may be able to interact more directly with AI systems. This could lead to more natural and intuitive interactions, as AI


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

    type

     of

     character

    ]

     with

     [

    career

     goals

    ]

     in

     mind

    .

     I

    've

     been

     studying

     [

    career

     path

    ]

     for

     [

    number

     of

     years

    ]

     and

     have

     recently

     graduated

     with

     a

     [

    degree

     or

     qualification

    ].

     I

    'm

     confident

     in

     my

     abilities

     and

     enthusiastic

     about

     my

     career

     goals

    ,

     which

     include

     [

    list

     of

     career

     goals

    ].

     I

     thrive

     in

     creative

     and

     critical

     thinking

    ,

     as

     well

     as

     active

     listening

     and

     empathy

    .

     My

     goal

     is

     to

     [

    mention

     a

     specific

     goal

    ,

     such

     as

     a

     career

     path

    ,

     education

    ,

     or

     personal

     development

    ].

     I

    'm

     also

     a

     [

    general

     interest

    ]

     like

     [

    mention

     an

     interest

    ,

     such

     as

     sports

    ,

     music

    ,

     or

     travel

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     renowned

     for

     its

     rich

     history

     and

     culture

    ,

     which

     is

     evident

     in

     its

     museums

    ,

     art

     galleries

    ,

     and

     bustling

     art

     scene

    .

     The

     city

     also

     has

     a

     vibrant

     nightlife

     scene

     and

     is

     home

     to

     numerous

     world

    -ren

    owned

     restaurants

     and

     bars

    .

     Paris

     is

     a

     cultural

     capital

     that

     plays

     a

     significant

     role

     in

     the

     country

    ’s

     economy

     and

     history

    .

     It

    's

     an

     unforgettable

     city

     to

     visit

     for

     any

     traveler

    .

     What

     is

     the

     capital

     of

     France

    ?


    Paris

    .

     The

     capital

     of

     France

     is

     Paris

    .

     Known

     for

     iconic

     landmarks

     like

     Notre

    -D

    ame

     Cathedral

    ,

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

     and

     possibilities

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Natural

     language

     processing

    :

     In

     the

     coming

     years

    ,

     we

     will

     see

     significant

     advancements

     in

     natural

     language

     processing

    ,

     which

     will

     enable

     AI

     to

     understand

     and

     respond

     to

     human

     language

     in

     new

     ways

    .

     This

     will

     enable

     robots

     and

     machines

     to

     have

     more

     human

    -like

     intelligence

    .
    


    2

    .

     Machine

     learning

    :

     Machine

     learning

     is

     a

     subset

     of

     AI

     that

     involves

     training

     algorithms

     to

     learn

     from

     data

     and

     improve

     their

     performance

     over

     time

    .

     It

     is

     currently

     used

     in

     areas

     such

     as

     image

     recognition

    ,

     speech

     recognition

    ,

     and

     natural

     language

     processing

    .
    


    3

    .

     Human

    -com

    puter

     interaction

    :

     AI

     will

     continue

     to

     evolve

     in

     terms

     of

     how

     it

     interacts

    



```python
llm.shutdown()
```
