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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:55,  4.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:55,  4.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:55,  4.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:55,  4.12s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:55,  4.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.67it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.43it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.74it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.42it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.91 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.91 GB):   3%|▎         | 2/58 [00:00<00:04, 12.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.88 GB):   3%|▎         | 2/58 [00:00<00:04, 12.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.88 GB):   3%|▎         | 2/58 [00:00<00:04, 12.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.88 GB):   7%|▋         | 4/58 [00:00<00:04, 12.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.39 GB):   7%|▋         | 4/58 [00:00<00:04, 12.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.22 GB):   7%|▋         | 4/58 [00:00<00:04, 12.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.21 GB):   7%|▋         | 4/58 [00:00<00:04, 12.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.21 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.21 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.21 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.31it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.20 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.20 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.20 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.20 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.19 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.19 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.19 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.80it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.18 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.80it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.18 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.18 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.18 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.74it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=72.17 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.17 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.15 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=960 avail_mem=72.17 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.05it/s] Capturing num tokens (num_tokens=896 avail_mem=72.16 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=832 avail_mem=72.16 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=768 avail_mem=72.16 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=704 avail_mem=72.15 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=704 avail_mem=72.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=640 avail_mem=72.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=576 avail_mem=72.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.50it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.14 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=480 avail_mem=72.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=448 avail_mem=72.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=448 avail_mem=72.15 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=416 avail_mem=72.15 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=384 avail_mem=72.15 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=352 avail_mem=72.14 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=320 avail_mem=72.13 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=288 avail_mem=72.13 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=288 avail_mem=72.13 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=256 avail_mem=72.13 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.60it/s]

    Capturing num tokens (num_tokens=240 avail_mem=72.13 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=224 avail_mem=72.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=208 avail_mem=72.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=192 avail_mem=72.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=192 avail_mem=72.12 GB):  71%|███████   | 41/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=176 avail_mem=72.11 GB):  71%|███████   | 41/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=160 avail_mem=72.11 GB):  71%|███████   | 41/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=144 avail_mem=72.11 GB):  71%|███████   | 41/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=128 avail_mem=72.11 GB):  71%|███████   | 41/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=112 avail_mem=72.10 GB):  71%|███████   | 41/58 [00:01<00:00, 43.73it/s]

    Capturing num tokens (num_tokens=112 avail_mem=72.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=96 avail_mem=72.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.77it/s] Capturing num tokens (num_tokens=80 avail_mem=72.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=64 avail_mem=72.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=48 avail_mem=72.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=32 avail_mem=72.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.77it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.09 GB):  88%|████████▊ | 51/58 [00:01<00:00, 21.46it/s]Capturing num tokens (num_tokens=28 avail_mem=72.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 21.46it/s]Capturing num tokens (num_tokens=24 avail_mem=72.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 21.46it/s]Capturing num tokens (num_tokens=20 avail_mem=72.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 21.46it/s]Capturing num tokens (num_tokens=16 avail_mem=72.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 21.46it/s]Capturing num tokens (num_tokens=16 avail_mem=72.07 GB):  95%|█████████▍| 55/58 [00:01<00:00, 23.97it/s]Capturing num tokens (num_tokens=12 avail_mem=72.07 GB):  95%|█████████▍| 55/58 [00:01<00:00, 23.97it/s]Capturing num tokens (num_tokens=8 avail_mem=72.07 GB):  95%|█████████▍| 55/58 [00:01<00:00, 23.97it/s] Capturing num tokens (num_tokens=4 avail_mem=72.06 GB):  95%|█████████▍| 55/58 [00:01<00:00, 23.97it/s]Capturing num tokens (num_tokens=4 avail_mem=72.06 GB): 100%|██████████| 58/58 [00:01<00:00, 29.33it/s]


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
    Generated text:  Iuriana and I'm from the United States. I started my language learning journey at an early age, but I was not the best student in my class. But, I went on to attend a good high school and used my language skills to get jobs. Although I wasn't the best student in my class, I was not so bad. 
    
    My background in high school and my lifelong passion for learning have given me a great deal of confidence. When I first entered my university, I was struggling with my first language. I found it hard to communicate with people who spoke different languages, and my grades were also lower than most.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a unique position in our democracy. They are the head of the executive branch of government. In addition to running the country, they are the commander-in-chief of the armed forces. They have the authority to issue the laws, grant pardons, and make other executive actions. They have no direct power to make policy. When it comes to policy, they are not only the only ones in the executive branch, but they are the only ones who can make policy without the legislative branch of government. The president’s power is not one of personal power, but rather, the power to determine policy. The president’s power to make policy is often
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. London
    B. Paris
    C. Moscow
    D. Athens
    Answer: B
    
    ______ is the foundation of the national spirit and the core of national spirit.
    A. Patriotism
    B. Love for peace
    C. Communism
    D. Americanization
    Answer: A
    
    When using the trial balance method to verify the accuracy of bookkeeping, if the total amount on the debit side of the 'Accounts Receivable' account is 25,000 yuan, and the total amount on the credit side of the same account is 27,000 yuan, then the amount
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but there are some promising developments in the field. Here are some of the most significant advancements in AI that are set to shape the industry in the coming years:
    
    1. Deep Learning: Deep learning is a subset of machine learning that involves the use of neural networks with multiple layers. It has been particularly effective in image and speech recognition, natural language processing, and other applications. The use of deep learning is expected to continue to grow as more data is available and more sophisticated models are developed.
    
    2. Natural Language Processing: Natural language processing is a subfield of AI that focuses on enabling computers to understand and interpret human language. This


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast who loves to explore the world and learn new things. I'm always looking for new experiences and adventures, and I'm always eager to try new things. I'm a [Favorite Thing] person who loves to laugh and have fun. I'm always looking for new ways to make people smile and have a good time. I'm a [Favorite Book or Movie] fan who loves to read and watch movies. I'm a [Favorite Sport] enthusiast who loves to play soccer and other team sports. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. It is also known for its fashion industry, with many famous designers and boutiques. Paris is a popular tourist destination and a cultural hub for France and the world. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations more effectively.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, privacy, and transparency.
    
    3. Greater use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more advanced, it is likely to be used in even more advanced ways, such as personalized
    


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
    Generated text:  [Name]. I have been in this world for [number] years now, and I'm currently seeking a way to balance my work and personal life. What can I do to help you in your search? Hello, my name is [Name]. I'm a [insert occupation here]. I've been in this world for [number] years now, and I'm currently seeking a way to balance my work and personal life. What can I do to help you in your search? [Name] currently seeks help in [insert problem or issue they are facing]. I can help you with [insert solution or method to address the issue].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement is: "Paris is the capital city of France." 
    
    This sentence is a concise fact about Paris, stating its role as the capital of the country. It is factual and can be easily communicated to a non-French speaker. 
    
    Other potential versions could include: "Paris is the largest city in France, and it serves as the capital of the country." or "Paris, being the largest city in France, also acts as the capital of the country." These versions provide additional context and detail about the significance of Paris as a city in France. 
    
    Both versions are accurate and concise factual statements about the capital city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and we can expect to see many changes and developments in the coming years. Here are some possible trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, but with further development, we can expect to see even more advancements in this field. AI-powered healthcare systems can analyze vast amounts of medical data and provide more accurate diagnoses, better treatment plans, and personalized care.
    
    2. AI in finance: AI is already being used in finance to improve trading algorithms, portfolio management, and fraud detection. With further development, we can expect to see even more automation and efficiency in finance,


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

    name

    ]

     and

     I

    'm

     a

     [

    age

    ]

     year

    -old

     [

    occupation

    ].

     I

    'm

     a

     [

    specific

     skill

     or

     talent

    ]

     expert

     in

     [

    describe

     your

     skill

     or

     talent

    ].

     What

     is

     your

     current

     project

     or

     goal

     at

     the

     moment

    ?
    


    [

    Name

    ],

     how

     have

     you

     been

    ?

     Have

     you

     worked

     on

     any

     notable

     projects

     lately

    ?

     [

    Name

    ],

     I

    'm

     excited

     to

     see

     what

     you

    're

     working

     on

    .

     What

    's

     your

     latest

     project

     or

     goal

    ?

     [

    Name

    ],

     I

    'd

     love

     to

     hear

     more

     about

     it

    .

     Thank

     you

    !

     [

    Name

    ],

     it

    's

     great

     to

     meet

     you

    !

     Let

    's

     get

     to

     know

     each

     other

     better

    .

     [

    Name

    ],

     this

     is

     [

    name

    ],

     I

     was

     wondering

    
    
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

     a

     historical

     center

     of

     art

    ,

     culture

    ,

     and

     intellectual

     life

    .

     It

     has

     a

     population

     of

     over

     

    2

     million

     and

     is

     the

     seat

     of

     government

    ,

     law

    ,

     diplomacy

    ,

     and

     finance

    .

     Paris

     is

     also

     known

     for

     its

     iconic

     landmarks

     like

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Pyramid

    .

     Its

     world

    -ren

    owned

     museums

    ,

     theaters

    ,

     and

     cafes

     are

     also

     a

     major

     draw

     for

     tourists

    .

     The

     city

     is

     considered

     one

     of

     the

     most

     cosm

    opolitan

     and

     culturally

     diverse

     cities

     in

     Europe

     and

     has

     a

     long

     and

     rich

     history

     dating

     back

     to

     ancient

     times

    .

     Paris

     is

     also

     known

     for

     its

     rich

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     number

     of

     potential

     trends

    ,

     including

    :
    


    1

    .

     Increased

     automation

     and

     precision

    :

     AI

     is

     expected

     to

     become

     more

     efficient

     and

     accurate

     in

     carrying

     out

     repetitive

     and

     routine

     tasks

    ,

     but

     it

     will

     also

     be

     necessary

     to

     develop

     more

     human

    -like

     capabilities

    ,

     such

     as

     creativity

     and

     decision

    -making

    ,

     to

     enable

     humans

     to

     better

     take

     control

     of

     complex

     tasks

    .
    


    2

    .

     Autonomous

     agents

    :

     The

     development

     of

     autonomous

     agents

     is

     expected

     to

     become

     more

     prevalent

    ,

     with

     the

     ability

     to

     perform

     tasks

     on

     a

     wide

     range

     of

     tasks

    ,

     including

     autonomous

     driving

    ,

     self

    -driving

     cars

    ,

     and

     the

     ability

     to

     repair

     and

     maintain

     human

    -made

     systems

    .
    


    3

    .

     Improved

     personal

    ization

     and

     inter

    activity

    :

     AI

     is

    



```python
llm.shutdown()
```
