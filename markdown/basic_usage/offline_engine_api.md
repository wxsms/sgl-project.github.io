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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:47,  3.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:47,  3.99s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.82it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 13.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 13.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:04, 13.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:04, 11.13it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:04, 11.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:04, 11.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:04, 11.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.93it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.17it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.17it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.17it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.17it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.17it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.17it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.09it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.93it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.93it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.93it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.93it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.93it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.93it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.14it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.14it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.14it/s]

    Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.14it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.14it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.14it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.89it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 36.54it/s]


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
    Generated text:  Frank, I'm 17 years old, and I'm a student. I'm very happy to have met you.
    A. Do you have a new school? B. No, I'm not happy.
    Answer:
    
    A
    
    When editing a document in Word, the "Review" tab can ____.
    A. open the "Document Review" dialog box
    B. display the "Review" status of the document
    C. set the page numbers
    D. display the "Review" status of the document and set the page numbers
    Answer:
    
    A
    
    The following is a choice question from the junior high school chemistry exam in China
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a small town. The president is currently at a conference center that is 300 feet away from the town center. He will stop by the town hall for a brief visit, which will take him 10 minutes. If he spends an additional 20% of his time at the town hall, how much time in total does the president spend on his visit to the town hall? To determine the total time the president spends on his visit to the town hall, we need to follow these steps:
    
    1. Calculate the total time spent on the visit to the town hall.
    2. Add the time spent at the town
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Italy is Rome. The capital of Italy is Rome. What is the capital of France?
    The answer to this problem is Paris. 
    
    To provide a more detailed explanation:
    
    1. The question states that the capital of France is Paris, which is the capital of France.
    
    2. The capital of Italy is Rome, as the question directly states that the capital of Italy is Rome.
    
    3. Given the information provided, it is clear that both Paris and Rome are indeed the capitals of their respective countries.
    
    Therefore, the capital of France is Paris. 
    
    It's important to note that the question is asking about a single
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of those who have the skills and passion to bring it to life. In the world of artificial intelligence, there are a variety of skills and abilities that are necessary to successfully create and develop AI systems. These skills can include technical knowledge, creativity, problem-solving skills, and communication skills. Additionally, there are many ways to learn and develop these skills, both online and offline.
    To begin your journey towards developing your skills in AI, there are a few steps you can take to get started. First, you should consider what you want to do with AI and what kind of role it could play in your future. This could be


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. The city is known for its cuisine, fashion, and art scene. It is also home to the French Parliament and the French National Library. Paris is a vibrant and dynamic city with a rich cultural heritage and is a popular tourist destination. The city is known for its iconic landmarks, cuisine, fashion, and art scene. It is also home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased automation and efficiency: AI is expected to continue to automate many tasks, reducing the need for human labor and increasing productivity. This could lead to the development of new types of AI, such as superintelligent machines that can perform tasks that require human intelligence, such as creative problem-solving or decision-making.
    
    2. Enhanced human-computer interaction: AI is expected to continue to improve the way humans interact with machines, making it easier and more efficient to control and operate AI systems. This could lead to the development of new types of AI, such as those that can understand and
    


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
    Generated text:  [Your Name], and I'm [Your Profession]. I have been working in this field for [X] years, but I'm not quite as experienced as some of my colleagues. However, I'm always eager to learn and improve, and I believe that being a true professional takes hard work, dedication, and a positive attitude. I'm always looking for ways to grow and expand my capabilities, and I'm looking forward to continuing my journey of professional development. Thank you for considering me for an interview. I look forward to hearing from you. 
    
    Note: Replace [Your Name] with your real name or your chosen professional title to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Question: How is Paris known for its cuisine? Paris is known for its cuisine for its exquisite dishes and its famous bistros and night markets. Paris is home to many fine restaurants, and the city's culinary scene is celebrated around the world. For example, the famous Escadais, a steakhouse near the Eiffel Tower, is widely recognized for its beautifully decorated interiors, and the French pavé, a popular dessert, is a classic French breakfast. 
    
    Here's a more detailed breakdown of Paris's cuisine:
    
    1. French cuisine: Paris is known for its French cuisine, which is characterized by its precision and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a number of trends that are currently under investigation by researchers and industry. Here are some potential future trends in AI:
    
    1. Increased focus on ethical AI: As more data is generated by AI systems, the need for ethical AI is becoming more important. This may lead to increased regulations and standards, such as GDPR for data privacy and AI accountability standards for AI systems.
    
    2. Enhanced transparency and explainability of AI: As AI systems become more complex and nuanced, there may be an increased need for transparency and explainability. This may lead to greater use of AI to provide insights into human behavior and decision-making processes.
    
    


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

     __

    ________

    .

     I

    've

     always

     loved

     words

     and

     now

     I

    'm

     a

     words

    mith

    .

     I

    'm

     a

     high

     school

     student

     who

     started

     learning

     programming

     languages

     as

     a

     hobby

    .

     My

     favorite

     hobby

     is

     reading

    ,

     and

     I

     love

     to

     help

     others

     with

     their

     writing

     and

     coding

     questions

    .

     My

     hobbies

     include

     listening

     to

     music

    ,

     writing

    ,

     playing

     guitar

    ,

     and

     cooking

    .

     I

     also

     love

     to

     travel

     to

     different

     parts

     of

     the

     world

     and

     learn

     about

     different

     cultures

     and

     cuis

    ines

    .

     I

    'm

     a

     self

    -pro

    claimed

     "

    words

    mith

    "

     and

     enjoy

     spending

     my

     time

     exploring

     the

     world

     and

     reading

     about

     new

     words

     and

     phrases

    .

     If

     you

    're

     interested

    ,

     you

     can

     learn

     more

     about

     me

     by

     clicking

     the

     link

     below

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    France

    's

     capital

     city

    ,

     Paris

    ,

     is

     renowned

     for

     its

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

    ,

     as

     well

     as

     its

     culinary

     scene

     and

     rich

     history

    .

     It

    's

     also

     known

     for

     its

     fashion

     industry

    ,

     particularly

     its

     cout

    ur

    ier

     labels

     and

     fashion

    -forward

     fashion

    ista

     culture

    .

     The

     city

    's

     annual

     Lou

    vre

     Museum

     exposition

     and

     opera

     houses

     are

     also

     significant

     tourist

     attractions

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     continued

     innovation

    ,

     growth

    ,

     and

     integration

     with

     other

     technologies

    .

     Here

     are

     some

     possible

     trends

     that

     could

     emerge

    :
    


    1

    .

     Deep

     Learning

    :

     Deep

     learning

     is

     an

     area

     of

     AI

     research

     that

     focuses

     on

     using

     neural

     networks

     to

     perform

     tasks

     that

     traditional

     machine

     learning

     algorithms

     are

     not

     good

     at

    .

     This

     could

     lead

     to

     new

     applications

     in

     areas

     such

     as

     computer

     vision

    ,

     natural

     language

     processing

    ,

     and

     recommendation

     systems

    .
    


    2

    .

     Explain

    ability

    :

     As

     AI

     systems

     become

     more

     complex

     and

     integrated

     with

     other

     technologies

    ,

     it

     is

     likely

     that

     they

     will

     become

     more

     explain

    able

     to

     humans

    .

     This

     could

     lead

     to

     more

     robust

     security

     measures

     and

     better

     transparency

     in

     decision

    -making

     processes

    .
    


    3

    .

     Personal

    ization

    



```python
llm.shutdown()
```
