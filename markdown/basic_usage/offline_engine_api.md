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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.21it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.85it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.85it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.85it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.85it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.85it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.85it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.85it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.85it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:05,  7.20it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:05<00:02, 12.12it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 18.86it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 26.68it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 36.15it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 36.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.45it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.45it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.49it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.49it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.67it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 35.63it/s]Capturing num tokens (num_tokens=384 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 35.63it/s]Capturing num tokens (num_tokens=352 avail_mem=76.66 GB):  55%|█████▌    | 32/58 [00:00<00:00, 35.63it/s]Capturing num tokens (num_tokens=320 avail_mem=76.66 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=288 avail_mem=76.66 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.63it/s]

    Capturing num tokens (num_tokens=288 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.99it/s]Capturing num tokens (num_tokens=256 avail_mem=76.64 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.99it/s]Capturing num tokens (num_tokens=240 avail_mem=76.56 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.99it/s]Capturing num tokens (num_tokens=224 avail_mem=76.56 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.99it/s]Capturing num tokens (num_tokens=208 avail_mem=76.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.99it/s]Capturing num tokens (num_tokens=192 avail_mem=76.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.99it/s]Capturing num tokens (num_tokens=192 avail_mem=76.15 GB):  71%|███████   | 41/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 36.62it/s]

    Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]

    Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.74it/s] Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 36.77it/s]


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
    Generated text:  Yorik and I am an intern at NetSuite. I am currently working on a project to develop a new application and to do so I need to understand the concept of a website. Can you please explain the difference between a website and a web application? Also, could you provide an example of a website and a web application? Lastly, could you explain the different types of web applications and how they work? Sure! A website is a web page that is hosted on a web server and is accessible over the internet. A web application, on the other hand, is a software program that runs on a web server and can be accessed
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful leader who has the ability to make decisions that will have a significant impact on the country. As a leader, the president must make difficult decisions to govern the country, including addressing issues such as global climate change, national security, and international relations.
    What is the role of the president of the United States?
    
    The role of the President of the United States is to serve as the head of state and government of the United States and represent the country on the world stage. The President serves a term of four years, during which they are expected to be a strong leader in guiding the country forward. They are also responsible for making important decisions
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Berlin
    C. Moscow
    D. London
    Answer:
    
    A
    
    Which of the following options does not belong to the category of literary forms?
    A. Drama
    B. Poetry
    C. Novel
    D. Novel
    Answer:
    
    D
    
    Which of the following options belongs to a literary form?
    A. Short story
    B. Novel
    C. Poetry
    D. Prose
    Answer:
    
    B
    
    In the application of the Freytag's Pyramid, what does the peak specifically refer to in terms of the development of a character's personality?
    A. The beginning of the character's life
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting, with new technologies and algorithms emerging all the time. In this chapter, we’ll cover some of the latest developments in artificial intelligence, including deep learning and quantum computing. We’ll also explore how these technologies are being used to solve real-world problems and improve our lives.
    What are the main applications of deep learning in AI?
    Deep learning is a subset of machine learning that focuses on training artificial neural networks to make predictions or classifications. The main applications of deep learning in AI are in image and speech recognition, natural language processing, and computer vision. These technologies are being used to improve the accuracy and efficiency of various applications, including self


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [job title] at [company name]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills. I'm always eager to learn and grow, and I'm always looking for new opportunities to contribute to the company. What's your favorite hobby or activity? I love [favorite hobby or activity]. I'm always looking for new experiences and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is the largest city in France and the third-largest in the world by population. The city is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub, with the Eiffel Tower serving as a symbol of the city's status as a global metropolis. The city is also known for its cuisine, with dishes like croissants, escargot,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in more complex and personalized ways, with
    


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
    Generated text:  [Your Name], and I'm a [职业/身份] who has been around [time since graduation] years. My love for writing has been a source of joy and inspiration for me throughout my life. I'm always eager to share my thoughts and experiences with others, and I'm eager to learn new things and discover new passions. I'm also proud to be part of a creative and forward-thinking community, and I'm excited to continue pushing the boundaries of what's possible with my writing. Thank you for taking the time to meet me. 
    
    Your name: [Your Name]
    Occupation: [Your Profession/Identity]
    Time
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the northwestern part of the country and known for its historical sites, vibrant culture, and world-class museums. The city is also one of the most visited cities in the world and hosts several annual festivals and events. Paris has a rich cultural history dating back to the Roman Empire, and today it is a city of art, science, and academia. In addition to its historical landmarks and museums, Paris is also a popular destination for tourists and residents alike due to its beautiful views and cultural attractions. The city has a population of over 2 million and is home to several famous landmarks such as the Eiffel Tower,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a wide range of technological advancements and changes in society. Here are some potential trends that are currently in the pipeline:
    
    1. Increased Human-AI Collaboration: One of the most significant trends in AI is the increased collaboration between humans and AI. As AI becomes more sophisticated, it is likely to be able to perform tasks that are currently handled by humans, such as playing chess, driving a car, and writing code. This will result in a more human-like interaction between humans and AI, with the latter being able to provide support and assistance.
    
    2. Enhanced AI Transparency: As AI systems become more advanced, they will likely be


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

    insert

     character

    's

     name

    ],

     and

     I

     am

     an

     [

    insert

     character

    's

     profession

     or

     role

    ]

     who

     is

     passionate

     about

     [

    insert

     a

     relevant

     topic

     such

     as

     [

    insert

     a

     hobby

    ,

     interest

    ,

     or

     achievement

     of

     the

     character

    's

     own

     choosing

    ],

     [

    insert

     an

     achievement

    ,

     achievement

    ,

     or

     accomplishment

     of

     a

     fictional

     character

     who

     is

     not

     their

     own

    ,

     if

     applicable

    ].

     I

     am

     a

     [

    insert

     a

     profession

    ]

     with

     a

     [

    insert

     a

     personal

     trait

     or

     trait

     that

     makes

     the

     character

     unique

    ]

     who

     works

     at

     [

    insert

     the

     name

     of

     the

     company

     or

     place

     where

     the

     character

     works

    ].

     I

     am

     constantly

     [

    insert

     a

     characteristic

     or

     trait

     that

     describes

     your

     character

    ,

     such

     as

     intelligence

    ,

     determination

    ,

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     towering

     E

    iff

    el

     Tower

     and

     beautiful

     gardens

    .

     It

     is

     home

     to

     many

     historical

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     is

     also

     renowned

     for

     its

     cultural

     and

     artistic

     scene

    ,

     with

     many

     renowned

     artists

    ,

     writers

    ,

     and

     musicians

     living

     and

     working

     in

     the

     city

    .

     The

     city

     is

     also

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

     Op

    éra

     Garn

    ier

    ,

     one

     of

     the

     oldest

     opera

     houses

     in

     the

     world

    .

     With

     its

     rich

     history

     and

     vibrant

     culture

    ,

     Paris

     is

     a

     city

     that

     continues

     to

     capt

    ivate

     visitors

     from

     all

     over

     the

     world

    .

     Would

     you

     like

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     includes

     several

     key

     trends

    :
    


    1

    .

     Increased

     sophistication

    :

     As

     technology

     continues

     to

     improve

    ,

     AI

     will

     become

     more

     sophisticated

    ,

     capable

     of

     performing

     tasks

     that

     were

     previously

     beyond

     its

     capabilities

    .

     This

     will

     lead

     to

     the

     development

     of

     more

     advanced

     applications

     and

     systems

    .
    


    2

    .

     Integration

     with

     natural

     language

     processing

     (

    N

    LP

    ):

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     be

     able

     to

     understand

     and

     process

     natural

     language

     inputs

    .

     This

     integration

     with

     N

    LP

     will

     make

     it

     possible

     for

     AI

     to

     generate

     more

     natural

    -s

    ounding

     and

     convers

    ational

     responses

    .
    


    3

    .

     Personal

    ization

    :

     AI

     will

     be

     able

     to

     learn

     from

     user

     data

     and

     provide

     personalized

     responses

     and

     recommendations

    .

     This

     will

     lead

     to

     a

     more

     effective

    



```python
llm.shutdown()
```
