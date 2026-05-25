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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:53,  4.09s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:53,  4.09s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:53,  4.09s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:53,  4.09s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:53,  4.09s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:04<00:00, 27.94it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 38.24it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 38.24it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 38.24it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 38.24it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.98 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.45 GB):   3%|▎         | 2/58 [00:00<00:03, 14.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.44 GB):   3%|▎         | 2/58 [00:00<00:03, 14.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.44 GB):   3%|▎         | 2/58 [00:00<00:03, 14.24it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.44 GB):   3%|▎         | 2/58 [00:00<00:03, 14.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.44 GB):   9%|▊         | 5/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.43 GB):   9%|▊         | 5/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.43 GB):   9%|▊         | 5/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.42 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.42 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.42 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.41 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.41 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.41 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.41 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.41 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.40 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.40 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.39 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.90it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.90it/s]Capturing num tokens (num_tokens=960 avail_mem=58.38 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.90it/s] Capturing num tokens (num_tokens=960 avail_mem=58.38 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=896 avail_mem=58.38 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=832 avail_mem=58.37 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=768 avail_mem=58.37 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=704 avail_mem=58.37 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=640 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.95it/s]Capturing num tokens (num_tokens=640 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=576 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=512 avail_mem=58.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.95it/s]

    Capturing num tokens (num_tokens=480 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=448 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=416 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=416 avail_mem=58.36 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=384 avail_mem=58.36 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=352 avail_mem=58.35 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=320 avail_mem=58.35 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=288 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=256 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=256 avail_mem=58.34 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=240 avail_mem=58.34 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=224 avail_mem=58.33 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.07it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.33 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=160 avail_mem=58.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=144 avail_mem=58.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=128 avail_mem=58.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.12it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.12it/s] Capturing num tokens (num_tokens=96 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:01<00:00, 39.83it/s]Capturing num tokens (num_tokens=80 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:01<00:00, 39.83it/s]Capturing num tokens (num_tokens=64 avail_mem=58.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.83it/s]Capturing num tokens (num_tokens=48 avail_mem=58.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.83it/s]Capturing num tokens (num_tokens=32 avail_mem=58.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.83it/s]Capturing num tokens (num_tokens=28 avail_mem=58.26 GB):  81%|████████  | 47/58 [00:01<00:00, 39.83it/s]Capturing num tokens (num_tokens=28 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=24 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.22it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=12 avail_mem=58.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=8 avail_mem=58.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.22it/s] Capturing num tokens (num_tokens=8 avail_mem=58.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.30it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.30it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.55it/s]


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
    Generated text:  Peter. I am a college student. I was born in a middle-class family. I went to school and then found a job, and now I am a college student. I have many hobbies. I have a boyfriend and a girlfriend, and we are happy. In addition, I have two children, and I am very devoted to them. When I was a child, I had a twin brother and sister, and they were very close to me. They often came to me for help. I was very happy. Now, after I have a brother and a sister, my twin brothers and sisters are missing. I wonder what happened to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the government of the country. What does the president do? The president of the United States is responsible for advising the president of the United States on the direction and policies of the country. The president also makes sure that the country's laws are being followed, and they also make sure that the country is safe and secure. The president is also responsible for convening the country's legislative and executive branches to make sure that the country is functioning properly. They also make sure that the country is funded and provides for the support of the people. The president has a lot of power to make decisions and directives. However, they
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris
    B: London
    C: Madrid
    D: Rome
    
    To determine the capital of France, we need to consider the political and historical significance of Paris. Paris is the capital of France and has a rich history dating back to the Middle Ages. The city is also known for its art, architecture, and music. Therefore, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but there are some promising developments that could change the course of the field. One such development is the emergence of new types of artificial intelligence that have the potential to revolutionize the way we work and live. With advancements in machine learning and computer vision, we are seeing a new wave of AI that is capable of performing tasks that were previously considered impossible.
    One of the most promising new AI developments is the development of neural networks that can understand and learn from the patterns in large amounts of data. Neural networks are currently the backbone of many machine learning algorithms, and they are able to perform tasks such as image recognition, natural language processing,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" or simply "Paris". It is the largest city in France and the third largest in the world, with a population of over 2. 5 million people. Paris is a cultural and historical center with many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its cuisine, fashion, and music scene. Paris is a popular tourist destination and a major economic center in France. It is the capital of France and the largest city in the country. The city is also known for its rich history and culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we are likely to see more automation and artificial intelligence in our daily lives. This could include things like self-driving cars, robots in manufacturing, and even the development of AI-powered healthcare systems.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, we are likely to see an increase in the need for privacy and security. This could include things like
    


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
    Generated text:  [Name] and I am a [occupation] with [number of years in industry]. I am here to provide you with [specific information about your role] and to help you make informed decisions. So, if you have any questions or concerns, please feel free to ask me anything. And thank you for choosing to work with me. Let's get started! Let's get to know each other, so I can help you find success. Good day! 📊💼💼
    
    Hey there, thank you for taking the time to meet me! I'm [Name] from [Industry], a professional with [number of years in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is a city that is known for its rich history, iconic landmarks, and vibrant cultural scene. Paris is home to many of the world's top museums, art galleries, and theaters, and its many museums and art galleries are considered some of the world's most important collections. The city is also known for its fashion industry and fashion-forward design. Finally, Paris is a city that is known for its French cuisine, with its famous dishes and restaurants serving up some of the best food in the world. Overall, Paris is a city that is full of history, culture, and delicious cuisine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of trends and developments that will shape the technology's direction, capabilities, and impact on society. Here are some possible trends in artificial intelligence that are currently being explored:
    
    1. Increased automation: As technology continues to evolve, the rate of automation will likely increase. AI will become more capable of performing tasks that require human-like intelligence, such as decision-making, creativity, and problem-solving.
    
    2. Improved understanding of human emotions: AI will be able to understand and interpret human emotions better, which could lead to more empathetic and compassionate AI systems.
    
    3. Personalization: AI will be able to learn from user data


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

    type

     of

     character

    ]

     who

     loves

     [

    occupation

     or

     hobby

    ].

     I

    'm

     [

    age

    ]

     years

     old

    ,

     [

    location

    ].

     I

     have

     a

     [

    unique

     skill

     or

     characteristic

    ].

     And

     what

    's

     your

     name

    ?

     [

    Name

    ]

    !

     (

    meaning

     "

    Hello

    ,

     my

     name

     is

     X

    y

    lo

    phone

     and

     I

    'm

     a

     typ

    ist

     who

     loves

     writing

     songs

    .

     I

    'm

     

    2

    5

     years

     old

    ,

     and

     I

     live

     in

     [

    City

    ],

     and

     I

     have

     a

     unique

     skill

     or

     characteristic

     that

     is

    ...

     )

     I

     don

    't

     have

     a

     name

    ,

     but

     I

     can

     create

     it

     for

     you

    .

     (

    name

    )

    !

     And

     you

    ?

     (

    name

    )

    !
    


    This

     reflection

     is

     meant

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     rich

     history

    ,

     arts

    ,

     and

     culture

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     has

     a

     population

     of

     approximately

     

    1

    .

    3

     million

     people

    .

     Paris

     is

     renowned

     for

     its

     iconic

     landmarks

    ,

     including

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

     and

     Notre

    -D

    ame

     Cathedral

    ,

     as

     well

     as

     its

     vibrant

     culinary

     scene

    .

     The

     city

     is

     also

     home

     to

     many

     cultural

     institutions

    ,

     including

     the

     Mus

    ée

     national

     de

     France

     and

     the

     Centre

     Pom

    pid

    ou

    .

     Paris

     is

     considered

     a

     global

     hub

     for

     art

    ,

     fashion

    ,

     and

     wine

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     With

     its

     stunning

     architecture

    ,

     world

    -ren

    owned

     cuisine

    ,

     and

     cultural

     heritage

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     complex

     and

     diverse

    ,

     with

     many

     potential

     directions

     that

     could

     emerge

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     in

     the

     next

     few

     years

    :
    


    1

    .

     Machine

     learning

     and

     deep

     learning

    :

     One

     of

     the

     key

     trends

     in

     AI

     is

     the

     development

     of

     machine

     learning

     and

     deep

     learning

     algorithms

     that

     can

     learn

     from

     large

     amounts

     of

     data

     to

     perform

     complex

     tasks

     with

     high

     accuracy

    .
    


    2

    .

     Aug

    mentation

     and

     personal

    ization

    :

     AI

     will

     continue

     to

     evolve

     to

     enable

     more

     powerful

     and

     efficient

     use

     of

     data

    ,

     as

     well

     as

     to

     provide

     more

     personalized

     experiences

     for

     users

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     has

     already

     been

     used

     to

     improve

     the

     accuracy

     of

     medical

     diagnoses

     and

     to

     personalize

     treatment

     plans

    ,

    



```python
llm.shutdown()
```
