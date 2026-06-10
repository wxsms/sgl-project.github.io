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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.11it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.90it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.90it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.35it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=256 avail_mem=76.66 GB):  55%|█████▌    | 32/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=256 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=240 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=208 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=192 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=176 avail_mem=76.62 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=176 avail_mem=76.62 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=160 avail_mem=76.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=144 avail_mem=76.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=128 avail_mem=76.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=112 avail_mem=76.14 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=96 avail_mem=76.04 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.63it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=76.04 GB):  81%|████████  | 47/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.04it/s]


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
    Generated text:  Sam Johnson, I am 25 years old, I am from New York City. I like to eat a lot of different things and I like to travel. I like to use my cellphone. I like to use my computer. I like to play games. I like to do my homework. I like to listen to music.
    What is the question I should ask?
    What do you like to do in your free time? To determine the most suitable question for asking about your free time activities, I would focus on what you currently enjoy doing and then expand to a broader question that could lead to discussing more details if desired. 
    
    Given
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have around the country. He likes the x base the least, but doesn't want more than 5. He also likes having 5 bases total, but doesn't want more than 9. How many bases does he have in total?
    If we know the answer to the above question is 12, what is the value of unknown variable x?
    We are given that the president likes the x base the least, but doesn't want more than 5 bases total. This means x can be any value from 1 to 5.
    We are also given that the president doesn't want
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. London
    C. Rome
    D. Madrid
    Answer:
    A
    
    Among the following four options, which one does not belong to the main functions of the Internet of Things (IoT) to achieve?
    A. Real-time monitoring
    B. Data analysis
    C. Data transmission
    D. Storage and processing
    Answer:
    D
    
    An athlete is jumping at a certain height and then falls vertically. Assuming the athlete's vertical velocity at the moment he hits the ground is 0, what can be inferred about the athlete's maximum height reached?
    A. The athlete must land at ground level
    
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s going to be incredibly powerful. If you’re not familiar, AI is an acronym that stands for Artificial Intelligence, which refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI encompasses a variety of techniques and technologies, including machine learning, natural language processing, computer vision, and robotics.
    So, what’s the point of having AI in your business? Well, there are a few main things that can happen. Firstly, AI can help to automate routine tasks, such as accounting and customer service. This can help to save time and increase efficiency, which can lead to cost savings


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


    Generated text:  [Name], and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] with [Experience] years of experience in [Field]. I'm passionate about [What I Love to Do], and I'm always looking for new challenges and opportunities to grow and learn. I'm always eager to learn and improve, and I'm always willing to put in the time and effort to achieve my goals. I'm a [Personality] person, and I'm always ready to make a positive impact on the world. I'm excited to meet you and learn more about you. How about you? What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination, attracting millions of visitors each year. The city is known for its cuisine, fashion, and art, and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and historical landmarks blending
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of technologies, including smartphones, cars, and smart homes. As more and more devices become connected to the internet, we can expect to see even more integration of AI into other technologies.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical and social
    


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
    Generated text:  [First Name] [Last Name], and I'm a [job title or profession] in [industry or company]. I've always been passionate about [what you'd like to include in the introduction].
    As a professional in the [industry or company], I am dedicated to [job title] and always strive to [specific goal or activity]. I am passionate about [motivational tagline or statement]. I have a strong work ethic and always aim to [specific achievement or goal].
    I am always eager to learn and push myself to become the best version of myself. I am a [intelligent, experienced, or calm] personality
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the heart of the region of the same name. It is the country's political, cultural, and economic center, with its population numbering around 1.2 million people. It is the most visited city in Europe and is considered one of the top tourist destinations in the world. Paris is known for its romantic architecture, vibrant culture, and annual festivals such as the Eiffel Tower celebration. The city is also home to numerous museums, theaters, and art galleries. It plays a crucial role in France's identity and is recognized as a cultural center in the world. Paris was founded by the French crown in the 
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but some potential trends we can expect to see include:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, leading to more efficient and effective decision-making processes.
    
    2. Increased use of AI for creative tasks: AI is increasingly being used for creative tasks such as image and video synthesis, natural language processing, and even human-like simulations of creative work.
    
    3. Greater focus on ethical considerations: As AI technology continues to evolve, there will likely be greater focus on ethical considerations, including issues such as bias, transparency, and accountability.
    
    4. Increased use of AI for personalization: AI will become


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

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    ].

     I

     have

     an

     interest

     in

     [

    field

     of

     interest

    ].

     I

     have

     a

     passion

     for

     [

    m

    otive

     or

     reason

     for

     interest

    ].

     I

     am

     a

     [

    active

    ,

     passive

    ,

     or

     in

    -between

    ]

     learner

     and

     I

     strive

     to

     [

    what

     I

    'm

     striving

     to

     do

    ].

     I

     enjoy

     [

    what

     I

     enjoy

     doing

    ].

     My

     [

    job

     or

     hobby

    ]

     is

     [

    what

     it

     is

    ].

     I

     am

     always

     looking

     for

     new

     ways

     to

     [

    what

     I

    'm

     looking

     for

    ],

     and

     I

     love

     to

     [

    what

     I

     love

    ].

     How

     would

     you

     describe

     your

     personality

     traits

    ,

     strengths

    ,

     and

     weaknesses

    ?

     I

     am

     [

    positive

    ,

     guarded

    ,

     open

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Here

    's

     the

     factual

     statement

     in

     a

     concise

     form

    :
    


    Paris

    ,

     the

     capital

     of

     France

    ,

     is

     renowned

     for

     its

     rich

     history

    ,

     iconic

     landmarks

     like

     the

     E

    iff

    el

     Tower

    ,

     and

     vibrant

     French

     culture

    .

     It

    's

     an

     urban

     hub

     with

     a

     population

     of

     around

     

    2

    .

    2

     million

     people

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

     and

     world

    -class

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Despite

     its

     size

    ,

     the

     city

     offers

     numerous

     cultural

     events

    ,

     from

     opera

     and

     ballet

     performances

     to

     famous

     museums

     and

     art

     galleries

    .

     Visitors

     can

     explore

     this

     stor

    ied

     met

    ropolis

     on

     foot

    ,

     by

     metro

    ,

     or

     even

     take

     the

     train

     or

     bus

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     one

     that

     is

     rapidly

     changing

    ,

     and

     there

     are

     many

     possibilities

     for

     what

     the

     technology

     could

     evolve

     into

     in

     the

     coming

     years

    .

     Some

     of

     the

     potential

     future

     trends

     for

     AI

     include

    :
    


    1

    .

     Increased

     specialization

     and

     AI

     bots

    :

     With

     the

     rise

     of

     cloud

     computing

     and

     AI

     assistants

    ,

     we

     may

     see

     more

     specialized

     AI

     algorithms

     and

     bots

     that

     can

     perform

     specific

     tasks

     with

     a

     high

     degree

     of

     accuracy

     and

     efficiency

    .
    


    2

    .

     Personal

    ized

     experiences

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     can

     expect

     more

     personal

    ization

     in

     our

     interactions

     with

     technology

    .

     This

     could

     lead

     to

     more

     advanced

     personalized

     recommendations

     and

     messaging

     systems

    ,

     as

     well

     as

     more

     intuitive

     and

     user

    -friendly

     AI

     interfaces

    .
    


    3

    .

     Eth

    ical

     and

     moral

    



```python
llm.shutdown()
```
