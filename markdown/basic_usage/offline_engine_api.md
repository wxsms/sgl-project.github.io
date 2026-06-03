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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:01, 19.75it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 27.41it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.00it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.00it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.82it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.82it/s] Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.70it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.70it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.70it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.70it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.70it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.70it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.21it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=352 avail_mem=75.74 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=320 avail_mem=75.74 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=288 avail_mem=75.29 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.10it/s]Capturing num tokens (num_tokens=256 avail_mem=75.03 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.10it/s]Capturing num tokens (num_tokens=256 avail_mem=75.03 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=240 avail_mem=75.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.18it/s]

    Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=208 avail_mem=75.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=192 avail_mem=75.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=144 avail_mem=75.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.23it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  81%|████████  | 47/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=80 avail_mem=75.00 GB):  81%|████████  | 47/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  81%|████████  | 47/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  81%|████████  | 47/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=32 avail_mem=74.99 GB):  81%|████████  | 47/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  81%|████████  | 47/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=24 avail_mem=74.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=12 avail_mem=74.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.43it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.43it/s] Capturing num tokens (num_tokens=8 avail_mem=74.97 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=4 avail_mem=74.96 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=4 avail_mem=74.96 GB): 100%|██████████| 58/58 [00:01<00:00, 37.89it/s]


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
    Generated text:  Daniel. I am 28 years old and I am a full-stack web developer in Chicago, IL. I have a passion for helping people with their tech problems and I really enjoy staying on top of the latest trends in web development and design.
    I have experience with a variety of technologies including HTML, CSS, JavaScript, SQL, and HTML5. I also have experience with server-side programming languages such as PHP, Python, and Ruby. I have also worked on front-end projects for both print and digital mediums including websites, media kits, and brochures.
    I believe that my experience as a front-end developer has helped me build
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military drones to use. He has decided that the number of drones will be an integer between 100 and 500, inclusive. Additionally, he wants to ensure that the total cost of purchasing the drones is as low as possible while still satisfying the conditions. 
    
    Drones have a total cost of $20000, and each drone uses 800 pounds of fuel. 
    
    The president also wants the drones to fly for a total distance of at least 1200 miles. He believes that if the distance is significantly greater than 1200 miles, the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris
    
    B: Rome
    
    C: London
    
    D: Berlin
    
    To determine the capital of France, let's analyze the options provided:
    
    A: Paris - This is a famous city in France, but it is not the capital of France.
    B: Rome - This is the capital of Italy, not France.
    C: London - This is the capital of England, not France.
    D: Berlin - This is the capital of Germany, which is the capital of France.
    
    Based on the information provided, the correct answer is:
    
    \boxed{D}
    ===============================
    Prompt: The future of AI is
    Generated text:  in the future. So it seems that the question of what it will be is already decided. It is already known that the AI in the future will be able to solve all sorts of problems. It is also clear that the AI will be able to do the most tedious work. Now, people who want to pursue their hobbies or personal interests are out of luck. They will have to find a new career path. The whole world will be surprised to see this phenomenon. 
    
    1. What is the main idea of this passage?
    A. The future of AI
    B. AI is able to solve all sorts of problems
    C. The


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art. Paris is a popular tourist destination and a cultural hub for Europe. It is home to many famous landmarks and museums, including the Louvre, the Musée d'Orsay, and the Centre Pompidou. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased integration with human decision-making: As AI systems become more sophisticated, they are likely to become more integrated with human decision-making processes. This could lead to more complex and nuanced AI systems that can make more informed and ethical decisions.
    
    2. Greater emphasis on ethical considerations: As AI systems become more advanced, there will
    


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
    Generated text:  Alex, I'm a professional developer with over 10 years of experience in software development. I have a knack for creating clean and efficient code, and I love helping people grow their digital businesses. I also have a strong passion for learning new technologies and technologies that are emerging, and I am always looking for ways to stay up-to-date with the latest trends and technologies. What's your favorite hobby or activity to do on a weekend? As an AI language model, I don't have personal preferences, but I can suggest some activities that you might enjoy. Do you have any hobbies or interests outside of work that you can tell me about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Sentence about the French education system, limited to 3 words. The French education system is the best in the world. 
    
    Compose an email to a high school student requesting information about the French education system. Include at least one question that requires the student to make a decision or provide an answer based on their information. Include an example of the type of question. Consider including a note that the student will have to use the information they have gathered to help them choose a French education system. I'm not sure if they're ready for the level of detail and rigor expected in a professional email. Please let me know if you have
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but there are a number of potential trends that could shape its development in the coming years. Here are a few possibilities:
    
    1. AI will become more integrated with everyday life: As AI technology becomes more advanced, we may see a greater integration of AI into our daily lives. This could include things like smart homes, self-driving cars, and personalized healthcare services.
    
    2. AI will become more accessible: As AI becomes more integrated into our daily lives, we may see an increase in the availability and accessibility of AI technology. This could include things like lower-cost AI tools, increased access to AI researchers, and increased participation in AI research


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

    Your

     Name

    ].

     I

     am

     a

     [

    insert

     your

     occupation

     here

    ,

     such

     as

     "

    teacher

    ",

     "

    engine

    er

    ",

     "

    law

    yer

    ",

     "

    writer

    ",

     etc

    .]

     with

     over

     [

    insert

     number

     of

     years

     of

     experience

     here

    ].

     I

     love

     [

    insert

     one

     or

     two

     hobbies

     you

     enjoy

    ,

     such

     as

     playing

     sports

    ,

     reading

    ,

     cooking

    ,

     etc

    .

    ].

     And

     I

     am

     always

     eager

     to

     learn

     new

     things

     and

     improve

     my

     craft

    .

     I

     am

     [

    insert

     your

     current

     age

     in

     years

    ]

     years

     old

     and

     I

     live

     in

     [

    insert

     your

     current

     location

     here

    ].


    As

     someone

     who

     enjoys

     learning

     and

     expanding

     their

     knowledge

    ,

     I

     am

     excited

     to

     share

     my

     passion

     for

     teaching

     and

     writing

     with

     anyone

     who

     is

     interested

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    ,

     and

     is

     the

     seat

     of

     government

    ,

     administration

    ,

     and

     culture

     in

     the

     country

    .

     It

     is

     the

     largest

     city

     in

     the

     European

     Union

    ,

     with

     over

     

    2

    .

    2

     million

     inhabitants

    .

     Paris

     has

     a

     rich

     history

    ,

     culture

    ,

     and

     art

     scene

    ,

     with

     numerous

     landmarks

     such

     as

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

    .

     It

     is

     also

     the

     capital

     of

     the

     department

     of

     Paris

    ,

     which

     includes

     the

     metropolitan

     area

     of

     Paris

     and

     the

     surrounding

     region

    ,

     and

     has

     been

     a

     major

     center

     for

     business

    ,

     trade

    ,

     and

     culture

     since

     the

     Middle

     Ages

    .

     Paris

     has

     a

     diverse

     population

     and

     economy

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     different

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     developments

     in

     natural

     language

     processing

    ,

     and

     improvements

     in

     ethics

     and

     privacy

     concerns

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

     Increased

     integration

     with

     humans

    :

     AI

     systems

     are

     likely

     to

     become

     more

     integrated

     with

     humans

     in

     the

     future

    .

     This

     could

     involve

     the

     use

     of

     AI

    -powered

     assistants

     to

     perform

     a

     range

     of

     tasks

    ,

     such

     as

     scheduling

     appointments

    ,

     managing

     finances

    ,

     and

     providing

     health

     advice

    .
    


    2

    .

     Greater

     use

     of

     AI

     for

     creative

     and

     artistic

     tasks

    :

     AI

     will

     likely

     become

     more

     involved

     in

     creative

     and

     artistic

     tasks

    ,

     including

     music

     composition

    ,

     visual

     arts

    ,

     and

     animation

    .
    


    3

    .

     Adv

    ancements

    



```python
llm.shutdown()
```
