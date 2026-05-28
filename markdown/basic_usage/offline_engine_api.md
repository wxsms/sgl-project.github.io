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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 19.94it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 28.16it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 38.01it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 38.01it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 38.01it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 38.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.64 GB):   3%|▎         | 2/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.51it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.19it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s] Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.74it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.54it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.54it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.54it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.54it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.54it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  71%|███████   | 41/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.75it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.54it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.54it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.54it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.54it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.54it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.54it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.04it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=12 avail_mem=74.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=12 avail_mem=74.24 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=8 avail_mem=74.24 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.60it/s] Capturing num tokens (num_tokens=4 avail_mem=74.23 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=4 avail_mem=74.23 GB): 100%|██████████| 58/58 [00:01<00:00, 35.77it/s]


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
    Generated text:  Mark. I live in the United States with my wife. We are a couple and we like to travel a lot. We like to go to the beach with our families and we like to go on hikes. I am a very hard worker and I have a great job. I like to read a lot of books and listen to music. I am always learning new things and getting better at my job. I have a very healthy lifestyle. I drink lots of water and I eat lots of vegetables. I like to take time to relax and enjoy my family time. I enjoy spending time with my family and friends. I also like to enjoy
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to increase the budget for the Department of Education by 10% this year. This year's budget is $22 billion, and the budget for last year was $20 billion. Calculate the budget increase as a percentage of this year's budget, and then find the original budget for the Department of Education.
    To determine the budget increase and the original budget for the Department of Education, we will follow these steps:
    
    1. Calculate the budget increase as a percentage of this year's budget.
    2. Use the budget increase as a percentage of this year's budget to find the original budget for the Department of Education.
    
    **Step
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Brussels
    C. Nice
    D. Lyon
    Answer:
    
    A
    
    According to China's 'Regulations on the Administration of Dangerous Goods Transportation', which of the following vehicles are prohibited from carrying dangerous goods? ____ 
    A. Passenger vehicles
    B. Vehicles transporting goods
    C. Tractors
    D. Special purpose vehicles
    Answer:
    
    C
    
    The time complexity of the following code is ____.
    ```csharp
    class Test {
        void f() {
            int x = 5;
            do {
                ++x;
            } while (--x != 0);
        }
    }
    ```
    A
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly evolving, and the landscape is constantly changing. The AI industry is advancing at a rapid pace, driven by the increasing demand for intelligent solutions to complex problems. One of the most promising areas of AI development is the field of natural language processing (NLP).
    NLP has the potential to revolutionize how we interact with machines, automate tasks, and enable us to learn from the world. In this blog post, we will explore the current state of NLP, discuss its applications, and discuss the latest advancements in the field.
    What is Natural Language Processing (NLP)?
    Natural Language Processing (NLP) is an interdisciplinary field that combines


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always up for a challenge and love to explore new experiences. What's your favorite book or movie? I'm a huge fan of [insert a book or movie here]. I'm always looking for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance in French culture and politics. It is a widely recognized and well-known fact about Paris that it is the capital city of France. 
    
    To summarize, the statement "The capital of France is Paris" is a factual statement that accurately describes the location and importance of the capital city in French society and politics. It is a widely recognized fact that Paris is the capital of France, and this statement provides a clear and concise overview of the capital city's location and significance in French culture and politics. 
    
    Therefore, the answer to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, reduce costs, and improve quality. As AI technology continues to improve, we can expect to see even more widespread use of AI in manufacturing.
    
    3. AI in finance
    


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
    Generated text:  [insert name], and I'm an [insert occupation]. I'm excited to meet you and chat about what you do for a living and what makes you unique.
    I hope you enjoyed the opportunity to meet me, and I'm here to learn more about what you do and how you approach your work. If you have any questions or would like to discuss anything in more detail, please feel free to let me know. Thank you for taking the time to learn about me. Happy to chat! 🙋‍♂️💼✨
    Hey there! I'm [insert name], a [insert occupation]! Can you tell me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris, the cultural and historical capital of France, is known for its iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Its rich history, art, and cuisine make it a popular tourist destination. It's also home to numerous museums, theaters, and festivals throughout the year. Paris is a city of contrasts, with its distinctive architecture and unique culture, making it a must-visit destination for anyone visiting France. (See the answer for more details.) 
    
    I hope this helps! Let me know if you have any other questions. Paris is truly a captivating
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and promising, with many potential applications and developments to consider. Here are some potential future trends in AI:
    
    1. AI for healthcare: AI is already being used to improve diagnosis and treatment outcomes, particularly in oncology, where AI is helping doctors to predict patient outcomes and develop personalized treatment plans. As AI technology advances, we may see even more sophisticated AI being used in healthcare, particularly for diagnosing diseases and treating chronic conditions.
    
    2. AI for agriculture: AI is being used to improve crop yields, minimize waste, and increase efficiency. For example, AI-powered drones can be used to monitor crops and detect pests and diseases in real


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

    profession

     or

     job

     title

    ]

    !

     I

    'm

     always

     up

     for

     a

     good

     challenge

     and

     enjoy

     [

    Your

     profession

     or

     job

     title

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    You

     can

     include

     a

     few

     personal

     anecdotes

    ,

     relevant

     work

     experiences

    ,

     or

     any

     other

     interesting

     facts

     about

     yourself

    .

     Make

     sure

     to

     keep

     your

     introduction

     short

     and

     to

     the

     point

    ,

     with

     a

     neutral

     tone

    .

    ]


    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    profession

     or

     job

     title

    ]

    !

     I

    'm

     always

     up

     for

     a

     good

     challenge

     and

     enjoy

     [

    Your

     profession

     or

     job

     title

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    You

     can

     include

     a

     few

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     third

     most

     populous

     city

     in

     the

     world

    ,

     after

     Beijing

     and

     Shanghai

    ,

     and

     is

     the

     largest

     city

     by

     area

    ,

     as

     well

     as

     by

     population

    ,

     in

     the

     European

     Union

    .

     Paris

     is

     also

     known

     as

     the

     "

    city

     of

     love

    "

     for

     its

     rich

     cultural

     heritage

     and

     romantic

     atmosphere

    .

     It

     is

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

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     art

     galleries

    ,

     and

     theaters

    ,

     and

     is

     a

     major

     hub

     for

     business

    ,

     commerce

    ,

     and

     culture

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     including

     the

     famous

     cout

    ur

    ier

     D

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     technology

    :
    


    1

    .

     Increased

     automation

    :

     As

     AI

     continues

     to

     improve

     and

     become

     more

     sophisticated

    ,

     we

     can

     expect

     more

     automation

     in

     various

     industries

    .

     Automation

     could

     include

     tasks

     such

     as

     data

     entry

    ,

     routine

     maintenance

    ,

     and

     administrative

     tasks

    ,

     as

     well

     as

     tasks

     that

     are

     typically

     carried

     out

     by

     humans

    .
    


    2

    .

     Eth

    ical

     considerations

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     will

     need

     to

     address

     ethical

     considerations

     such

     as

     bias

    ,

     privacy

    ,

     and

     security

    .

     AI

     systems

     will

     need

     to

     be

     designed

     with

     these

     in

     mind

    ,

     and

     we

     will

     need

     to

     ensure

     that

     they

     are

     used

     in

     a

     way

     that

     is

     transparent

     and

     responsible

    .
    


    3

    .

    



```python
llm.shutdown()
```
