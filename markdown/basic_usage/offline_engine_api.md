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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.78it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.05it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 31.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.70 GB):  31%|███       | 18/58 [00:00<00:01, 31.71it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=71.70 GB):  31%|███       | 18/58 [00:00<00:01, 31.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.68 GB):  31%|███       | 18/58 [00:00<00:01, 31.71it/s]Capturing num tokens (num_tokens=960 avail_mem=71.69 GB):  31%|███       | 18/58 [00:00<00:01, 31.71it/s] Capturing num tokens (num_tokens=960 avail_mem=71.69 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=896 avail_mem=71.67 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=832 avail_mem=71.20 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=768 avail_mem=71.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=704 avail_mem=71.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=640 avail_mem=71.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.70it/s]

    Capturing num tokens (num_tokens=640 avail_mem=71.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.34it/s]Capturing num tokens (num_tokens=576 avail_mem=71.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.34it/s]Capturing num tokens (num_tokens=512 avail_mem=71.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.34it/s]Capturing num tokens (num_tokens=480 avail_mem=71.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.34it/s]Capturing num tokens (num_tokens=448 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.34it/s]Capturing num tokens (num_tokens=416 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.34it/s]Capturing num tokens (num_tokens=416 avail_mem=71.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=384 avail_mem=71.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=352 avail_mem=71.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=320 avail_mem=71.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.17it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=240 avail_mem=71.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=224 avail_mem=71.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.06it/s]Capturing num tokens (num_tokens=160 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.06it/s]Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.06it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.06it/s]Capturing num tokens (num_tokens=112 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.06it/s]Capturing num tokens (num_tokens=96 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.06it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=70.98 GB):  81%|████████  | 47/58 [00:01<00:00, 45.08it/s]Capturing num tokens (num_tokens=80 avail_mem=70.97 GB):  81%|████████  | 47/58 [00:01<00:00, 45.08it/s]Capturing num tokens (num_tokens=64 avail_mem=70.97 GB):  81%|████████  | 47/58 [00:01<00:00, 45.08it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 45.08it/s]Capturing num tokens (num_tokens=32 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 45.08it/s]Capturing num tokens (num_tokens=28 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 45.08it/s]Capturing num tokens (num_tokens=28 avail_mem=70.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=24 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=20 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.79it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=4 avail_mem=70.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=4 avail_mem=70.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.67it/s]


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
    Generated text:  Chen Hua, I’m 18 years old and I’m in the eighth grade. I’m going to an eighth-grade school this year. This year is one of the best years I’ve been in school. Because this year I got a lot of good things. In the first month of school I had a very bad cold. I had to stay in the hospital for a whole week. But I feel much better now. The second month, my teacher made us do a math problem. I was nervous but happy. I even felt excited. I tried my best to solve it. The last month I got some math problems.
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of Brazil. The president of Brazil is 2 times older than the president of France. If France is currently 30 years old, how old will the president of Brazil be in 10 years?
    To determine the current age of the president of Brazil, we start by identifying the given information and calculating step by step.
    
    1. The president of the United States is 30 years older than the president of Brazil.
    2. The president of Brazil is 2 times older than the president of France.
    
    We know the president of France is currently 30 years old. Let's denote
    ===============================
    Prompt: The capital of France is
    Generated text: ____
    A. London
    B. Paris
    C. Rome
    D. Madrid
    Answer:
    
    B
    
    Which of the following uses of substances is determined by their chemical properties?
    A. Using dry ice for artificial precipitation
    B. Using alcohol for disinfection
    C. Using liquefied petroleum gas for cooking
    D. Using activated carbon for water purification
    Answer:
    
    B
    
    The graph of the function \( y = \log_{2}(x+1) \) is transformed by first shifting it left by 1 unit and then reflecting it about the x-axis. What is the equation of the resulting graph? 
    A.
    ===============================
    Prompt: The future of AI is
    Generated text:  about making the right decisions.
    
    The Future of AI is about making the right decisions. The future of AI is about making the right decisions. The future of AI is about making the right decisions.
    
    -Elon Musk
    
    Let's work on AI in a way that makes the right decisions. What should we do to ensure that AI systems are not only efficient but also ethical?
    
    ### 1. **Establish Clear Guidelines and Standards:**
    
    - **Ethical Standards:** Establish clear, transparent, and accessible ethical standards for AI development and deployment. These should include principles like fairness, accountability, privacy, and transparency.
    
    - **Regulation:** Develop


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


    Generated text:  [Name] and I am a [Age] year old [Gender] [Occupation]. I have always been passionate about [Your passion or interest]. I am always looking for new experiences and learning new things. I am always eager to try new things and push myself to the limit. I am a [Your favorite hobby or activity]. I am always looking for ways to improve myself and make the world a better place. I am a [Your favorite book, movie, or song]. I am always looking for new ways to inspire and motivate others. I am a [Your favorite person or place]. I am always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The French capital is a vibrant and dynamic city that is a must-visit for anyone interested in French culture and history. 
    
    The French capital is also known for its cuisine, with Paris being
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust AI systems that are designed to be transparent, accountable, and
    


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
    Generated text:  [Name]. I'm a [occupation] and I enjoy [activity/debate/topic] in [field]. What brings you to [city/state]?
    
    (1. Write a short, neutral introduction that explains the character's profession and the topic they are passionate about.)
    
    (2. Write a brief response to the question, letting the reader know how the question was answered and any additional information that was provided.)
    
    (3. Include a quote or memorable phrase from the character to show their personality and make the introduction more engaging.)
    
    (4. The intro should be as short and to the point as possible, with a focus on highlighting the character
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its cultural institutions, cuisine, and historic landmarks. It is often referred to as the "City of Light" and "The City of Lights". Paris is famous for its Notre-Dame Cathedral and the Louvre Museum, among other attractions. It is the seat of the French government and is home to many important museums, including the Musée d'Orsay and the Centre Pompidou. The city is also known for its diverse cultural scene and is home to many famous artists, including Vincent van Gogh and Pablo Picasso.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, with many potential applications and developments. Here are some possible trends and technologies that are likely to shape the AI landscape in the coming years:
    
    1. Automation and robotics: The AI field is already seeing significant automation, with robots and other AI-powered systems replacing human workers in manufacturing, manufacturing, healthcare, and transportation sectors. However, as AI becomes more sophisticated, we may see even more automation in areas like finance, transportation, and retail.
    
    2. Natural language processing: With the increasing amount of data being generated and analyzed by AI systems, natural language processing is becoming increasingly important. This includes tasks like language translation, sentiment analysis,


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

     am

     [

    Name

    ]'

    s

     friend

    ,

     and

     I

     am

     a

     [

    whatever

     it

     is

    ]

     who

     is

     always

     [

    whatever

     it

     is

    ].

     I

     am

     a

     [

    whatever

     it

     is

    ]

     who

     has

     [

    whatever

     it

     is

    ]

     interests

    .

     I

     am

     [

    whatever

     it

     is

    ]

     and

     I

     am

     always

     [

    whatever

     it

     is

    ].

     How

     are

     you

    ,

     [

    Name

    ]?

     I

     am

     [

    Name

    ],

     and

     I

     am

     a

     [

    whatever

     it

     is

    ].

     I

     am

     a

     [

    whatever

     it

     is

    ]

     who

     is

     always

     [

    whatever

     it

     is

    ].

     I

     have

     [

    whatever

     it

     is

    ]

     interests

    ,

     and

     I

     am

     always

     [

    whatever

     it

     is

    ].

     How

     are

     you

    ,

     [

    Name

    ]?

     I

     am

     [

    Name

    ],

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     majestic

     architecture

    ,

     charming

     old

     neighborhoods

    ,

     and

     vibrant

     cultural

     scene

    .

     The

     city

     was

     founded

     in

     

    7

    8

    9

     by

     Char

    lem

    agne

    ,

     and

     has

     been

     the

     political

    ,

     cultural

    ,

     and

     economic

     capital

     of

     France

     since

     

    1

    8

    3

    0

    .

     It

     is

     known

     for

     its

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Se

    ine

     River

    .

     Paris

     is

     also

     famous

     for

     its

     annual

     E

    iff

    el

     Tower

     celebration

    ,

     which

     attracts

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     restaurants

    ,

     and

     its

     diverse

     population

     of

     

    2

    .

    1

     million

     residents

     make

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

     with

     a

     wide

     range

     of

     possible

     trends

    .

     Some

     of

     the

     most

     promising

     areas

     include

    :
    


    1

    .

     More

     autonomous

     and

     ethical

     AI

    :

     As

     autonomous

     vehicles

     become

     more

     common

    ,

     we

     can

     expect

     to

     see

     more

     advanced

     AI

     systems

     that

     can

     make

     decisions

     on

     our

     behalf

    ,

     reduce

     human

     error

    ,

     and

     take

     risks

     on

     our

     behalf

    .

     This

     could

     lead

     to

     a

     more

     ethical

     and

     responsible

     use

     of

     AI

     in

     society

    .
    


    2

    .

     Improved

     natural

     language

     processing

    :

     With

     the

     increase

     in

     the

     number

     of

     tasks

     that

     AI

     can

     perform

    ,

     we

     can

     expect

     to

     see

     an

     increase

     in

     the

     complexity

     and

     sophistication

     of

     natural

     language

     processing

     systems

    .

     This

     could

     lead

     to

     even

     more

     powerful

     and

     flexible

     AI

     that

     can

    



```python
llm.shutdown()
```
