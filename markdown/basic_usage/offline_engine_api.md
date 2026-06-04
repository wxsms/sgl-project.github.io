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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.45it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.43it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.43it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.43it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.43it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.43it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.43it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.43it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.13it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.13it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.73 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.73 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.48it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.13 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.06 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.06 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.06 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.05 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.33it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=71.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.04 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.04 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.39it/s]Capturing num tokens (num_tokens=960 avail_mem=71.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.39it/s] Capturing num tokens (num_tokens=896 avail_mem=71.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.39it/s]Capturing num tokens (num_tokens=832 avail_mem=71.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.39it/s]Capturing num tokens (num_tokens=768 avail_mem=71.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.39it/s]Capturing num tokens (num_tokens=704 avail_mem=71.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.39it/s]Capturing num tokens (num_tokens=640 avail_mem=71.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.39it/s]Capturing num tokens (num_tokens=640 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=576 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.43it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.00 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=480 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=448 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=416 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=384 avail_mem=71.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=384 avail_mem=71.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=352 avail_mem=71.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=320 avail_mem=71.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=288 avail_mem=71.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=240 avail_mem=70.99 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=224 avail_mem=70.99 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.70it/s]

    Capturing num tokens (num_tokens=224 avail_mem=70.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=176 avail_mem=70.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=160 avail_mem=70.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.30it/s]Capturing num tokens (num_tokens=112 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.30it/s]Capturing num tokens (num_tokens=96 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.30it/s] Capturing num tokens (num_tokens=80 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.30it/s]Capturing num tokens (num_tokens=64 avail_mem=70.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.30it/s]

    Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.30it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=32 avail_mem=70.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=28 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=24 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=20 avail_mem=70.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=16 avail_mem=70.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=16 avail_mem=70.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.29it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.29it/s]Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.29it/s] Capturing num tokens (num_tokens=4 avail_mem=70.93 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.29it/s]Capturing num tokens (num_tokens=4 avail_mem=70.93 GB): 100%|██████████| 58/58 [00:01<00:00, 39.57it/s]


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
    Generated text:  Amin and I am 30 years old. I am a software developer who is in the process of designing a website. I have two core problems that I am trying to solve using my expertise.
    
    **Problem 1:** I am trying to enhance the user experience by designing a responsive website. I have two solutions: HTML5, and CSS3. I've tried both but I am not sure which one is better. 
    
    **Problem 2:** I am trying to design a mobile website as well. I am also looking to make it fully responsive and accessible. I have tried implementing a responsive design with both HTML5 and CSS
    ===============================
    Prompt: The president of the United States is
    Generated text:  a wealthy man who lives in a small country with a very small economy. The president has a total of 500 acres of farmland. He plans to sell the farmland to a private company to use for agriculture. The company will pay the president $100 per acre for the farmland. However, the president plans to use the farmland for a specific purpose, which requires 10 acres of land. How much money will the president receive for selling the remaining 400 acres of farmland?
    
    The president's total earnings from selling the farmland will be:
    500 acres * $10
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris
    
    B: London
    
    C: Rome
    
    D: Moscow To determine the capital of France, let's list the capital cities of different countries:
    
    1. **France**: The capital of France is Paris.
    2. **London**: The capital of the United Kingdom is London.
    3. **Rome**: The capital of Italy is Rome.
    4. **Moscow**: The capital of Russia is Moscow.
    
    From the list, we can see that Paris is the capital of France.
    
    Therefore, the correct answer is: \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  here. The question is, how does it evolve? What do you need to be smart in the future to succeed? As the world shifts to artificial intelligence, do you need to be good at programming or be able to take care of the machine in a way that enables a robot to think in a way that is similar to a human?
    The future of AI is rapidly advancing, and while it promises exciting possibilities, it also comes with significant challenges. One of the most pressing challenges is how to ensure that AI is used for the betterment of society. In order to achieve this, it is essential to consider the ethical implications of AI and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age], [gender], [nationality], [occupation], and I have [number] years of experience in [field]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age], [gender],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its annual festivals and events, including the Eiffel Tower Festival and the Paris Fashion Week. The city is a popular tourist destination and attracts millions of visitors each year. Paris is a cultural and intellectual center of France and the world. It is also a major economic hub and a major transportation hub. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: One of the most significant trends in AI is the increasing automation of tasks that are currently performed by humans. This could include tasks such as data analysis, decision-making, and problem-solving. As AI becomes more capable of performing these tasks, it is likely to become more efficient and cost-effective, leading to increased productivity and job creation.
    
    2. Improved privacy and security: As AI becomes more integrated into
    


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
    Generated text:  [First Name] and I'm a [Job Title/Position], where I'm currently working on [Job Title/Position] at [Company Name]. I'm a [personal trait or quality] that has led me to my current position and I'm excited to continue growing as a professional. My [job title] has helped me to develop [specific skill or expertise], which I believe will make me a valuable asset to any team. I'm [any traits or qualities that make you a good fit for this role]. I'm looking forward to meeting you and getting to know you better.
    Your self-introduction is comprehensive, informative,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, an ancient city with a rich history and a beautiful skyline. It is the cultural and economic center of France, hosting many of the country's major landmarks and museums, as well as the iconic Eiffel Tower. With its opulent architecture and vibrant street life, Paris is a popular tourist destination and a cultural hub for France. 
    
    The city has a history dating back to the Roman Empire and the French Revolution, with its streets lined with grand palaces and the eponymous Eiffel Tower standing as a symbol of the city's rich history. Paris is also home to numerous museums
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  predicted to be characterized by an ever-increasing focus on developing more powerful and accurate models, as well as an emphasis on the ethical and social implications of AI. Here are some potential trends in AI that may be developed in the future:
    
    1. Increased AI capabilities: AI models will become more capable of handling complex problems and solving problems in new and unforeseen ways. This will be particularly important in areas like healthcare, where AI can be used to improve patient care and treatment outcomes.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing pressure to address the ethical and privacy issues surrounding its use


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

     [

    Your

     Profession

    ]

     and

     I

     have

     been

     working

     for

     [

    Your

     Company

     Name

    ]

     for

     [

    X

     years

    ]

     since

     [

    Year

    ].

     I

     am

     passionate

     about

     [

    Your

     Passion

    /

    Interest

    ].

     I

     enjoy

     [

    How

     you

     like

     to

     spend

     your

     time

    ].

     I

     am

     constantly

     learning

     and

     growing

     as

     a

     professional

    .

     I

     love

     being

     outdoors

     and

     spending

     time

     in

     nature

    .

     I

     love

     to

     cook

     and

     I

     am

     always

     trying

     to

     learn

     new

     recipes

    .

     I

     am

     known

     for

     my

     [

    Your

     Unique

     Selling

     Points

    ]

    !

     I

     thrive

     on

     challenging

     myself

     and

     working

     towards

     my

     goals

    .

     I

     am

     always

     up

     for

     a

     good

     challenge

     and

     I

     am

     not

     afraid

     to

     take

     risks

    .

     I

     am

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Please

     paraph

    rase

     the

     statement

     above

     in

     French

    .

     Le

     capital

     de

     la

     France

     est

     Paris

    .

     
    


    This

     is

     a

     concise

     factual

     statement

     about

     the

     capital

     of

     France

    .

     In

     French

    ,

     it

     reads

     "

    Le

     capital

     de

     la

     France

     est

     Paris

    ".

     The

     structure

     of

     the

     sentence

     has

     been

     altered

     to

     make

     it

     more

     formal

     and

     concise

    ,

     but

     the

     meaning

     remains

     the

     same

    .

     The

     French

     language

     uses

     di

    ac

    ritical

     marks

    ,

     which

     is

     why

     the

     spelling

     and

     pronunciation

     are

     slightly

     different

     from

     the

     English

     version

    .

     The

     French

     capital

     is

     the

     capital

     of

     the

     French

     Republic

    ,

     a

     country

    ,

     and

     Paris

     is

     the

     capital

     city

     of

     France

    .

     The

     rest

     of

     the

     statement

     has

     been

     translated

     into

     French

     while

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     one

     of

     rapid

     growth

    ,

     advancement

    ,

     and

     integration

     of

     new

     technologies

     and

     approaches

    .

     Some

     of

     the

     possible

     future

     trends

     in

     artificial

     intelligence

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     social

     implications

    :

     As

     the

     technology

     advances

    ,

     we

     are

     likely

     to

     see

     more

     and

     more

     AI

     systems

     that

     incorporate

     ethical

     considerations

     and

     social

     implications

    ,

     such

     as

     privacy

    ,

     fairness

    ,

     and

     transparency

    .
    


    2

    .

     Integration

     of

     AI

     with

     human

     capabilities

    :

     We

     are

     likely

     to

     see

     a

     growing

     integration

     of

     AI

     with

     human

     capabilities

    ,

     such

     as

     image

     and

     speech

     recognition

    ,

     natural

     language

     processing

    ,

     and

     decision

    -making

    .
    


    3

    .

     Advanced

     AI

     systems

    :

     AI

     systems

     are

     expected

     to

     become

     even

     more

     advanced

    ,

     with

     the

     ability

    



```python
llm.shutdown()
```
