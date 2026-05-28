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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.58it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.66it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 32.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:02, 21.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.80it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.83it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.41it/s]

    Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.03it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.03it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.03it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.03it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.03it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.03it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.13it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.13it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=208 avail_mem=73.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=192 avail_mem=73.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.13it/s]

    Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.05it/s] Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  81%|████████  | 47/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  81%|████████  | 47/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  81%|████████  | 47/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  81%|████████  | 47/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=32 avail_mem=73.88 GB):  81%|████████  | 47/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  81%|████████  | 47/58 [00:01<00:00, 45.45it/s]

    Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]

    Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s] Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  98%|█████████▊| 57/58 [00:01<00:00, 27.09it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  98%|█████████▊| 57/58 [00:01<00:00, 27.09it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:01<00:00, 34.31it/s]


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
    Generated text:  Maria and I'm a computer science student at CMU. I'm not sure what I want to do after I graduate and this is my first time discussing my life. Please provide me with some information about my interests, what programming languages I know, and what computer science courses I've taken so far.
    
    Sure, I'd be happy to help! What about you? Do you have any specific questions or would you like to know more about your interests? Let me know! I'm here to listen and answer any questions you may have. 
    
    ---
    
    **Maria**: Hi! I'm just starting my journey in computer science and I'm interested
    ===============================
    Prompt: The president of the United States is
    Generated text:  200 cm tall. His assistant is 75% taller than him. How tall is the assistant in feet? To determine the height of the assistant in feet, we first need to find the height of the assistant in centimeters. We know that the president's height is 200 cm and that the assistant is 75% taller than the president.
    
    First, we calculate 75% of the president's height:
    \[ 75\% \text{ of } 200 \text{ cm} = 0.75 \times 200 \text{ cm}
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    B. Paris
    C. Calais
    D. Luxembourg
    
    1. **Identify the capital of France**: 
       The capital of France is Paris, which is the largest city in France. It is located in the center of the country, surrounded by the Alps and the Mediterranean Sea.
    
    2. **Analyze the given options**:
       - Option A: Paris
       - Option B: Paris
       - Option C: Calais
       - Option D: Luxembourg
    
    3. **Determine the correct answer**:
       Among the given options, Paris is the capital of France. It is
    ===============================
    Prompt: The future of AI is
    Generated text:  very uncertain. That’s why Microsoft has always supported the adoption of open source technologies such as OpenAI. This technology was built on the following premise: the programming of AI must be transparent and open to review.
    AI technologies can be used for good, but the responsible use of the new AI technologies is a challenge. This is particularly true of open source AI technologies.
    In the tech industry, the security of the systems that the AI algorithms run on is a central issue. We see an increasing trend towards “homogenization” or the creation of AI systems that are difficult to identify or control. The US government has announced that it will prioritize


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic description of your personality]. I enjoy [insert a short, positive, enthusiastic description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, positive, enthusiastic description of your favorite hobby or activity]. I'm always looking for new ways to challenge myself and expand my horizons. What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is the largest city in France and a major economic and political center in Europe. Paris is also home to the French Parliament and the French Academy of Sciences. The city is known for its cuisine, including French cuisine, and is a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient care, from personalized treatment plans to automated diagnostic tools. As AI technology continues to advance, we can expect to see even more sophisticated applications in healthcare, such as personalized medicine and virtual assistants for patients.
    
    2. Increased use of AI in manufacturing: AI is already being used to optimize production processes, reduce costs, and improve quality. As AI technology continues to evolve, we can expect to see even more sophisticated
    


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
    Generated text:  Alex. I am a science fiction writer and I am in my early twenties. I have a degree in English and a master’s degree in computer science. I have a few years experience as a journalist and have also developed my own web development skills. I have a love for the stars and technology and I am always looking for new ideas to create. I enjoy writing about science fiction and have a passion for exploring the potential of technology to change the world. What's your favorite genre? I don't have a favorite genre as I am always exploring new ways to write and experiment with different styles and genres. But, I do enjoy writing space
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its rich cultural heritage, stunning architecture, and vibrant nightlife. It serves as the city's political, economic, and cultural center. Paris is also famous for its iconic landmarks such as the Eiffel Tower and the Louvre Museum. It is often referred to as the "City of Light" due to its numerous light shows and neon signs. Paris is a must-visit destination for visitors interested in French culture and history. As one of the most cosmopolitan cities in the world, Paris has attracted many foreign visitors and has made a significant impact on the cultural, economic, and political life of France. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  diverse and uncertain, with many potential trends emerging as technology advances. Here are some of the most likely future trends in artificial intelligence:
    
    1. Increased privacy concerns: As AI systems become more sophisticated, we will see an increased focus on privacy and data protection. Developers will likely incorporate more robust security measures into their AI systems, and users will need to be more mindful of the data they share with AI-powered services.
    
    2. Autonomous vehicles: The development of fully autonomous vehicles is likely to become a significant trend in the coming years. AI will play a crucial role in making these vehicles safer and more efficient, and developers will need to develop new algorithms


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

     come

     from

     a

     humble

     background

    ,

     but

     I

     am

     now

     a

     rising

     star

     in

     the

     tech

     industry

    .

     I

     have

     a

     knack

     for

     problem

    -solving

     and

     a

     passion

     for

     innovation

    .

     I

     believe

     in

     taking

     risks

     and

     pushing

     boundaries

     to

     achieve

     success

    .

     Let

    's

     start

     this

     conversation

     with

     a

     brief

     summary

     of

     your

     goals

     and

     current

     projects

    .

     As

     a

     quick

     survey

    ,

     I

     don

    ’t

     know

     what

     else

     to

     say

    .

     I

     have

     been

     working

     on

     [

    Your

     project

     or

     mission

    ].

     Let

    's

     talk

     about

     your

     journey

     so

     far

     and

     how

     you

     got

     into

     the

     tech

     industry

    .

     I

     can

     see

     you

    're

     passionate

     about

     sharing

     your

     story

    ,

     so

     could

     you

     tell

     us

     a

     bit

     about

     your

     journey

     and

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    1

    .

     Find

     the

     average

     temperature

     in

     Paris

     during

     winter

    .


    2

    .

     Provide

     the

     code

     for

     a

     popular

     street

     in

     Paris

    .


    3

    .

     Identify

     the

     main

     languages

     spoken

     in

     Paris

    .


    4

    .

     Describe

     the

     famous

     landmarks

     in

     the

     city

     of

     Paris

    .


    5

    .

     Explain

     the

     significance

     of

     Paris

     in

     French

     culture

     and

     history

    .

     

    1

    .

     The

     average

     temperature

     in

     Paris

     during

     winter

     is

     around

     

    1

    5

    °C

     (

    5

    9

    °F

    ).


     

     

    2

    .

     The

     code

     for

     a

     popular

     street

     in

     Paris

     is

     Rue

     de

     Riv

    oli

    .


     

     

    3

    .

     The

     main

     languages

     spoken

     in

     Paris

     are

     French

    ,

     French

    ,

     English

    ,

     and

     several

     other

     languages

    .


     

     

    4

    .

     The

     famous

     landmarks

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     will

     continue

     to

     evolve

     in

     many

     ways

    .

     Some

     of

     the

     potential

     trends

     that

     are

     expected

     to

     occur

     include

    :
    


    1

    .

     Increased

     autonomy

    :

     AI

     will

     continue

     to

     gain

     more

     autonomy

    ,

     allowing

     it

     to

     make

     decisions

     based

     on

     less

     direct

     and

     less

     controlled

     inputs

    ,

     such

     as

     through

     natural

     language

     processing

     and

     machine

     learning

     algorithms

    .
    


    2

    .

     Semantic

     processing

    :

     AI

     will

     gain

     the

     ability

     to

     understand

     and

     interpret

     human

     language

     more

     accurately

    ,

     allowing

     it

     to

     handle

     complex

     natural

     language

     tasks

     that

     are

     currently

     handled

     by

     humans

    .
    


    3

    .

     Enhanced

     predictive

     analytics

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     make

     accurate

     predictions

     and

     forecasts

     based

     on

     large

     amounts

     of

     data

    ,

     allowing

     businesses

     to

     make

     more

     informed

     decisions

    .
    


    



```python
llm.shutdown()
```
