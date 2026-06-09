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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.09it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.09it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.09it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.09it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.09it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.09it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.09it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.09it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:06,  6.74it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:06,  6.74it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:06,  6.74it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:06,  6.74it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:06,  6.74it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:06,  6.74it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:05<00:06,  6.74it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:05<00:06,  6.74it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 18.63it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 18.63it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 18.63it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 18.63it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 18.63it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 18.63it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 18.63it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 18.63it/s]

    Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 24.66it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 24.66it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 24.66it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 24.66it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 24.66it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 24.66it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 24.66it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 24.66it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 29.77it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 29.77it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 29.77it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 29.77it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 29.77it/s]

    Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 29.77it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.77it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 29.77it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 34.73it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 34.73it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 34.73it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 34.73it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 34.73it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 34.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.72 GB):   2%|▏         | 1/58 [00:00<00:08,  6.74it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.68 GB):   2%|▏         | 1/58 [00:00<00:08,  6.74it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.68 GB):   3%|▎         | 2/58 [00:00<00:08,  6.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.68 GB):   3%|▎         | 2/58 [00:00<00:08,  6.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.68 GB):   5%|▌         | 3/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.68 GB):   5%|▌         | 3/58 [00:00<00:07,  7.26it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.68 GB):   7%|▋         | 4/58 [00:00<00:07,  7.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.68 GB):   7%|▋         | 4/58 [00:00<00:07,  7.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.68 GB):   9%|▊         | 5/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.67 GB):   9%|▊         | 5/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.66 GB):   9%|▊         | 5/58 [00:00<00:06,  7.76it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=42.66 GB):   9%|▊         | 5/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.66 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.66 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.01it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=42.65 GB):  21%|██        | 12/58 [00:01<00:03, 15.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.34 GB):  21%|██        | 12/58 [00:01<00:03, 15.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.34 GB):  21%|██        | 12/58 [00:01<00:03, 15.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.34 GB):  21%|██        | 12/58 [00:01<00:03, 15.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.34 GB):  21%|██        | 12/58 [00:01<00:03, 15.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.33 GB):  21%|██        | 12/58 [00:01<00:03, 15.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.33 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.33 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.33 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.33 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.31 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.52it/s]Capturing num tokens (num_tokens=960 avail_mem=60.32 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.52it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=60.32 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.69it/s]Capturing num tokens (num_tokens=896 avail_mem=60.32 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.69it/s]Capturing num tokens (num_tokens=832 avail_mem=60.31 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.69it/s]Capturing num tokens (num_tokens=768 avail_mem=60.31 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.69it/s]Capturing num tokens (num_tokens=704 avail_mem=60.31 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.69it/s]Capturing num tokens (num_tokens=640 avail_mem=60.30 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.69it/s]Capturing num tokens (num_tokens=640 avail_mem=60.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=576 avail_mem=60.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=512 avail_mem=60.29 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=480 avail_mem=60.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=448 avail_mem=60.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=416 avail_mem=60.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.76it/s]

    Capturing num tokens (num_tokens=416 avail_mem=60.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=384 avail_mem=60.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=352 avail_mem=60.29 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=320 avail_mem=60.29 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=288 avail_mem=60.28 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=256 avail_mem=60.28 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=256 avail_mem=60.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=240 avail_mem=60.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=224 avail_mem=60.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=208 avail_mem=60.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=192 avail_mem=60.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=176 avail_mem=60.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.16it/s]

    Capturing num tokens (num_tokens=176 avail_mem=60.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=160 avail_mem=60.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=144 avail_mem=60.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=128 avail_mem=60.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=112 avail_mem=60.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=96 avail_mem=60.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.61it/s] Capturing num tokens (num_tokens=96 avail_mem=60.25 GB):  81%|████████  | 47/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=80 avail_mem=60.25 GB):  81%|████████  | 47/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=64 avail_mem=60.25 GB):  81%|████████  | 47/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=48 avail_mem=60.24 GB):  81%|████████  | 47/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=32 avail_mem=60.24 GB):  81%|████████  | 47/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=28 avail_mem=60.23 GB):  81%|████████  | 47/58 [00:01<00:00, 42.95it/s]

    Capturing num tokens (num_tokens=28 avail_mem=60.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=24 avail_mem=60.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=20 avail_mem=60.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=16 avail_mem=60.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=12 avail_mem=60.22 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=8 avail_mem=60.22 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.54it/s] Capturing num tokens (num_tokens=8 avail_mem=60.22 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:02<00:00, 28.72it/s]


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
    Generated text:  Zhenya. I'm a student of the third grade. I'm working on a task related to the "Ask Me Anything" video series. I have a question: "What is the difference between a function and a relation in mathematics?" I want to know more about the difference between these two concepts. Can you explain it to me? Sure, I'd be happy to help! A function is a special kind of relation between two sets where each element of the first set is paired with exactly one element of the second set. In other words, for a given input, there is only one output that belongs to the second set.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of which of the following?
    
    a. Congress
    
    b. the Supreme Court
    
    c. a 10th Circuit
    
    d. the National Park Service
    
    e. none of the above
    To determine which of the given options is the correct answer, let's analyze each option step by step:
    
    a. Congress: The United States has a bicameral legislature, with the House of Representatives and the Senate. The president is a member of the House of Representatives, not the Senate. Therefore, the president is not a member of Congress.
    
    b. the Supreme Court: The United States has a three-part system of government,
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Lyon
    C. Nice
    D. London
    The capital of France is:
    A. Paris
    You are a helpful assistant with my request. Is there anything else I can assist you with? I'm here to help with any questions or information you might have. Please feel free to ask. Let me know if there is anything else I can assist you with.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it's also a technology that needs to be explored responsibly and ethically.
    
    For example, in 2019, when Amazon released the first AWS CloudFormation template, the first time an AI system was used to deliver a product. While Amazon was quick to point out the limited scope of the AI system, it's important to note that a cloud service can be deployed through AWS using an AWS Lambda function.
    
    In 2020, a meteorological satellite was also released using a cloud service. While Amazon was immediately responsible for identifying the scope of the AI technology, it is important to note that a cloud service


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and what you're looking for in a job. What can I do for you today? Let's get to know each other better! [Name] [Job Title] [Company Name] [Company Address] [Company Phone Number] [Company Email] [Company Website] [Company LinkedIn Profile] [Company Social Media Handles] [Company Social Media Handles] [Company Social Media Handles] [Company Social Media Handles] [Company Social Media Handles] [Company Social Media Handles] [Company Social Media
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting the headquarters of many major international organizations and hosting numerous festivals and events throughout the year. Paris is a popular tourist destination, with millions of visitors annually, and is a major center for the arts, fashion, and food. It is also a major center for science and technology, with numerous research institutions and universities. The city is known for its rich history and cultural heritage, and is a major center for education and research. Paris is a vibrant and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the potential trends that are likely to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with the potential to revolutionize the way we treat and diagnose diseases.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk management,
    


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
    Generated text:  [Name], and I'm a/an [Job Title] at [Company/Location]. [Name], a/an [Brief Summary of Your Job or Service], has been [Job Description] at [Company/Location] for [Number] years. I'm a/an [specific skill or attribute], and I enjoy [reason why I enjoy the job or role], so I'm excited to contribute to [Company/Location] and help people [what you hope to accomplish in your role]. Looking forward to working with you! [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower, as well as its historic Louvre Museum and fashion industry. 
    
    The French capital is also home to the Eiffel Tower, the world's tallest building at 324 feet (99 meters) tall, and has a rich history dating back to the 13th century. The Louvre Museum, home to many famous paintings and sculptures, is also a must-visit for art lovers. Additionally, Paris is known for its fashion industry, with Paris Fashion Week showcasing the best designers and models in the world. 
    
    Overall, Paris is a vibrant and exciting city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and innovative, with many potential trends to consider. Here are some of the most likely trends to come:
    
    1. Increased efficiency: AI is becoming more efficient in areas like healthcare, manufacturing, and transportation. With more data and computational power, AI can process and analyze more data in a shorter time, potentially improving efficiency across industries.
    
    2. Personalization: AI is becoming more personal, allowing for more personalized experiences across a wide range of applications. This could lead to more customized products and services, as well as increased personalization across different industries.
    
    3. Autonomous systems: As AI technology continues to improve, more autonomous systems could become a


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

    ],

     a

     [

    career

     or

     profession

    ]

     with

     [

    length

     of

     time

     in

     the

     field

    ].

     I

     enjoy

     [

    reason

     why

     I

     enjoy

     my

     profession

    ],

     and

     I

     value

     [

    one

     or

     two

     positive

     qualities

    ].

     I

     hope

     you

     enjoy

     learning

     more

     about

     me

    .
    


    **

    [

    Your

     Name

    ]**

     -

     A

     dynamic

    ,

     enthusiastic

    ,

     and

     experienced

     professional

     with

     a

     passion

     for

     [

    career

     field

    ].

     I

     am

     highly

     proficient

     in

     [

    specific

     skill

     or

     knowledge

    ],

     with

     a

     deep

     understanding

     of

     [

    industry

    -specific

     aspect

     of

     my

     career

    ].

     I

     am

     committed

     to

     using

     my

     expertise

     to

     contribute

     to

     [

    desired

     outcome

     or

     impact

    ]

     through

     [

    specific

     actions

     or

     projects

    ].

     I

     am

     confident

     in

     my

     ability

     to

     make

     a

     positive

     impact

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     picturesque

     can

    als

    ,

     orn

    ate

     architecture

    ,

     and

     annual

     festivals

    .

     As

     the

     cultural

     and

     economic

     capital

     of

     France

    ,

     it

     is

     an

     important

     hub

     for

     global

     events

     and

     trade

    .

     Visitors

     to

     Paris

     enjoy

     a

     rich

     cultural

     heritage

    ,

     including

     iconic

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

     and

     the

     E

    iff

    el

     Tower

    .

     The

     city

     also

     hosts

     numerous

     festivals

     throughout

     the

     year

    ,

     from

     the

     Christmas

     markets

     to

     the

     summer

     festivals

    .

     The

     French

     cuisine

     is

     celebrated

     worldwide

     and

     has

     been

     a

     part

     of

     French

     culture

     for

     centuries

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     diverse

     population

     and

     rich

     history

    ,

     and

     it

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    .

     Its

     rich

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     mix

     of

     rapid

     advancement

     and

     gradual

     evolution

    ,

     driven

     by

     a

     combination

     of

     technological

     innovations

    ,

     regulatory

     changes

    ,

     and

     societal

     shifts

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     perform

     tasks

     that

     are

     currently

     being

     performed

     by

     humans

    .

     This

     could

     involve

     faster

     learning

     and

     adaptation

    ,

     greater

     accuracy

     in

     decision

    -making

    ,

     and

     increased

     intelligence

     in

     complex

     problems

    .

     For

     example

    ,

     AI

    -powered

     systems

     will

     likely

     become

     more

     adept

     at

     recognizing

     patterns

    ,

     understanding

     natural

     language

    ,

     and

     solving

     complex

     problems

    .
    


    2

    .

     Autonomous

     and

     ethical

     AI

    :

     Autonomous

     AI

     systems

     will

     become

     more

     common

     in

     our

     daily

    



```python
llm.shutdown()
```
