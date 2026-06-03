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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.03it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.06it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.24it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 17.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.54it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 29.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 29.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 29.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 29.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 29.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.85it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.85it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.06it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:01, 27.64it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:01, 27.64it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.64it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.64it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.64it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.86it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.86it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.86it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.86it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.86it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=288 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.25it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=224 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=160 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  81%|████████  | 47/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.46it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.46it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 32.59it/s]


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
    Generated text:  Amy, a junior in high school. I have a few questions about a recent video that I watched about the history of democracy. Can you help me out?
    Sure, I'd be happy to help you out with your questions about the history of democracy. What video are you referring to? Please provide me with the link to the video. Once I have the link, I'll do my best to answer your questions about the history of democracy. Is there anything else you need help with? I'm here to assist you with any questions you may have. Let me know! [Pause] I apologize, but I'm not able to access
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to make some new science initiatives. For example, he wants to make science education a priority. He has announced a new plan to see if it will work. He chose to take a group of students to the National Science Center, located in Washington, D. C. . It's a new scientific museum. At the Science Center, they will meet with a scientist. They will learn about his work in the past and talk with him about his future plans. The students will also visit a lab with a special machine that can make models of objects. The machine will be used to create models of many different types of objects. One of
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]. ____
    A. Paris
    B. Lille
    C. Lyon
    D. Reims
    Answer:
    
    A
    
    In the process of trial operations, for the first time, what should a company do for its subsidiaries to start conducting trial operations? 
    A. Make a feasibility study report
    B. Complete registration with relevant administrative departments
    C. Apply for a trial operation permit
    D. Conduct a trial operation report
    Answer:
    
    C
    
    In the process of trial operation, for the first time, a company should apply for a trial operation permit to the ___.
    A. Construction administrative department
    B. Quality supervision and administration
    ===============================
    Prompt: The future of AI is
    Generated text:  changing fast, and more and more people are concerned about the security of their personal data. Fortunately, there is a tool that can help us solve this problem: an AI-powered data encryption tool. This tool, which is called Secure ENCRYPTION, is a tool that can encrypt and decrypt sensitive data in real-time. It can also protect your data from unauthorized access, ensuring that your personal information remains safe and confidential.
    The process of using Secure ENCRYPTION is simple. First, you need to create a secure key. This key is used to encrypt and decrypt the data. Once you have a secure key, you can use it to


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast who has been [Number of Years] years in this field. I'm always looking for new challenges and opportunities to learn and grow. I'm always eager to share my knowledge and experience with others. I'm a [Favorite Activity] lover who enjoys [Number of Activities] and [Number of Books] on the subject. I'm a [Favorite Book] lover who reads [Number of Books] a year and [Number of Books] a month. I'm a [Favorite Music] lover who listens to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is home to many iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its cuisine, fashion, and art scene, making it a popular tourist destination. The city is also home to many cultural institutions, including the Louvre Museum, the Musée d'Orsay, and the Musée d'Orsay. Paris is a vibrant and dynamic city that continues to be a major center of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve risk management, fraud detection, and portfolio optimization
    


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
    Generated text:  [Name]. I'm a [age] year old [gender] [occupation]. I'm passionate about [what interests me]. I enjoy [activities that make me happy]. My favorite movie genre is [movie genre]. What's your favorite hobby, and what's your current location? I'm from [country]. That's all I can think of right now. I'm looking forward to meeting you! [Do not include any personal information like my name or age in your introduction. Focus on the personality of the character. Let the audience imagine a character with whom you can connect based on this introduction. This prompt encourages creativity and invites the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history, a thriving economy, and a diverse population of around 10 million people. Paris is also known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is home to many different cultural institutions, including the Louvre, the museum of modern art, and the Musée d'Orsay. Paris is a popular tourist destination, attracting millions of visitors each year from around the world. The city is also known for its fashion industry, which is a major part of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of changes and advancements over the coming decades. Some potential trends that may be seen in AI include:
    
    1. Increased automation: AI is already being used to automate tasks, from customer service to production, but it's likely to continue to become more prevalent in future. This could lead to the widespread adoption of AI-powered automation systems, freeing up human workers to focus on more complex and creative tasks.
    
    2. Autonomous agents: AI is already able to perform tasks that require human intelligence, such as playing chess or driving a car. As AI continues to improve, it's likely to become capable of performing tasks that were previously


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

     [

    Country

    ]

     and

     I

     am

     an

     [

    occupation

    ].

     I

    've

     always

     been

     fascinated

     by

     [

    something

     or

     someone

    ]

     and

     have

     always

     been

     interested

     in

     [

    something

     or

     someone

    ].

     I

     am

     always

     on

     the

     lookout

     for

     new

     experiences

     and

     adventure

    ,

     so

     I

     have

     been

     traveling

     the

     world

     for

     years

     now

    .

     I

     enjoy

     [

    something

     or

     someone

    ]

     and

     have

     always

     felt

     that

     my

     work

     and

     life

     are

     in

     harmony

     with

     each

     other

    .

     I

    'm

     always

     looking

     for

     a

     new

     challenge

     and

     adventure

    ,

     and

     I

    'm

     always

     eager

     to

     explore

     new

     places

     and

     new

     things

    .

     I

    'm

     always

     ready

     to

     start

     a

     new

     adventure

     and

     to

     learn

     new

     things

    .

     What

    's

     your

     favorite

     hobby

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     the

     economic

     and

     cultural

     center

     of

     the

     country

    .

     Paris

     is

     also

     one

     of

     the

     world

    ’s

     most

     famous

     cities

     and

     hosts

     many

     tourist

     attractions

    .

     The

     city

     is

     located

     on

     the

     banks

     of

     the

     River

     Se

    ine

     and

     is

     known

     for

     its

     romantic

     architecture

    ,

     delicious

     food

    ,

     and

     world

    -class

     museums

    .

     It

     is

     also

     home

     to

     many

     famous

     landmarks

     and

     cultural

     institutions

     such

     as

     the

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

     Lou

    vre

     Museum

    .

     Paris

     is

     a

     popular

     tourist

     destination

     for

     both

     locals

     and

     tourists

     alike

    .

     The

     city

     has

     a

     diverse

     and

     multicultural

     population

    ,

     with

     many

     French

    -speaking

     communities

     and

     visitors

     from

     around

     the

     world

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     many

     more

     complex

    ,

     sophisticated

     algorithms

     and

     systems

     that

     can

     process

     and

     analyze

     vast

     amounts

     of

     data

     in

     real

    -time

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     can

     be

     used

     to

     analyze

     medical

     images

    ,

     diagnose

     diseases

    ,

     and

     even

     assist

     in

     drug

     discovery

    .

     It

     could

     also

     be

     used

     to

     improve

     patient

     care

     and

     reduce

     costs

    .
    


    2

    .

     Enhanced

     self

    -driving

     cars

    :

     AI

     is

     already

     being

     used

     in

     self

    -driving

     cars

    ,

     but

     there

     are

     still

     many

     technological

     challenges

     to

     overcome

    .

     As

     AI

     technology

     continues

     to

     improve

    ,

     it

     is

     likely

     that

     self

    -driving

     cars

     will

     become

     more

     common

     and

     more

     reliable

    .
    


    3

    .

     Personal

    ized

     medicine

    :

    



```python
llm.shutdown()
```
