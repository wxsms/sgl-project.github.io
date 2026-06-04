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

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.47it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.40it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.04it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.04it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.04it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.04it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.04it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.04it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.04it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.04it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 24.04it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 24.04it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 24.04it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.13it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.13it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.13it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.13it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.13it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.13it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.13it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.18 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.18 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.18 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.18 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.18 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.95it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.95it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=53.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.15 GB):  21%|██        | 12/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.14 GB):  21%|██        | 12/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.14 GB):  21%|██        | 12/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.14 GB):  21%|██        | 12/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.14 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.14 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.13 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.13 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=53.13 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=960 avail_mem=53.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.83it/s] Capturing num tokens (num_tokens=896 avail_mem=53.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=896 avail_mem=53.12 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=832 avail_mem=53.11 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=768 avail_mem=53.11 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=704 avail_mem=53.11 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.87it/s]

    Capturing num tokens (num_tokens=640 avail_mem=53.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=640 avail_mem=53.10 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.42it/s]Capturing num tokens (num_tokens=576 avail_mem=53.10 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.42it/s]Capturing num tokens (num_tokens=512 avail_mem=53.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.42it/s]Capturing num tokens (num_tokens=480 avail_mem=53.10 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.42it/s]Capturing num tokens (num_tokens=448 avail_mem=53.10 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.42it/s]Capturing num tokens (num_tokens=416 avail_mem=53.10 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=416 avail_mem=53.10 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=384 avail_mem=53.10 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.55it/s]

    Capturing num tokens (num_tokens=352 avail_mem=53.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=320 avail_mem=53.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=288 avail_mem=53.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=288 avail_mem=53.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=256 avail_mem=53.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=240 avail_mem=53.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=224 avail_mem=53.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=208 avail_mem=53.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=192 avail_mem=53.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.11it/s]

    Capturing num tokens (num_tokens=192 avail_mem=53.07 GB):  71%|███████   | 41/58 [00:01<00:00, 36.34it/s]Capturing num tokens (num_tokens=176 avail_mem=53.06 GB):  71%|███████   | 41/58 [00:01<00:00, 36.34it/s]Capturing num tokens (num_tokens=160 avail_mem=53.06 GB):  71%|███████   | 41/58 [00:01<00:00, 36.34it/s]Capturing num tokens (num_tokens=144 avail_mem=53.06 GB):  71%|███████   | 41/58 [00:01<00:00, 36.34it/s]Capturing num tokens (num_tokens=128 avail_mem=53.06 GB):  71%|███████   | 41/58 [00:01<00:00, 36.34it/s]Capturing num tokens (num_tokens=128 avail_mem=53.06 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.49it/s]Capturing num tokens (num_tokens=112 avail_mem=53.06 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.49it/s]Capturing num tokens (num_tokens=96 avail_mem=53.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.49it/s] Capturing num tokens (num_tokens=80 avail_mem=53.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.49it/s]

    Capturing num tokens (num_tokens=64 avail_mem=53.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.49it/s]Capturing num tokens (num_tokens=64 avail_mem=53.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=48 avail_mem=53.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=32 avail_mem=53.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=28 avail_mem=53.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=24 avail_mem=53.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=20 avail_mem=53.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=20 avail_mem=53.03 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.25it/s]Capturing num tokens (num_tokens=16 avail_mem=53.03 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.25it/s]Capturing num tokens (num_tokens=12 avail_mem=53.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.25it/s]Capturing num tokens (num_tokens=8 avail_mem=53.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.25it/s] Capturing num tokens (num_tokens=4 avail_mem=53.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.25it/s]

    Capturing num tokens (num_tokens=4 avail_mem=53.01 GB): 100%|██████████| 58/58 [00:01<00:00, 33.70it/s]


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
    Generated text:  Ashley and I am a 14 year old female from Australia. I have been diagnosed with cerebral palsy. I have had no more than 3 interventions for it and I have been in therapy with a physiotherapist since age 16. I have had my feet treated for a condition of flat feet that has been going on since childhood. I have not been treated for this condition with any kind of therapy. My father has had a heart attack in the past that has left him with a heart condition. He had a pacemaker implanted when he was 47. I was not present when that happened. He has
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, but how important can he really be? One reason why many people believe that the president is important is that the president makes the laws and the executive branch makes the laws that the other branches of government can then execute. It is also important to note that the president holds a high status in the US government because of the power they hold. The president can decide what jobs should be held by what levels of government officials, and they also have the power to issue executive orders. To read more, click on the image below. Yes, the president can actually make laws, and the president also has a lot of power.
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Berlin
    C. Moscow
    D. London
    Answer:
    
    A
    
    The correct method for transforming the equation 3x - 5y = 20 into the form of y = mx + b is:
    A. Convert to slope-intercept form
    B. Convert to intercept form
    C. Convert to standard form
    D. Convert to point-slope form
    Answer:
    
    C
    
    Consider the following three propositions about two lines in space:
    
    ① If line a is parallel to plane α, and line b is parallel to plane β, then a is parallel to b;
    
    ② If
    ===============================
    Prompt: The future of AI is
    Generated text:  more than just a bunch of artificial intelligence machines with the ability to think like humans. The future of AI is in the future of the human race. AI is not just about machines, it's about the human race. Here are 7 ways that AI is changing the way we live our lives:
    
    1. Personalized AI: With AI, we can now get personalized care for those with chronic illness. AI can analyze patient data and make recommendations to treat the patient, rather than treating the patient as a one-size-fits-all solution.
    
    2. Virtual Assistants: AI can also be used to create virtual assistants that can help us with


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the European Union and the third largest city in the world by population. It is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich cultural heritage, including its museums, theaters, and art galleries, and is a major center for fashion, music, and food. The city is also home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a vibrant and diverse
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective AI systems that can handle a wider range of tasks.
    
    3. Increased focus on ethical and social implications: As AI becomes
    


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
    Generated text:  [Name], and I am an [Age] year old [Gender] [Occupation]. I have [number] years of experience in [field] and [number] years of experience in [field]. I am currently [current position], and I have been working in [industry] for [number] years. I have always been passionate about [specific interest], and I enjoy [specific hobby, interest, or skill]. I believe in [value proposition, if applicable]. I am excited to meet you! Can you tell me more about your background and why you are interested in our industry? Additionally, what specific skills or experiences do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    **Step 1: Identify the key elements of the statement**
    
    - **City Name**: Paris
    - **Country**: France
    - **Location**: Capital of France
    
    **Step 2: Determine if any additional information is necessary**
    
    - No additional information is necessary beyond the basic elements of the statement.
    
    **Step 3: Formulate the factual statement**
    
    - **Factual Statement**: Paris is the capital city of France.
    
    This statement is concise, accurate, and provides the essential information required to understand the capital city of France. 
    
    **Answer**: Paris is the capital city of France. (30 words) 
    
    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a combination of rapid progress and rapid decline. Some of the key trends that are expected to shape the AI landscape include:
    
    1. Increased focus on ethical AI: There is a growing recognition of the need to balance the potential benefits of AI with its potential risks. This includes considerations of bias, privacy, and transparency.
    
    2. AI becomes more autonomous: AI systems are likely to become more autonomous, with a focus on developing systems that can operate with minimal human intervention.
    
    3. AI becomes more integrated into everyday life: AI is already playing a growing role in our daily lives, from voice assistants to self-driving cars.


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

     Sarah

    ,

     and

     I

    'm

     a

     

    2

    8

    -year

    -old

     digital

     marketing

     professional

     with

     a

     passion

     for

     helping

     businesses

     grow

     and

     thrive

    .

     I

    'm

     a

     data

    -driven

     individual

     who

     thr

    ives

     on

     finding

     innovative

     ways

     to

     streamline

     processes

     and

     enhance

     user

     experiences

    .

     With

     a

     proven

     track

     record

     of

     creating

     engaging

     content

     and

     driving

     results

    ,

     I

    'm

     eager

     to

     bring

     my

     skills

     to

     bear

     on

     your

     projects

    .

     Let

    's

     connect

     to

     learn

     more

     about

     how

     we

     can

     work

     together

     to

     achieve

     your

     business

     goals

    !

     

    🌟

    ✨

    
    


    Your

     message

     has

     been

     sent

     successfully

    !

     

    🚀

    ✨

    
    


    What

     is

     your

     favorite

     hobby

     or

     activity

    ,

     and

     how

     do

     you

     maintain

     it

    ?

     As

     someone

     who

     enjoys

     exploring

     new

     experiences

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    City

     Name

    :

     Paris

    


    Country

    :

     France

    


    Capital

    :

     Paris

    


    Currency

    :

     Euro

    


    Major

     Language

    :

     French

    ,

     Spanish

    ,

     English

    ,

     etc

    .

     


    Year

     of

     Independence

    :

     

    1

    7

    9

    2

    


    Population

    :

     

    2

    ,

     

    6

    7

    7

    ,

     

    6

    2

    4

     (

    as

     of

     

    2

    0

    2

    1

    )


    Capital

    :

     Paris

    


    Capital

    :

     Paris

    


    Latitude

    :

     

    4

    8

    °

     N

    


    Longitude

    :

     

    2

    °

     E

    


    Official

     Language

    :

     French

    


    Official

     Language

    :

     French

    


    G

    DP

    :

     $

    5

    8

    5

    ,

     

    2

    8

    6

    ,

     

    0

    0

    0

    ,

    0

    0

    0

     (

    as

     of

     

    2

    0

    2

    1

    )


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     shaped

     by

     a

     number

     of

     trends

     and

     developments

     that

     will

     shape

     its

     potential

     applications

    ,

     development

    ,

     and

     impacts

    .

     Here

     are

     some

     of

     the

     most

     likely

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     transparency

    :

     As

     AI

     systems

     become

     more

     sophisticated

     and

     complex

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     making

     them

     more

     transparent

     and

     accountable

     to

     their

     users

     and

     stakeholders

    .

     This

     will

     require

     an

     increased

     focus

     on

     ethical

     and

     social

     factors

    ,

     such

     as

     bias

    ,

     fairness

    ,

     and

     accountability

    .
    


    2

    .

     Expansion

     of

     AI

     applications

    :

     AI

     is

     increasingly

     being

     used

     in

     various

     industries

     and

     applications

     beyond

     just

     data

     analysis

    ,

     such

     as

     autonomous

     vehicles

    ,

     smart

     homes

    ,

     and

     virtual

     assistants

    .

     As

    



```python
llm.shutdown()
```
