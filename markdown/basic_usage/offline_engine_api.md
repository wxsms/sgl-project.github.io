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

    [W526 22:02:42.408200734 socket.cpp:207] [c10d] The hostname of the client socket cannot be retrieved. err=-3


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.18it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.10it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.17it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.17it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 22.17it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 22.17it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 22.17it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 22.17it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 22.17it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 22.17it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 22.17it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 22.17it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   2%|▏         | 1/58 [00:00<00:07,  7.94it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   2%|▏         | 1/58 [00:00<00:07,  7.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   2%|▏         | 1/58 [00:00<00:07,  7.94it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   2%|▏         | 1/58 [00:00<00:07,  7.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   7%|▋         | 4/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   7%|▋         | 4/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.45it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.40it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  31%|███       | 18/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 29.48it/s] Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.26it/s]

    Capturing num tokens (num_tokens=640 avail_mem=72.20 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=640 avail_mem=72.20 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.31it/s]Capturing num tokens (num_tokens=576 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.31it/s]Capturing num tokens (num_tokens=512 avail_mem=71.96 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.31it/s]Capturing num tokens (num_tokens=480 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.31it/s]Capturing num tokens (num_tokens=448 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=416 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=416 avail_mem=71.98 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=384 avail_mem=71.97 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=352 avail_mem=71.97 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=320 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.55it/s]

    Capturing num tokens (num_tokens=288 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=256 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=256 avail_mem=71.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=240 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=224 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=208 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=192 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=176 avail_mem=71.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=176 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=160 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=144 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=128 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.71it/s]

    Capturing num tokens (num_tokens=112 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=96 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.71it/s] Capturing num tokens (num_tokens=96 avail_mem=71.93 GB):  81%|████████  | 47/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=80 avail_mem=71.92 GB):  81%|████████  | 47/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=64 avail_mem=71.92 GB):  81%|████████  | 47/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=48 avail_mem=71.92 GB):  81%|████████  | 47/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=32 avail_mem=71.91 GB):  81%|████████  | 47/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=28 avail_mem=71.91 GB):  81%|████████  | 47/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=28 avail_mem=71.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=24 avail_mem=71.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=20 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.09it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=12 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=8 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.09it/s] Capturing num tokens (num_tokens=8 avail_mem=71.90 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB): 100%|██████████| 58/58 [00:01<00:00, 35.30it/s]


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
    Generated text:  Ashley Miller. I have a BS degree in Mechanical Engineering from the University of Tennessee, Knoxville, and I am currently working towards a Master of Business Administration degree in the area of finance at the University of Tennessee, Knoxville.
    What led you to choose mechanical engineering, and what did you find particularly exciting about your studies?
    My undergraduate education was primarily in mechanical engineering at the University of Tennessee, Knoxville, and I was very excited about the engineering aspect of my degree, which ultimately led me to consider pursuing a career in the field. I enjoyed the concept of making tangible objects (like engines or machines) and the technical aspects of engineering design. However
    ===============================
    Prompt: The president of the United States is
    Generated text:  expected to address the nation on Thursday in a statement about the climate crisis. The president will also call on the world to get serious about reducing the damage caused by climate change. But the most important part of the president’s statement will be his words on the environment, which will come as no surprise. The president’s message will be that he is serious about taking care of the earth. He will also call for a change to the way we produce and consume energy. The president will tell us that we must make the switch to renewable sources of energy. The president will also ask us to think about taking action to reduce greenhouse gases. He will
    ===============================
    Prompt: The capital of France is
    Generated text:  _____. A: Paris B: Lyon C: Strasbourg D: Lyon
    The correct answer is A: Paris. Paris is the capital and largest city of France. Lyon is the second largest city in France, and Strasbourg is a historic city in France. The other options, while located in France, are not capitals. Therefore, Paris is the capital city of France. 
    
    So the capital of France is: **A: Paris**. 
    
    I apologize, but the options provided do not include Paris as a capital. The correct answer should be Paris. But based on the options given, I can infer that the question might have
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and the stakes are high. Companies are scrambling to build tools and services that make better decisions, improve patient outcomes and save money. But in order to do so, they need to understand the world around them. To do this, companies need data. Data is everywhere. And yet, access to data is not as common as you might think. While organizations have access to rich datasets through open source tools and cloud providers, the reality is that businesses often have little control over who has access to their data, and what they look at.
    
    One solution to this problem is open data. Open data is data that is freely available and publicly accessible


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few details about your personality, skills, or interests that you can share with me]. I'm looking forward to meeting you and discussing how I can help you. How can I assist you today? [Name] is looking to learn more about [insert a few details about the topic you're discussing]. I'm here to help you understand the topic better. How can I assist you today? [Name] is looking to learn more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Riviera, a popular tourist destination for its beaches and luxury resorts. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and preferences. This could lead to more personalized and effective AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI systems become
    


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
    Generated text:  [Name], and I am a computer scientist with a passion for AI and machine learning. My current projects include developing algorithms to enhance natural language processing, and exploring new methods for natural language understanding. I also love to travel and explore new cities, as it helps me stay creative and open-minded. I'm always looking for ways to improve myself and stay up-to-date with the latest trends in technology and science. Thank you! [Name] is a neutral self-introduction for the fictional character. The response does not include any personal information or bias and is presented in a neutral and factual manner. The use of "neutral" also implies that
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, also known as "La Tombeau de la vie de Louis XVI," is the capital city of France and the largest city and most populous urban area in the country. It is located in the Loire Valley and is situated in the heart of the Ile de la Cité. The city has a population of around 2. 7 million, with the majority of the population residing in the city centre. Paris is famous for its famous landmarks, including the Eiffel Tower, the Notre-Dame Cathedral, and the Louvre Museum. It is a major cultural and economic hub, hosting the world-renowned
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, with many different areas where it could have a significant impact. Here are some potential trends:
    
    1. Increased focus on ethical AI: As more people are becoming concerned about the impact of AI on society, there will be a greater focus on ethical considerations. AI researchers will work to develop algorithms that are designed to minimize harm to individuals and minimize the potential for misuse or abuse.
    
    2. Integration of AI with traditional industries: AI is already having a significant impact on traditional industries, such as healthcare, finance, and manufacturing. As the technology continues to advance, we can expect more integration between AI and these industries, with AI becoming


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

     __

    ________

    _

     and

     I

    'm

     a

    /an

     [

    fill

     in

     your

     occupation

     here

    ]

    !
    


    I

    'm

     currently

     working

     in

     [

    where

     you

     find

     yourself

     in

     the

     company

    ],

     and

     I

    'm

     looking

     for

     a

     role

     that

     allows

     me

     to

     contribute

     to

     the

     team

     and

     make

     a

     positive

     impact

     on

     [

    mention

     the

     company

     or

     industry

     you

     work

     in

    ].

     I

     believe

     that

     my

     unique

     skills

     and

     experiences

     make

     me

     the

     ideal

     candidate

     for

     this

     position

    ,

     and

     I

    'm

     excited

     about

     the

     opportunity

     to

     work

     alongside

     [

    insert

     colleagues

     or

     clients

    ].

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    ,

     and

     I

    'm

     always

     open

     to

     new

     experiences

     and

     ideas

    .
    


    Thank

     you

     for

     considering

     me

     for

     this

     position

    !

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     True

    


    B

    .

     False

    
    


    A

    .

     True

    


    Paris

     is

     the

     capital

     city

     of

     France

    .

     It

     is

     located

     in

     the

     heart

     of

     the

     French

     countryside

     and

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     annual

     world

    -ren

    owned

     festivals

    .

     Paris

     is

     also

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    ,

     with

     millions

     of

     visitors

     annually

    .

     It

     is

     the

     cultural

    ,

     political

    ,

     and

     economic

     hub

     of

     France

     and

     plays

     a

     central

     role

     in

     the

     country

    's

     history

    ,

     economy

    ,

     and society

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

     and

     museums

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Arc

     de

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     bright

     and

     there

     are

     many

     possible

     paths

     it

     could

     take

    .

     Here

     are

     a

     few

     potential

     trends

     that

     are

     likely

     to

     shape

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     Use

     of

     AI

     for

     Medical

     Diagnosis

     and

     Treatment

    :

     AI

     is

     being

     used

     more

     and

     more

     in

     medical

     diagnosis

     and

     treatment

    .

     It

     can

     analyze

     vast

     amounts

     of

     medical

     data

     to

     identify

     patterns

     and

     make

     predictions

     about

     patient

     outcomes

    .

     AI

     is

     also

     being

     used

     to

     develop

     personalized

     treatment

     plans

     for

     each

     patient

    .
    


    2

    .

     AI

     in

     Manufacturing

    :

     AI

     is

     being

     used

     in

     manufacturing

     to

     improve

     efficiency

     and

     reduce

     errors

    .

     For

     example

    ,

     AI

     can

     be

     used

     to

     analyze

     production

     data

     to

     identify

     patterns

     and

     optimize

     processes

    .

     It

     can

     also

     be

     used

    



```python
llm.shutdown()
```
