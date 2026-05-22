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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.60it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.62it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]

    Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.36it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.06 GB):   3%|▎         | 2/58 [00:00<00:03, 17.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.04 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.14it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  21%|██        | 12/58 [00:00<00:01, 27.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 27.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 27.84it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 27.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  26%|██▌       | 15/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  26%|██▌       | 15/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.75it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.75it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.75it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.75it/s]

    Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.75it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.60it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.60it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.60it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.60it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.60it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.60it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.01it/s]

    Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  60%|██████    | 35/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  60%|██████    | 35/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  60%|██████    | 35/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  60%|██████    | 35/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  60%|██████    | 35/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  60%|██████    | 35/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.96it/s]

    Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=112 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.83it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.29it/s]

    Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.50it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 33.70it/s]


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
    Generated text:  Sara, and I’m from the United States. I’m currently a research student at the University of Michigan. For this internship, I’m interested in the field of the design of manufacturing processes for the semiconductor industry. The semiconductor industry is a leading global market, and it’s critical that the process of manufacturing semiconductor components is efficient and cost-effective. As a research student, I would like to gain a deep understanding of the design and optimization of semiconductor manufacturing processes, as well as to apply my knowledge and skills to improve the efficiency and cost-effectiveness of the semiconductor industry.
    As an intern, I have been working on various projects related to semiconductor
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. The world’s greatest leader is also a man. The world’s greatest leader is also the leader of the United States. The United States has the highest rate of murder and the most unemployment. The president of the United States has been assassinated on three separate occasions. He was assassinated 7 times during his term of office. The president of the United States is considered a criminal. He is also considered a leader. He is a man who is a criminal and a leader. What is the correct logical connection between these statements?
    A) If the president of the United States is a man, then the world’s greatest leader
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    
    To determine the capital of France, we can follow these steps:
    
    1. Identify the capital of France. The capital of France is Paris.
    2. Verify the capital's position relative to other European countries. The capital of France is indeed located in the south-central part of France, and it is situated on the Loire River.
    
    Therefore, the capital of France is \(\boxed{Paris}\).
    ===============================
    Prompt: The future of AI is
    Generated text:  fast moving in a number of directions, and the report published in its entirety covers this topic. AI has been considered one of the most promising and transformative technologies to emerge since the introduction of the Internet in the mid-2000s. It has already undergone a rapid evolution, with a number of breakthroughs in areas such as machine learning, natural language processing, and computer vision. However, the future of AI is far from certain, and the report highlights the need for continued development and improvement of the technology. The report also acknowledges that while the AI world is rapidly evolving, it is also facing challenges, such as the need for


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and art scene. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, with a diverse population of over 2 million residents. Paris is a city of love
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see more automation and robotics in various industries, from manufacturing to healthcare. This could lead to job displacement and changes in work patterns, but it could also create new job opportunities and opportunities for innovation.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we may see even more sophisticated AI-powered healthcare solutions,
    


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
    Generated text:  [Name] and I'm [Age] years old. I'm a [occupation], and I bring [unique skill or attribute]. How can I help you today? [Introduction] Let me know what you need, and I'll do my best to help. [End of introduction] 
    
    Note: Replace [Name], [Age], [Occupation], [unique skill or attribute], and [introduction] with the appropriate information and details for your character. The self-introduction should be neutral and informative, providing an opportunity for the reader to get to know the character better and build a connection. The introduction should be short and to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an historic city that is renowned for its vibrant art, culture, and sophisticated nightlife. Paris is a metropolis with a population of around 10 million people, and it is the most populous city in the European Union and the world's third most populous city after Beijing and Tokyo. The city is home to iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral, and it is the economic and cultural center of France. Paris has a rich history dating back to the 6th century and has undergone numerous architectural and cultural transformations over the centuries. The city is a UNESCO World Heritage Site and is celebrated for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bound to be dominated by a combination of new technologies and complex human-machine interactions. Here are a few possible trends we can expect to see in the AI landscape in the coming years:
    
    1. AI becomes more ubiquitous: As more companies and organizations adopt AI technologies for automation and other purposes, we're likely to see a rise in AI's use. This could lead to a wider adoption of AI, as well as a greater integration of AI into different industries.
    
    2. AI will become more personalized: As AI systems become better at learning and adapting, we may see more personalized experiences and interactions with AI systems. This could lead to new ways of


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

    Character

     Name

    ].

     I

    'm

     a

     [

    Professional

     Title

    ]

     with

     [

    Professional

     Title

    ]

     experience

     in

     [

    Industry

    /

    Field

    ],

     specializing

     in

     [

    Your

     specialty

     or

     expertise

    ].

     I

    'm

     always

     eager

     to

     learn

     and

     adapt

     to

     new

     challenges

     and

     opportunities

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     contribute

     to

     [

    Your

     goal

     or

     mission

    ].

     I

    'm

     always

     up

     for

     a

     challenge

     and

     I

    'm

     willing

     to

     go

     the

     extra

     mile

     to

     achieve

     my

     goals

    .

     What

    's

     your

     name

    ,

     and

     what

     brings

     you

     here

     today

    ?

     [

    Character

     Name

    ]

     Welcome

    !

     I

    'm

     excited

     to

     meet

     you

     and

     to

     discuss

     how

     I

     can

     contribute

     to

     your

     success

    .

     Let

    's

     get

     started

    !

     [

    Character

     Name

    ].

     [

    Tell

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Î

    le

     de

     la

     C

    ité

     and

     the

     Se

    ine

     River

    .

     The

     city

     is

     a

     major

     cultural

     and

     economic

     center

     with

     many

     museums

    ,

     galleries

    ,

     and

     theaters

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     rich

     history

    ,

     picturesque

     architecture

    ,

     and

     world

    -ren

    owned

     food

     culture

    .

     France

    ’s

     capital

     is

     also

     known

     for

     its

     scenic

     views

     of

     the

     city

     from

     its

     towers

     and

     bridges

    .

     The

     city

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     the

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

     With

     its

     diverse

     population

     and

     vibrant

     culture

    ,

     Paris

     is

     a

     popular

     destination

     for

     both

     locals

     and

     tourists

     alike

    .

     
    


    ##

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     will

     likely

     evolve

     rapidly

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     AI

     Transparency

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     become

     more

     transparent

     and

     understandable

     to

     humans

    .

     This

     will

     help

     to

     reduce

     the

     risk

     of

     bias

     in

     the

     algorithms

     that

     power

     AI

     systems

    ,

     and

     make

     it

     easier

     for

     individuals

     to

     understand

     and

     trust

     the

     AI

     systems

     that

     they

     interact

     with

    .
    


    2

    .

     AI

     Personal

    ization

    :

     AI

     will

     continue

     to

     become

     more

     personal

    ,

     allowing

     for

     more

     accurate

     predictions

     and

     recommendations

    .

     This

     will

     require

     more

     advanced

     machine

     learning

     algorithms

     that

     can

     understand

     and

     interpret

     large

     amounts

     of

     data

     to

     provide

     personalized

    



```python
llm.shutdown()
```
