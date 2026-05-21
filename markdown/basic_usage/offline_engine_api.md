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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.55it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.19it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.34it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.85it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.85it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.85it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.85it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.85it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.85it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.85it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.60 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.59 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.59 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.59 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.59 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.58 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=75.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.31it/s]Capturing num tokens (num_tokens=960 avail_mem=75.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.31it/s] Capturing num tokens (num_tokens=896 avail_mem=75.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.31it/s]

    Capturing num tokens (num_tokens=832 avail_mem=75.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.31it/s]Capturing num tokens (num_tokens=832 avail_mem=75.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=768 avail_mem=75.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=704 avail_mem=75.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=640 avail_mem=75.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=576 avail_mem=75.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=512 avail_mem=75.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=512 avail_mem=75.50 GB):  50%|█████     | 29/58 [00:00<00:00, 43.91it/s]Capturing num tokens (num_tokens=480 avail_mem=75.51 GB):  50%|█████     | 29/58 [00:00<00:00, 43.91it/s]Capturing num tokens (num_tokens=448 avail_mem=75.51 GB):  50%|█████     | 29/58 [00:00<00:00, 43.91it/s]Capturing num tokens (num_tokens=416 avail_mem=75.51 GB):  50%|█████     | 29/58 [00:00<00:00, 43.91it/s]Capturing num tokens (num_tokens=384 avail_mem=75.51 GB):  50%|█████     | 29/58 [00:00<00:00, 43.91it/s]

    Capturing num tokens (num_tokens=352 avail_mem=75.50 GB):  50%|█████     | 29/58 [00:00<00:00, 43.91it/s]Capturing num tokens (num_tokens=352 avail_mem=75.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.45it/s]Capturing num tokens (num_tokens=320 avail_mem=75.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.45it/s]Capturing num tokens (num_tokens=288 avail_mem=75.49 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.45it/s]Capturing num tokens (num_tokens=256 avail_mem=75.49 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.45it/s]Capturing num tokens (num_tokens=240 avail_mem=75.49 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.45it/s]Capturing num tokens (num_tokens=224 avail_mem=75.48 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.45it/s]Capturing num tokens (num_tokens=224 avail_mem=75.48 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=208 avail_mem=75.48 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=192 avail_mem=75.21 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=176 avail_mem=75.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=160 avail_mem=75.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.78it/s]

    Capturing num tokens (num_tokens=144 avail_mem=75.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=144 avail_mem=75.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=128 avail_mem=74.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=112 avail_mem=74.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=96 avail_mem=74.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.38it/s] Capturing num tokens (num_tokens=80 avail_mem=74.48 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=64 avail_mem=74.48 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=64 avail_mem=74.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=48 avail_mem=74.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=32 avail_mem=74.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=28 avail_mem=74.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=24 avail_mem=74.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.48it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.46 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=20 avail_mem=74.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=16 avail_mem=74.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=12 avail_mem=74.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=8 avail_mem=74.45 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s] Capturing num tokens (num_tokens=4 avail_mem=74.45 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=4 avail_mem=74.45 GB): 100%|██████████| 58/58 [00:01<00:00, 42.27it/s]


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
    Generated text:  Claudia and I'm a photographer from Australia. I'm in love with capturing moments of the moment and I'm also in love with making art. I work in a studio in Sydney, New South Wales. My studio is a beautiful, 2 bedroom, 2 bathroom home. I have a good level of expertise in making art in 2 forms: painting and sculpture. I also have a passion for photography and I'm the owner of the photography studio. I've had my own studio for 15 years and I'm always learning and growing in my craft. I hope you find my work and your gallery to be interesting and fun
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He likes the idea of 50 bases, but wants to keep that number below 150. He also likes the idea of having 150 military bases, but wants to keep that number above 70. Finally, he likes the idea of 250 military bases, but wants to keep that number even with the most military bases. How many different ways can he arrange these bases?
    
    To determine the number of different ways the president of the United States can arrange the bases, we need to analyze the given constraints step by step.
    
    1. **Maximum number
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 错误
    B. 正确
    答案:
    
    B
    
    皮肤黏膜淋巴结综合征（急性淋巴细胞白血病）的严重者，其死亡率为
    A. 40%
    B. 20%
    C. 10%
    D. 30%
    E. 15%
    答案:
    
    D
    
    在分类变量的分类中有多个分组，当变量值落在两个或两个以上的分组区间时，它们之间是
    A. 不同的
    B. 相同的
    C. 无法确定的
    D. 相同或不同的
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be shaped by the future of the software industry, and how it is developing. This means that the future of AI will not be defined by what we currently know, but by what we know in the coming years. This article discusses the potential impact of AI in the software industry, and how it is expected to change the way we work and live.
    The article discusses several potential areas where AI can impact the software industry, including the development of AI algorithms, machine learning, and natural language processing. It also discusses how these technologies are currently being developed, and how they are expected to evolve in the coming years.
    One of the key


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique characteristic or skill that sets me apart from others]. And what's your name? I'm [insert your name]. I'm always looking for new opportunities to grow and learn, and I'm eager to contribute to [company name] and [job title]. What's your favorite hobby or activity? I love [insert something fun or relaxing that you enjoy doing]. And what's your favorite book or movie? I love [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and being a major hub for international trade and diplomacy. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its food, fashion, and music scenes, and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial general intelligence: As AI technology continues to advance, we can expect to see more automation and the development of AI that can perform tasks that were previously done by humans. This could lead to the creation of more efficient and effective systems that can perform a wide range of tasks, from manufacturing to healthcare.
    
    2. Improved privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This could lead to the development
    


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
    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm passionate about [Purpose or hobby]. I'm an [Adjective] person and I believe that [Reason for passion]. My strengths are [Strengths], and my weaknesses are [Weaknesses]. What brings me to the table is [Reason for joining the group].
    The [Name] is a [occupation] who is dedicated to [purpose or hobby]. My passion for this is [reason for passion], and I believe that my unique strengths and weaknesses make me a valuable member of the group. I'm excited to bring my enthusiasm and creativity to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the third-largest city in the world, with a population of over 2 million people. The city is located on the river Seine and is home to many important landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Arc de Triomphe. The city is also a major center of arts, culture, and business, and is a popular tourist destination. Paris is a vibrant and dynamic city that has played a significant role in French history and culture for centuries. Its unique blend of old and new, historical and modern, makes it a city of contrasts and beauty.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising, with many possible trends that could shape the technology's direction. Here are some of the most significant ones:
    
    1. Increased AI privacy and security: As AI systems become more advanced, there will be an increased risk of data breaches, hacking, and other malicious activities. Therefore, there will be a growing emphasis on improving security measures to protect the privacy and security of AI systems.
    
    2. AI-driven healthcare: AI is already being used in healthcare to assist doctors and improve patient care. As AI technology continues to evolve, we can expect to see even more applications in healthcare, such as personalized medicine, disease prediction, and patient tracking.
    
    


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

    insert

     name

    ].

     I

     am

     an

     AI

     language

     model

     designed

     to

     assist

     people

     in

     generating

     writing

     prompts

    ,

     generating

     questions

     and

     answers

    ,

     answering

     questions

    ,

     and

     helping

     with

     brainstorm

    ing

    .

     I

    ’m

     a

     versatile

     tool

     with

     access

     to

     a

     vast

     array

     of

     knowledge

     and

     information

     on

     a

     wide

     range

     of

     topics

    .

     I

     can

     help

     you

     create

     writing

     prompts

    ,

     generate

     questions

    ,

     answer

     questions

    ,

     and

     assist

     with

     brainstorm

    ing

    ,

     and

     I

     can

     even

     help

     you

     with

     your

     writing

     goals

     and

     objectives

    .

     Whether

     you

     need

     help

     with

     your

     writing

     project

     or

     just

     want

     to

     have

     some

     fun

    ,

     I

    ’m

     here

     to

     assist

     you

    .

     Let

     me

     know

     how

     I

     can

     be

     of

     service

    !

     [

    insert

     name

    ]

     [

    insert

     character

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     culture

    .

     The

     city

     is

     a

     melting

     pot

     of

     various

     ethnic

     groups

     and

     is

     home

     to

     some

     of

     the

     world

    's

     most

     famous

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Paris

     is

     also

     known

     for

     its

     elaborate

     and

     romantic

     cafes

    ,

     as

     well

     as

     its

     famous

     annual

     street

     festivals

     such

     as

     the

     E

    iff

    el

     Tower

     parade

    .

     Overall

    ,

     Paris

     is

     a

     bustling

     and

     exciting

     city

     with

     a

     rich

     history

     and

     a

     unique

     blend

     of

     cultures

    .

     
    


    In

     conclusion

    ,

     Paris

     is

     the

     capital

     city

     of

     France

    ,

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

     vibrant

     culture

    .

     Its

     unique

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     as

     it

     continues

     to

     evolve

     rapidly

     and

     in

     different

     areas

    .

     Here

     are

     some

     possible

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

     Machine

     learning

     and

     deep

     learning

    :

     Machine

     learning

     and

     deep

     learning

     are

     expected

     to

     play

     an

     increasingly

     significant

     role

     in

     AI

    ,

     particularly

     in

     areas

     such

     as

     natural

     language

     processing

    ,

     computer

     vision

    ,

     and

     speech

     recognition

    .

     These

     technologies

     can

     analyze

     large

     amounts

     of

     data

     quickly

     and

     accurately

    ,

     which

     could

     lead

     to

     more

     intelligent

     and

     efficient

     AI

     systems

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     becoming

     more

     and

     more

     integrated

     into

     various

     industries

    ,

     and

     it

     is

     expected

     that

     this

     trend

     will

     continue

     in

     the

     future

    .

     This

     could

     lead

     to

     more

    



```python
llm.shutdown()
```
