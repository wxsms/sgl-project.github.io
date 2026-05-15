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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.20it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.19it/s]


    2026-05-15 22:48:50,922 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 22:48:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.45it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.07it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.07it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.07it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.07it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.07it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.07it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.07it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.82 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.81 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.81 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.81 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.81 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.80 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.79 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.79 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.79 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.77 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.77 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.76 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.73 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]Capturing num tokens (num_tokens=960 avail_mem=71.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s] Capturing num tokens (num_tokens=896 avail_mem=71.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.74 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]Capturing num tokens (num_tokens=832 avail_mem=71.74 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.80it/s]Capturing num tokens (num_tokens=768 avail_mem=71.74 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.80it/s]Capturing num tokens (num_tokens=704 avail_mem=71.74 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.80it/s]Capturing num tokens (num_tokens=640 avail_mem=71.73 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.80it/s]Capturing num tokens (num_tokens=576 avail_mem=71.73 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.80it/s]Capturing num tokens (num_tokens=512 avail_mem=71.72 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.80it/s]Capturing num tokens (num_tokens=512 avail_mem=71.72 GB):  50%|█████     | 29/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=480 avail_mem=71.73 GB):  50%|█████     | 29/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=448 avail_mem=71.73 GB):  50%|█████     | 29/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=416 avail_mem=71.73 GB):  50%|█████     | 29/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=384 avail_mem=71.73 GB):  50%|█████     | 29/58 [00:00<00:00, 42.90it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.72 GB):  50%|█████     | 29/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=352 avail_mem=71.72 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=320 avail_mem=71.72 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=288 avail_mem=71.71 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=256 avail_mem=71.71 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=240 avail_mem=71.71 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=224 avail_mem=71.70 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=224 avail_mem=71.70 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=208 avail_mem=71.70 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=192 avail_mem=71.70 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=176 avail_mem=71.70 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=160 avail_mem=71.69 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.65it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.69 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=144 avail_mem=71.69 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=128 avail_mem=71.69 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=112 avail_mem=71.69 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=96 avail_mem=71.68 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s] Capturing num tokens (num_tokens=80 avail_mem=71.68 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=64 avail_mem=71.67 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=64 avail_mem=71.67 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=48 avail_mem=71.67 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=32 avail_mem=71.67 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=28 avail_mem=71.66 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=24 avail_mem=71.66 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.45it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.66 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=20 avail_mem=71.66 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=16 avail_mem=71.66 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=12 avail_mem=71.65 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=8 avail_mem=71.65 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.69it/s] Capturing num tokens (num_tokens=4 avail_mem=71.65 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=4 avail_mem=71.65 GB): 100%|██████████| 58/58 [00:01<00:00, 41.46it/s]


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
    Generated text:  Luis A. M. Roldán. I am a professor of mathematics, specializing in mathematics education and mathematical methods in science education in the public school system, and have been teaching mathematics at the university level since 1984.\nI have received teaching certification from the Universidad de Castellón and from the Universidad de Guayaquil. My academic training is in mathematics and I have received a teaching qualification certificate from the Universidad de Castellón. I have also received the Spanish Department of the Universidad de Guayaquil teaching qualification certificate.\nI am a member of the National Council of Professional Development for Teachers of Mathematics, the
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to appoint a woman to a cabinet position. The president has 872 different ways to choose a president, and there are two men and two women on the current team. If the president wants to hire someone with the highest chance of success in their first year, which of the following would be the most efficient way to find the most qualified candidate?
    
    A) Hire a man
    B) Hire a woman
    C) Hire a woman and a man
    D) Hire a woman and a woman
    
    To determine which option is the most efficient way to find the most qualified candidate, we need to consider the number of
    ===============================
    Prompt: The capital of France is
    Generated text:  ( )
    A: Paris
    B: Bordeaux
    C: Marseille
    D: Nice
    1. **Identify the capital of France:**
       - The capital of France is typically Paris, which is the largest city in the country and is home to many of its historical and cultural landmarks.
    
    2. **List the given options:**
       - A: Paris
       - B: Bordeaux
       - C: Marseille
       - D: Nice
    
    3. **Compare the capital cities:**
       - Paris is the capital of France.
       - Bordeaux is the capital of France.
       - Marseille is the capital of France.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  in the making, and it’s happening at a faster pace than ever before. This exponential growth in AI technology is causing a shift in how we perceive and interact with technology. In the last decade, AI has become a dominant force in many industries, from healthcare to finance to retail. But with this rapid growth comes the need to adapt to the rapidly changing landscape. As technology continues to advance, so too will the skills and knowledge that individuals need to succeed in this field.
    One of the key skills that will be in demand in the AI field is natural language processing (NLP). NLP is a critical area of AI that deals with


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] [Vehicle Name]. I have been driving for [Number of Years] years and have [Number of Miles] miles driven. I have a passion for [Favorite Activity/Interest/Job/Other]. I am always looking for new experiences and adventures, and I am always eager to learn and grow. I am a [Type of Person] [Personality Type]. I am [Age] years old and I am [Occupation]. I am a [Type of Vehicle] [Vehicle Name]. I have been driving for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and diverse culture. It is located in the south of France and is the largest city in the country. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also known for its fashion industry, art, and cuisine. Paris is a popular tourist destination and is home to many world-renowned museums, theaters, and restaurants. The city is also home to the French Parliament and the French National Library. Paris is a cultural and historical center that has
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends in AI that are expected to shape the future:
    
    1. Increased automation: AI is already being used in a wide range of industries, from manufacturing to healthcare to customer service. As AI becomes more advanced, we can expect to see even more automation in the future, with machines taking on tasks that were previously done by humans.
    
    2. Enhanced human-computer interaction: AI is already being used to enhance human-computer interaction, such as through voice recognition and natural language processing
    


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
    Generated text:  [Your Name], and I'm a [type of character, like a professional, an entertainer, or a regular person]. I'm a [major characteristic of your character], and I love [the reason why you love it]. I strive to be [a positive trait, like kind, helpful, or respectful]. What do you do, [any other information you'd like to include]?
    Your introduction should be brief yet engaging, allowing the reader to instantly connect with you. Try to include a sense of curiosity or interest in what you do, and make sure your name and major characteristic are clear. When writing your introduction, think
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the French Quarter. It is also the home to the French Parliament, the headquarters of the French government, and is considered one of the most important cities in the world. Paris is a popular tourist destination with a rich cultural and historical heritage. It is known for its cuisine, fashion, and dance, and is home to many renowned artists, including Picasso and Rembrandt. The city is also known for its vibrant nightlife and fashion scene. Paris is a city that has a rich and diverse history, and is recognized
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be heavily influenced by several key trends, including:
    
    1. **Increase in AI in Healthcare**: AI is increasingly being used in healthcare to improve diagnosis, treatment, and patient care. This includes using AI to analyze medical images, identify patterns in medical records, and assist doctors in making more accurate diagnoses. AI-powered diagnostic tools like AI-powered imaging analysis systems and AI-powered decision support systems are already being used in various healthcare settings.
    
    2. **AI in Agriculture**: AI is being used to improve crop yields, reduce costs, and increase efficiency in the agricultural sector. AI can be used to analyze weather patterns, soil conditions, and genetic


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

     and

     I

     am

     an

     experienced

     and

     versatile

     professional

     with

     a

     passion

     for

     innovation

     and

     problem

    -solving

    .

     I

     have

     a

     diverse

     range

     of

     skills

     and

     knowledge

     across

     various

     industries

    ,

     including

     [

    list

     any

     relevant

     skills

     or

     areas

     of

     expertise

    ].

     I

     thrive

     in

     fast

    -paced

     environments

     and

     thrive

     on

     collaboration

     and

     teamwork

    .

     I

     am

     a

     proactive

     problem

     solver

     who

     can

     quickly

     analyze

     and

     resolve

     complex

     issues

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     and

     grow

    ,

     and

     I

     am

     always

     looking

     for

     ways

     to

     contribute

     to

     the

     success

     of

     others

    .

     Thank

     you

     for

     considering

     me

     for

     an

     interview

    ,

     and

     I

     look

     forward

     to

     discussing

     my

     skills

     and

     experience

     further

    .
    


    [

    Your

     Name

    ]

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     second

    -largest

     city

     in

     the

     European

     Union

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

    .

     

    4

     million

     as

     of

     

    2

    0

    2

    0

    .

     It

     is

     the

     capital

     city

     of

     the

     country

    's

     largest

     metropolitan

     area

    ,

     the

     Î

    le

    -de

    -F

    rance

     metropolitan

     area

    ,

     and

     is

     the

     seat

     of

     government

     for

     the

     country

    's

     largest

     metropolitan

     region

    ,

     the

     Paris

     Region

    .

     It

     is

     also

     the

     official

     capital

     of

     France

    ,

     the

     residence

     of

     the

     President

     of

     the

     Republic

     and

     the

     office

     of

     the

     State

     Secretary

     of

     State

    ,

     and

     the

     main

     city

     of

     a

     significant

     number

     of

     other

     countries

     and

     regions

     in

     Europe

     and

     beyond

    .

     It

     is

     the

     largest

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     evolving

     rapidly

    ,

     with

     new

     technologies

     and

     applications

     emerging

     all

     the

     time

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

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

     develop

    ,

     we

     can

     expect

     to

     see

     more

     automated

     systems

     and

     applications

     being

     developed

     and

     deployed

    .

     This

     could

     lead

     to

     significant

     cost

     savings

     and

     increased

     efficiency

     for

     businesses

     and

     individuals

    .
    


    2

    .

     Greater

     personal

    ization

    :

     AI

     is

     already

     being

     used

     to

     personalize

     customer

     experiences

     and

     product

     recommendations

    .

     In

     the

     future

    ,

     we

     can

     expect

     to

     see

     even

     more

     advanced

     personal

    ization

     capabilities

    ,

     with

     AI

     being

     able

     to

     learn

     and

     adapt

     to

     individual

     user

     preferences

     and

     behavior

    .
    


    3

    .

     Enhanced

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

    



```python
llm.shutdown()
```
