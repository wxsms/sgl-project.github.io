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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.00it/s]


    2026-05-05 23:52:46,877 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 23:52:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.39it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.39it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.71it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.12it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.26it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.26it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.26it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.26it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.63it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.90it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.90it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.90it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.90it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.90it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.90it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.90it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.20it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.88it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.88it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.88it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.88it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.88it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.88it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.05it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.25it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.25it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.25it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.25it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.21it/s]


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
    Generated text:  Vanessa, and I'm a dedicated single parent of two young children. I want to know more about your experience and advice for working from home during the pandemic. Could you please provide some insight on the challenges and benefits of working from home for single parents like myself, and what accommodations and support systems I may need to implement in order to succeed? Additionally, could you share any tips or resources that you have found helpful in finding and maintaining work-life balance while working from home? Of course, Vanessa! Thank you for reaching out. Working from home during the pandemic can be both a blessing and a challenge for single parents like yourself. Here are
    ===============================
    Prompt: The president of the United States is
    Generated text:  a rich person. He or she lives in a large house on an island. He or she wears a special suit every morning and an umbrella every day of the year. He or she drinks coffee and eggs for breakfast. He or she goes to work by bus. In the evening, he or she plays tennis and swims. In the morning, he or she goes to work by plane. He or she lives on the island of the future and doesn't need to work. There are no cars in the city. There are no trees on the island. People like to go for a walk around the island. The president lives in the "
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the capital of the United Kingdom is London, and the capital of the United States is Washington, D.C. The smallest capital of the United States is Charleston, South Carolina. Given these facts, what is the capital of the United States? The answer is Charleston, South Carolina. Given the facts:
    
    - The smallest capital of the United States is Charleston, South Carolina.
    - The capital of France is Paris.
    - The capital of the United Kingdom is London.
    - The capital of the United States is Washington, D.C.
    
    Since Charleston is the smallest capital of the United States, it must be the capital. The capital of the
    ===============================
    Prompt: The future of AI is
    Generated text:  decidedly one of disruption. With the shift from a purely physical world to a digital one, AI is looking to transform the way we live and work. The world of AI is largely unexplored territory, making it ripe with opportunities for innovation, but also for the potential for unintended consequences.
    
    One of the most prevalent trends in AI right now is the emergence of machine learning algorithms, which have revolutionized the way we interact with technology. Machine learning algorithms are capable of learning from data, and using this data, they can make predictions and decisions without explicit human intervention. This has made it possible to create more advanced AI systems, and to automate


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter, a historic district known for its French colonial architecture and French cuisine. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter, a historic district known for its French colonial architecture and French cuisine. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration with human decision-making: As AI becomes more integrated with human decision-making, it is likely to become more effective and efficient. This could lead to more accurate predictions and better decision-making in various industries.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, it is likely to be used in more areas, including diagnosis, treatment planning, and patient care.
    
    3. Increased use of AI
    


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
    Generated text:  [Name], and I'm a [Occupation]. I'm passionate about [What inspires me in my work]. And I thrive on [What I do best in my work]. I'm a [How to get started or what your starting point is for the project]. I'm excited to dive into [Project/Role]. 
    [Name] has a strong work ethic and is always [How they approach their work]. 
    Let me know if you'd like to learn more about my background or if you're interested in getting to know [Name]. To begin, what can you tell me about yourself? [Name] is a [Occup
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Is the answer to the question "What is the capital of France?" ? Paris is the capital city of France. It's the largest city in both area and population of the country. As a French city, Paris is a UNESCO World Heritage site, famous for its museums, art galleries, historical landmarks, opera houses and street art, to name just a few attractions. Paris is also known as "la Chanson de Roland", it's French translation of the legendary French hero Roland. The city is home to the city hall, the Louvre, the Notre Dame Cathedral, the Eiffel Tower, the Champs-É
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly anticipated and is constantly evolving. Some of the possible future trends that AI is expected to undergo include:
    
    1. Increase in accuracy: AI is capable of processing complex information and making decisions with greater accuracy than humans. This will lead to better decision-making in fields like healthcare, finance, and manufacturing.
    
    2. Integration with human emotions: AI will be able to understand and respond to emotions and opinions, which will lead to more human-like interactions in virtual and augmented reality.
    
    3. Enhanced creativity: AI is expected to become more creative and innovative, as it is capable of exploring new concepts and ideas in a range of fields.
    
    4. Increased


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

    ],

     and

     I

     am

     an

     AI

     assistant

     designed

     to

     assist

     users

     in

     a

     wide

     range

     of

     tasks

    ,

     including

     language

     translation

    ,

     information

     gathering

    ,

     and

     problem

    -solving

    .

     I

     strive

     to

     be

     a

     helpful

     tool

     for

     anyone

     seeking

     help

    ,

     whether

     they

     are

     looking

     for

     advice

     on

     something

     specific

     or

     simply

     looking

     for

     an

     AI

     to

     assist

     them

     in

     their

     day

    -to

    -day

     tasks

    .

     I

     am

     here

     to

     provide

     assistance

     and

     answer

     any

     questions

     that

     a

     user

     may

     have

    ,

     whether

     it

    's

     in

     the

     form

     of

     a

     text

     or

     an

     email

    .

     I

     am

     always

     here

     to

     help

    ,

     

    2

    4

    /

    7

    .

     If

     you

     have

     any

     questions

     or

     need

     any

     assistance

    ,

     please

     don

    't

     hesitate

     to

     reach

     out

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     largest

     city

     and

     the

     capital

     of

     France

    .

     It

     is

     located

     in

     the

     western

     region

     of

     France

    ,

     on

     the

     Se

    ine

     River

    .

     It

     is

     the

     third

    -largest

     metropolitan

     area

     in

     the

     world

     and

     is

     home

     to

     many

     international

     organizations

     and

     international

     corporations

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

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

     Palace

     of

     Vers

    ailles

    .

     It

     is

     also

     famous

     for

     its

     rich

     history

     and

     cultural

     heritage

    .

     The

     French

     capital

     is

     a

     vibrant

     and

     diverse

     city

     with

     a

     population

     of

     over

     

    1

     million

     people

    .

     The

     city

     is

     known

     for

     its

     fashion

    ,

     food

    ,

     and

     wine

    ,

     and

     is

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     unpredictable

    ,

     but

     there

     are

     several

     potential

     trends

     that

     could

     lead

     to

     significant

     changes

     in

     the

     way

     we

     interact

     with

     technology

     and

     society

     as

     a

     whole

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

     autonomy

     and

     decision

    -making

    :

     AI

     systems

     are

     becoming

     more

     capable

     of

     making

     decisions

     and

     performing

     tasks

     on

     their

     own

    ,

     which

     could

     lead

     to

     more

     autonomous

     and

     self

    -directed

     AI

    .

     This

     would

     enable

     AI

     to

     make

     better

    -in

    formed

     decisions

     and

     make

     more

     complex

     decisions

     in

     real

    -time

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     becomes

     more

     prevalent

     in

     our

     daily

     lives

    ,

     we

     are

     likely

     to

     see

     increased

     emphasis

     on

     privacy

     and

     security

    .

     AI

     systems

     will

     need

     to

     be

    



```python
llm.shutdown()
```
