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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.30it/s]


    2026-05-04 23:31:45,650 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 23:31:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=192):  52%|█████▏    | 30/58 [00:04<00:01, 15.98it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.19it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.41it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.41it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.41it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.41it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.41it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.41it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.41it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.35 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.35 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.35 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.44it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.35 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.34 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.34 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.44it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=69.34 GB):  21%|██        | 12/58 [00:00<00:01, 29.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.34 GB):  21%|██        | 12/58 [00:00<00:01, 29.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.34 GB):  21%|██        | 12/58 [00:00<00:01, 29.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.33 GB):  21%|██        | 12/58 [00:00<00:01, 29.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.33 GB):  21%|██        | 12/58 [00:00<00:01, 29.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.33 GB):  21%|██        | 12/58 [00:00<00:01, 29.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.30 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=960 avail_mem=69.31 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.85it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=69.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=896 avail_mem=69.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=832 avail_mem=69.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=768 avail_mem=69.30 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=704 avail_mem=68.88 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=640 avail_mem=68.87 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=576 avail_mem=68.87 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=576 avail_mem=68.87 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=512 avail_mem=68.86 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=480 avail_mem=68.87 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=448 avail_mem=68.87 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=416 avail_mem=68.86 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=384 avail_mem=68.86 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.87it/s]

    Capturing num tokens (num_tokens=384 avail_mem=68.86 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=352 avail_mem=68.86 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=320 avail_mem=68.85 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=288 avail_mem=68.85 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=256 avail_mem=68.85 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=240 avail_mem=68.84 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=224 avail_mem=68.84 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=224 avail_mem=68.84 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.82it/s]Capturing num tokens (num_tokens=208 avail_mem=68.84 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.82it/s]Capturing num tokens (num_tokens=192 avail_mem=68.83 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=176 avail_mem=68.83 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=160 avail_mem=68.83 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=144 avail_mem=68.83 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.82it/s]

    Capturing num tokens (num_tokens=128 avail_mem=68.82 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=128 avail_mem=68.82 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=112 avail_mem=68.82 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=96 avail_mem=68.82 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.92it/s] Capturing num tokens (num_tokens=80 avail_mem=68.82 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=64 avail_mem=68.81 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=48 avail_mem=68.81 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=48 avail_mem=68.81 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=32 avail_mem=68.80 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=28 avail_mem=68.80 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=24 avail_mem=68.80 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=20 avail_mem=68.79 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.05it/s]

    Capturing num tokens (num_tokens=16 avail_mem=68.79 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=16 avail_mem=68.79 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.59it/s]Capturing num tokens (num_tokens=12 avail_mem=68.79 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.59it/s]Capturing num tokens (num_tokens=8 avail_mem=68.79 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.59it/s] Capturing num tokens (num_tokens=4 avail_mem=68.78 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.59it/s]Capturing num tokens (num_tokens=4 avail_mem=68.78 GB): 100%|██████████| 58/58 [00:01<00:00, 42.30it/s]


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
    Generated text:  Trisha and I am a mathematician at the University of Michigan. I am interested in areas of algebra, number theory, and arithmetic geometry. My interests include cryptography, specifically I am interested in elliptic curves and the related problem of the "man-in-the-middle attack" on elliptic curve cryptography.
    I am also interested in communication theory. This is the study of the transmission of information, whether in the physical world or over the internet, using the many symbols that represent words, numbers, and letters. The key issue in communication is to ensure that the messages do not contain any errors. The study of error-correcting codes and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office. Elections are held every four years in the United States. The president and vice president are elected by the people to be in charge of the country. Elections are held in January. Candidates for the job are usually chosen by the people. The president has a lot of powers. He can make laws, stop other people, and even refuse to follow orders. The vice president can only do the things the president does. The president must make peace between a country and its neighbors. The president is supposed to be a good leader. If he does his job well, he will get a big pay. If he does not do his
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The region around Paris, which we now call Île de France, contains many attractions. The Canal Sainte-Genevieve, the Canal Saint-Jean and the Seine River are three of them. For a young man named Louis, he decided to visit the Seine River and the Canal Saint-Jean. Louis has two options: he can either take a boat ride on the Seine River, which would take him 1 hour, or he can take a cable car ride on the Canal Saint-Jean, which would take him 3 hours. He wants to choose the option that takes him the shortest time to
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, but it's still in its infancy. While the current AI-driven innovations have brought immense benefits to society, it is still important to continue to monitor its progress and ensure that it remains ethical and sustainable. The following are some potential areas for improvement in the field of AI, such as:
    
    1. Ethical AI: The development and deployment of AI systems should be guided by ethical principles that prioritize the well-being of individuals and society as a whole. This includes ensuring that AI systems are designed and deployed with minimal bias and that they are transparent and explainable.
    
    2. Transparency: AI systems should be transparent by default, providing clear


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] and [Country]. I'm a [Skill] with [Number] years of experience in [Industry/Field]. I'm a [Number] year old, [Gender] and [Country]. I'm a [Skill] with [Number] years of experience in [Industry/Field]. I'm a [Skill] with [Number] years of experience in [Industry/Field]. I'm a [Skill] with [Number] years of experience in [Industry/Field]. I'm a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" (The City of the Sea). It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also home to many famous museums, including the Musée d'Orsay, the Musée Rodin, and the Musée d'Orsay. Paris is a popular tourist destination and a major economic center in France. It is the seat of the French government
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we are likely to see more widespread adoption of AI-powered technologies. This could include things like smart home devices, self-driving cars, and virtual assistants that can assist with tasks like grocery shopping or scheduling appointments.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ensuring
    


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
    Generated text:  [Your Name]. I'm a [Your Profession] with a passion for [Your Hobby/Interest]. Let me know if you'd like to discuss my background or accomplishments. And to wrap it up, feel free to introduce yourself and tell me a little bit about yourself. [Your Name]. Hello, my name is [Your Name] and I'm a [Your Profession] with a passion for [Your Hobby/Interest]. Let me know if you'd like to discuss my background or accomplishments. And to wrap it up, feel free to introduce yourself and tell me a little bit about yourself. [Your Name]. [Your Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known as the “City of Light” and the “City of Art.” It is a major cultural center, home to the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and many other significant landmarks. France’s capital is also a major economic and political center, with Paris serving as the seat of government and the country’s largest city. Visitors to Paris can enjoy its rich cultural and historical heritage, and be immersed in the vibrant urban life of the city. With its stunning architecture, vibrant nightlife, and delicious food, Paris is a must-visit destination for anyone seeking to experience the beauty
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several changes and developments, depending on the specific industries and applications that are being used. Some possible future trends in artificial intelligence include:
    
    1. Increased automation: AI is already becoming more and more automated, with new algorithms and machine learning techniques being developed to perform tasks that were previously done by humans. In the future, this automation could extend to all areas of human work, from manufacturing to customer service.
    
    2. Improved ethics and transparency: As AI systems become more sophisticated, there is a risk that they may become more opaque and difficult to understand. To mitigate this risk, there is an increasing emphasis on developing ethical guidelines and transparency


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

    ],

     and

     I

    'm

     a

     [

    insert

     age

    ,

     gender

    ,

     occupation

    ,

     etc

    .

    ].

     I

    'm

     a

     [

    insert

     occupation

    ],

     but

     I

    've

     always

     enjoyed

     [

    insert

     why

     I

     enjoy

     my

     occupation

    ].

     As

     a

     [

    insert

     occupation

    ],

     I

    've

     always

     been

     passionate

     about

     [

    insert

     passion

     you

     have

     for

    ],

     and

     I

     love

     to

     [

    insert

     any

     other

     qualities

     you

     have

    ,

     such

     as

     kindness

    ,

     reliability

    ,

     or

     humor

    ].

     I

    've

     always

     been

     [

    insert

     any

     hobbies

     you

     have

    ,

     such

     as

     reading

    ,

     playing

     sports

    ,

     or

     singing

    ].

     My

     goal

     is

     to

     [

    insert

     goal

     you

     have

    ,

     such

     as

     becoming

     a

     [

    insert

     title

    ],

     traveling

    ,

     or

     learning

     something

     new

    ].

     I

    'm

     always

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     world

    -f

    amous

     city

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     home

     to

     the

     Lou

    vre

     Museum

    ,

     where

     the

     Mona

     Lisa

     is

     housed

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     history

    ,

     with

     its

     rich

     historical

     significance

     dating

     back

     over

     

    3

    ,

    0

    0

    0

     years

    .

     Paris

     is

     a

     bustling

     and

     diverse

     city

     with

     a

     diverse

     range

     of

     cultures

     and

     languages

     spoken

    .

     It

     is

     an

     important

     cultural

    ,

     political

    ,

     and

     economic

     center

     of

     France

     and

     the

     world

    .

     French

     cuisine

     is

     also

     renowned

     for

     its

     delicious

     bread

    ,

     cheese

    ,

     and

     wines

    .

     Paris

     is

     also

     famous

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     expand

     and

     evolve

     at

     a

     rapid

     pace

    .

     Here

     are

     some

     possible

     trends

     we

     can

     expect

     to

     see

     in

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

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     assist

     doctors

     in

     diagn

    osing

     diseases

     and

     developing

     treatment

     plans

    .

     In

     the

     future

    ,

     we

     can

     expect

     AI

     to

     become

     more

     sophisticated

     and

     personalized

    ,

     allowing

     doctors

     to

     provide

     better

     care

     to

     their

     patients

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     is

     already

     being

     used

     to

     predict

     financial

     trends

    ,

     detect

     fraud

    ,

     and

     optimize

     investment

     portfolios

    .

     In

     the

     future

    ,

     we

     can

     expect

     AI

     to

     become

     even

     more

     sophisticated

    ,

     allowing

     financial

     institutions

     to

     process

     more

     transactions

     and

    



```python
llm.shutdown()
```
