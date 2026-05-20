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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]


    2026-05-20 19:52:29,073 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 19:52:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.94it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.98it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.17it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.62it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.62it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.62it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.62it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.62it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.62it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.62it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.91 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.91 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.90 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.89 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.88 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.88 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.88 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.85 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.85 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.85 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.85 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.84 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.83 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]Capturing num tokens (num_tokens=960 avail_mem=72.84 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s] Capturing num tokens (num_tokens=896 avail_mem=72.84 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.83 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]Capturing num tokens (num_tokens=832 avail_mem=72.83 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=768 avail_mem=72.83 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=704 avail_mem=72.83 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=640 avail_mem=72.82 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=576 avail_mem=72.82 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=512 avail_mem=72.81 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=512 avail_mem=72.81 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=480 avail_mem=72.82 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=448 avail_mem=72.82 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=416 avail_mem=72.82 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=384 avail_mem=72.82 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.81 GB):  50%|█████     | 29/58 [00:00<00:00, 43.86it/s]Capturing num tokens (num_tokens=352 avail_mem=72.81 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=320 avail_mem=72.81 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=288 avail_mem=72.80 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=256 avail_mem=72.80 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=240 avail_mem=72.80 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=224 avail_mem=72.79 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=224 avail_mem=72.79 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.66it/s]Capturing num tokens (num_tokens=208 avail_mem=72.79 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.66it/s]Capturing num tokens (num_tokens=192 avail_mem=72.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=176 avail_mem=72.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=160 avail_mem=72.78 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.66it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.78 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=144 avail_mem=72.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=128 avail_mem=72.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=112 avail_mem=72.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=96 avail_mem=72.77 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.48it/s] Capturing num tokens (num_tokens=80 avail_mem=72.77 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=64 avail_mem=72.76 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=64 avail_mem=72.76 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.52it/s]Capturing num tokens (num_tokens=48 avail_mem=72.76 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.52it/s]Capturing num tokens (num_tokens=32 avail_mem=72.76 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.52it/s]Capturing num tokens (num_tokens=28 avail_mem=72.75 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.52it/s]Capturing num tokens (num_tokens=24 avail_mem=72.75 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.52it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.75 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.52it/s]Capturing num tokens (num_tokens=20 avail_mem=72.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=16 avail_mem=72.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=12 avail_mem=72.74 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=8 avail_mem=72.74 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.89it/s] Capturing num tokens (num_tokens=4 avail_mem=72.74 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.89it/s]Capturing num tokens (num_tokens=4 avail_mem=72.74 GB): 100%|██████████| 58/58 [00:01<00:00, 42.29it/s]


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
    Generated text:  Elena and I have been working with a new C++ library called Cudnn. It's very cool! It's a great way to work with C++ and it has lots of interfaces and libraries to choose from. Can you tell me more about Cudnn and what it can do?
    
    Cudnn is a C++ library that provides a low-level interface to the cuDNN library, which is a C++ wrapper for the NVIDIA cuDNN library. It allows developers to use the cuDNN library to perform computations on GPUs in a C++ environment. Cudnn supports a wide range of operations such as matrix operations
    ===============================
    Prompt: The president of the United States is
    Generated text:  now considering a new policy proposal. There are 10 candidates running for this position, each with a different probability of winning. The probability that the president will win a specific race is 0.6. Calculate the probability that the president will win the election if:
    
    a) All 10 candidates win
    b) Exactly 5 of the candidates win
    c) At least 4 of the candidates win
    d) No candidate wins
    
    To solve this problem, we need to use the binomial probability formula. The binomial probability formula is given by:
    
    \[ P(X = k) = \binom{n}{k}
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it has a population of 2,524,552. The economy of Paris is primarily based on tourism, but it has a strong cultural and artistic sector. The city is also home to many important museums and cultural institutions, such as the Louvre and the Musée d'Orsay. The city is also known for its food culture, with many popular cuisines, such as croissants, amuse-bouche, and pastries.
    Based on the above information, is Paris a city with a strong tourism industry? Answer with "yes" or "no".
    Yes. Paris is a
    ===============================
    Prompt: The future of AI is
    Generated text:  fully dependent on human beings, and it is the human brain that is the source of all innovation and progress. The core of AI is the human brain, and the core of AI research is to improve the human brain. In the age of machine learning, a new generation of artificial intelligence is emerging, with the technological potential to address many of humanity’s most pressing problems. One of the most important areas of AI research, therefore, is understanding the human brain.
    Brain-computer interfaces (BCIs) aim to harness the natural human brain to perform tasks such as writing, speech, and even thinking. In this paper, we develop and demonstrate


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has [Number of Years] years of experience in [Field]. I'm passionate about [What I Love About My Profession]. I'm always looking for ways to [What I Want to Improve]. I'm [What I Do Best]. I'm [What I Can Do Best]. I'm [What I Can Do Best]. I'm [What I Can Do Best]. I'm [What I Can Do Best]. I'm [What I Can Do Best]. I'm [What I Can Do Best]. I'm [What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and art, and is a popular tourist destination for its beautiful architecture and historical sites. Paris is a city of contrasts, with its modern skyscrapers and historic neighborhoods, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from and adapt to new information and situations.
    
    2. Enhanced machine learning capabilities: AI is likely to become more capable of learning from large amounts of data and making more accurate predictions and decisions. This could lead to more advanced and sophisticated machine learning algorithms that can handle complex and intricate problems.
    
    3. Increased focus on ethical considerations: As AI becomes more integrated with human
    


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
    Generated text:  [Your Name] and I am a [Your profession]. I am passionate about [Your career-related topic or goal], and I am always looking for ways to improve my skills and knowledge in this field. I am a highly organized and detail-oriented person, with a keen interest in connecting with people and learning from their experiences. I enjoy traveling, reading, and pursuing new hobbies. I am excited to share my experiences and knowledge with others and to be a valuable asset to anyone who trusts me. How would you like to meet you? I would love to have a conversation with you and find out more about you. Good luck! [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which has a population of over 7 million people.
    
    Sure, here is a concise factual statement about Paris:
    
    The capital of France is Paris, with a population of over 7 million people. 
    
    Please let me know if you need any further assistance! 
    
    Additionally, I'll be happy to provide more information about Paris if you're interested in learning more about its culture, history, or other aspects of the city. Let me know how I can assist you! 
    
    Please let me know if you need any further assistance! 
    
    I'll be happy to provide more information about Paris if you're interested in learning more about its culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field, with various trends and advancements shaping how it will continue to develop. Some possible trends that are likely to shape the future of AI include:
    
    1. Increased focus on ethical and responsible AI: As more and more AI is being deployed in various domains, it is becoming increasingly important to ensure that AI systems are ethical and responsible. This will require a greater focus on ethical considerations and regulatory compliance.
    
    2. Growing importance of AI in healthcare: With the increasing demand for personalized medicine, AI is expected to play a more significant role in healthcare. AI will be used to analyze patient data, provide personalized treatment recommendations, and improve diagnosis


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

    /an

     [

    Age

    ],

     [

    Character

     Name

    ]

     [

    Job

     Title

    ].

     I

     was

     born

     in

     [

    Birth

    place

    ]

     and

     grew

     up

     in

     [

    Place

     of

     Birth

    ].

     I

     have

     always

     had

     an

     [

    In

    stitution

    ]

     education

     and

     am

     currently

     a

     [

    degree

    ]

     graduate

     with

     a

     [

    Major

     Degree

    ]

     degree

    .

     I

     have

     a

     passion

     for

     [

    Aff

    ection

    /

    Interest

    /

    Challenge

    ],

     and

     I

    'm

     currently

     [

    Status

    /

    Goal

    /

    Current

    ]

     in

     [

    Occup

    ation

    ].

     I

     love

     [

    Sports

    /H

    obby

    /

    Activity

    /

    Interest

    ]

     and

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     try

     [

    New

     Skill

    /

    Experience

    /

    Activity

    ].

     I

    'm

     a

     talented

     [

    Skill

    /

    Ability

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

     area

    ,

     on

     the

     Se

    ine

     River

     in

     the

     south

    -west

    ern

     part

     of

     the

     country

    .


    Paris

     is

     the

     oldest

     and

     most

     populous

     city

     in

     France

    ,

     and

     one

     of

     the

     world

    's

     most

     important

     cities

    ,

     with

     a

     population

     of

     around

     

    2

    .

     

    3

     million

     people

    .

     The

     city

     has

     a

     rich

     cultural

     and

     historical

     heritage

    ,

     and

     is

     home

     to

     many

     world

    -ren

    owned

     institutions

     and

     landmarks

    .

     It

     is

     also

     a

     major

     economic

     hub

     and

     a

     major

     cultural

     center

    ,

     with

     its

     many

     museums

    ,

     theaters

    ,

     and

     landmarks

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     unique

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     it

     is

     likely

     to

     continue

     to

     transform

     various

     aspects

     of

     society

     in

     many

     different

     ways

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     automation

     and

     artificial

     intelligence

     in

     manufacturing

    :

     AI

     and

     automation

     will

     continue

     to

     increase

     in

     their

     integration

     into

     manufacturing

     industries

    ,

     with

     the

     goal

     of

     reducing

     costs

     and

     increasing

     efficiency

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

     sophisticated

    ,

     it

     is

     likely

     to

     require

     more

     advanced

     security

     measures

     to

     protect

     data

     and

     ensure

     that

     it

     is

     handled

     eth

    ically

    .
    


    3

    .

     AI

    -driven

     medical

     diagnoses

    :

     AI

     has

     already

     made

     significant

     advances

     in

     medical

     diagnosis

    ,

     with

     tools

     like

     machine

     learning

     being

     used

     to

     analyze

     medical

     images

     and

     identify

     patterns

    



```python
llm.shutdown()
```
