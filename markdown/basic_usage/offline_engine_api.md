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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.92it/s]


    2026-05-16 07:22:09,707 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 07:22:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.47it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.02it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.36it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.36it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.36it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.36it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.36it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.36it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.36it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.36it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   7%|▋         | 4/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   7%|▋         | 4/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   7%|▋         | 4/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):   7%|▋         | 4/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.31 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.52it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.24it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.82it/s] Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.97it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.97it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.97it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.97it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.97it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.97it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.19it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.19it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.19it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.19it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.19it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.19it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:00<00:00, 45.54it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:00<00:00, 45.54it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:00<00:00, 45.54it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 45.54it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  71%|███████   | 41/58 [00:01<00:00, 46.55it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 46.55it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 46.55it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 46.55it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 46.55it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  71%|███████   | 41/58 [00:01<00:00, 46.55it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.05it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.05it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.47it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 40.93it/s]


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
    Generated text:  Tuan Dung, I am a Chinese-American.
    
    My mother tongue is Vietnamese, and I grew up in Australia, but I still speak English. I have a good knowledge of my native language. My English is very good, but I'm not a native speaker of English.
    
    I was born in Australia in 1985, and I am now living in Australia. My parents are both teachers, and I am the only child. My younger sister and I were born in Vietnam, but my parents separated when I was young. My mother is a pianist, and my father is an English teacher. I have a brother,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader, the leader of a country, the highest ranking official of the government. The American presidential election occurs every four years. The president-elect is chosen by the electoral college consisting of the Senate and the House of Representatives of the country. The president-elect and the vice president-elect are chosen in order to choose the next president in a close succession. However, there are other important political offices that are not on a fixed term as the president of the United States is. The other important political office is that of the Supreme Court justice. The president of the United States is appointed by the United States Congress with approval by the Senate.
    
    While
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is famous for its museums, monuments, and architecture. It is also famous for its 19th-century grandiose architecture and its 18th-century 18th-century architecture. Paris is also famous for its distinctive La Belle Époque clothes and its 19th-century cafes. It is home to the Académie Françoise. It is famous for its romance and romanticism.
    
    Choose your answer: Is the following statement correct based on the paragraph?
    
    "The attractions in Paris are primarily French and are famous for their romantic and romantic themes."
    
    OPTIONS: (A). no.
    ( B).
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable. It will continue to evolve and become more advanced with each passing year. That’s because AI is based on a vast amount of data that is constantly changing and updating. If we don’t have access to this data and have to rely on external sources, the AI will become less reliable. To ensure that AI will always be reliable, we need to make sure that we are collecting, storing, and using the right data.
    There are several factors that contribute to the reliability of AI. First, it’s important to have access to the right data. This means collecting data from multiple sources and ensuring that the data is accurate and up to


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] [Ability] who has been [Career] for [Number of Years] years. I'm passionate about [What I Love to Do]. I'm always looking for new challenges and opportunities to grow and learn. I'm [What I Like to Do] and I'm always ready to learn and improve. I'm [What I Like to Do] and I'm always ready to learn and improve. I'm [What I Like to Do] and I'm always ready to learn and improve. I'm [What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its rich history, including the French Revolution and the French Revolution Monument. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Paris is a vibrant and diverse city with a rich cultural scene, and it is a popular tourist destination. It is the largest city in France by population and is a major economic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is likely to play a greater role in healthcare, with more personalized and accurate diagnoses and treatments being developed.
    
    4. Greater use of AI in transportation
    


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
    Generated text:  [insert first name] and I'm [insert second name]. I'm [insert a brief bio or about you here]. I'm excited to share the adventures that I've been on and what I've learned along the way. [insert a brief summary or quote about your character here]. I hope you enjoy our conversation and I look forward to hearing about your own experiences. [insert a closing statement or quote here].
    
    Hello, my name is [insert first name] and I'm [insert second name]. I'm excited to meet you and to share the adventures that I've been on, what I've learned along the way, and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historical significance, rich cultural heritage, and iconic landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower. Paris is often referred to as the "City of Love" and is a popular tourist destination for its romantic ambiance and beautiful architecture. It is one of the most visited cities in the world and is an important cultural and political center of France. 
    
    France's capital city is Paris. It is famous for its historical significance, cultural heritage, iconic landmarks, romantic ambiance, and beautiful architecture. It is also one of the most visited cities in the world and an important cultural and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be profoundly shaped by a combination of advancements in computing power, data analytics, and machine learning, as well as broader societal and cultural changes. Here are some of the most likely future trends in AI:
    
    1. Increased collaboration between AI and other disciplines: As AI becomes more integrated into various fields, there may be a growing emphasis on interdisciplinary collaboration between AI and other areas of research and development, such as computer science, mathematics, biology, and psychology.
    
    2. Advances in natural language processing: Advances in natural language processing (NLP) are likely to revolutionize AI by enabling computers to understand and interpret human language, enabling more natural


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

    ]

     and

     I

    'm

     a

    /an

     [

    Character

     Type

    ].

     I

    'm

     [

    Age

    ]

     years

     old

     and

     [

    Sex

    ].

     I

     have

     [

    What

     makes

     you

     unique

    ?]

     and

     [

    What

     do

     you

     enjoy

     most

    /

    least

     about

     your

     job

    ?]

     I

    'm

     [

    job

     title

    /

    role

    ]

     and

     I

    'm

     here

     to

     [

    Purpose

     of

     Visit

    ].

     Let

    's

     talk

     about

     [

    why

     you

    're

     here

     and

     what

     interests

     you

    ],

     and

     if

     you

     have

     any

     questions

    ,

     don

    't

     hesitate

     to

     ask

    !

     (

    I

     hope

     you

     enjoy

     your

     visit

    ,

     and

     have

     a

     great

     day

    !)
    


    ---
    


    Feel

     free

     to

     add

     any

     additional

     information

     or

     anecdotes

     that

     might

     help

     human

    ize

     the

     character

    .

     The

     key

     is

     to

     engage

     the

     reader

    
    
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

     in

     the

     north

     of

     the

     country

    ,

     and

     is

     the

     largest

     city

     in

     Europe

     by

     population

    .

     The

     city

     is

     a

     major

     cultural

    ,

     economic

    ,

     and

     diplomatic

     center

    ,

     and

     is

     known

     for

     its

     rich

     history

    ,

     notable

     landmarks

    ,

     and

     various

     arts

     and

     cultural

     institutions

    .

     Paris

     is

     home

     to

     several

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

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Mar

    ais

     district

    ,

     which

     is

     known

     for

     its

     historic

     French

     art

     and

     architecture

    .

     Paris

     is

     also

     a

     center

     for

     science

     and

     technology

    ,

     with

     numerous

     world

    -ren

    owned

     research

     institutions

     and

     scientific

     centers

    .

     The

     city

     has

     a

     diverse

     population

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

    ,

     and

     it

    's

     hard

     to

     predict

     what

     exactly

     will

     happen

     in

     the

     next

     few

     years

    .

     However

    ,

     based

     on

     current

     trends

    ,

     it

    's

     likely

     that

     AI

     will

     continue

     to

     evolve

     in

     the

     following

     areas

    :
    


    1

    .

     Improved

     Machine

     Learning

    :

     With

     advances

     in

     computing

     power

     and

     data

     availability

    ,

     we

     can

     expect

     to

     see

     even

     more

     powerful

     and

     sophisticated

     machine

     learning

     algorithms

     in

     the

     future

    .

     These

     algorithms

     will

     be

     able

     to

     learn

     from

     vast

     amounts

     of

     data

    ,

     identify

     patterns

    ,

     and

     make

     predictions

     on

     a

     wide

     range

     of

     tasks

    .
    


    2

    .

     Autonomous

     and

     Self

    -

    Driving

     Vehicles

    :

     As

     technology

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     more

     and

     more

     autonomous

     and

     self

    -driving

     vehicles

     on

    



```python
llm.shutdown()
```
