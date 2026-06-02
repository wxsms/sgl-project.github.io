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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.18it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 19.22it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 37.21it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 37.21it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 37.21it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 37.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 15.51it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:03, 17.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:03, 17.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:03, 17.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:03, 17.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.32it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.92it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.45it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.45it/s] Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.16it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.16it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.16it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.16it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.16it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.43it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.43it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.43it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.43it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:01<00:01, 26.76it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:01<00:01, 26.76it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:01<00:01, 26.76it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:01<00:01, 26.76it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.97it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.97it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.97it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.97it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 26.52it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.57it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.57it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.57it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.57it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.57it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.16it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.16it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.16it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.16it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.10it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.10it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.10it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.10it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 24.77it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 24.77it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:02<00:00, 24.77it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:02<00:00, 24.77it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.97it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.97it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.97it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.97it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.39it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.39it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.39it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.39it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:02<00:00, 24.58it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:02<00:00, 24.58it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:02<00:00, 24.18it/s]


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
    Generated text:  Lisa and I want to volunteer at a local library. What should I do?
    Volunteering at a local library can be a great way to give back to your community and help connect with people in your area. Here are some steps to help you get started:
    
      1. Start by researching local libraries in your area to see what kind of volunteering opportunities are available and what kind of skills you may be able to bring to the table.
      2. Decide which type of volunteer you would like to do. This could be anything from assisting with research, shelving books, answering questions, or helping with special events.
      
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking, government official who serves as the leader of the executive branch of the federal government. In many countries, the president is a member of the ruling party. In the United States, the president is elected through a process called the presidential election, which is a general election that is held every four years. The president is the leader of the executive branch and is responsible for guiding and managing the federal government. The president is also responsible for representing the interests of the United States in international forums. The president has the power to issue executive orders, dissolve Congress, declare war, and declare a state of emergency. The president's term of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, one of the most famous historical and cultural cities in the world. But there is another city worth visiting in the heart of Paris: Luxembourg. After two centuries of colonization, it became the 7th largest city in the world. It is well worth the visit if you can spare the time to visit all the landmarks. The main sights include the Eiffel Tower, the Champ de Mars, the Louvre and the Luxembourg Gardens. The capital is also a good place to take a day trip to nearby Luxembourg City. All the beauty is just out of the way, so you can visit all the places without worrying about traffic and
    ===============================
    Prompt: The future of AI is
    Generated text:  constantly evolving and growing. While AI is still considered to be a relatively new field, it has already become an integral part of many aspects of our lives. AI is responsible for driving many of the technological advancements that we see today, from the devices we use every day to the automation of industries. The potential of AI is enormous, and it has the potential to drive significant changes in many industries, including healthcare, transportation, and finance. However, there are also significant challenges and limitations that need to be addressed in order to fully harness the potential of AI.
    One of the biggest challenges with AI is the question of accountability. AI systems are often


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance in French culture and politics. However, it could be expanded to include additional information, such as the city's historical significance, notable landmarks, or cultural attractions. For example:
    
    "Paris, the capital of France, is known as the "City of Light" and is home to many iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also home to a rich cultural heritage, with a diverse range of museums, galleries, and theaters, including the Mus
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is likely to become more prevalent in many industries, with automation becoming more prevalent in areas such as manufacturing, transportation, and customer service.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be an increased focus on privacy and security. This will lead to more stringent regulations and standards for AI development and use.
    
    3. AI-driven healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in more complex and personalized ways, potentially leading to
    


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
    Generated text:  [Your Name], and I'm a [occupation] who enjoys spending time outdoors. My favorite outdoor activity is hiking and I've been hiking for [number] years now. I've been hiking to [reason for choosing the location]. What's your favorite activity and why do you enjoy doing it? [Your Name]...
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, located in the Île de France on the banks of the River Seine, which is the longest river in Europe. It is the 15th-largest city in the world and the seventh-most populous city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is one of the most important cities in the world and is the largest European city by population, with more than half of France's population living there. The city is home to the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral, among others
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to see increasing automation and integration with the physical world, leading to a more human-like interaction with machines. In the long-term, AI is expected to have a profound impact on society, from improving healthcare and education to making transportation more efficient and providing new forms of entertainment. However, it's important to note that the development of AI is a complex process with many potential risks and challenges, including privacy concerns, job displacement, and ethical considerations. As such, it is important for society to approach AI with caution and to prioritize its development and use responsibly. It is up to the individual and society as a whole to navigate the complex and often


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

    ]

     and

     I

     am

     a

     professional

     software

     developer

     with

     over

     

    1

    0

     years

     of

     experience

     in

     creating

     and

     maintaining

     software

     applications

    .

     My

     background

     includes

     a

     degree

     in

     Computer

     Science

     and

     my

     hands

    -on

     experience

     has

     taught

     me

     how

     to

     design

    ,

     develop

    ,

     and

     debug

     complex

     software

     systems

    .

     I

     have

     a

     deep

     understanding

     of

     programming

     languages

     such

     as

     Java

    ,

     C

    ++,

     and

     Python

     and

     am

     proficient

     in

     using

     version

     control

     systems

     such

     as

     Git

    .

     I

     am

     also

     experienced

     in

     testing

    ,

     debugging

    ,

     and

     troubleshooting

     software

     applications

    ,

     and

     I

     have

     worked

     on

     a

     wide

     range

     of

     projects

    ,

     including

     web

    -based

     applications

    ,

     mobile

     applications

    ,

     and

     enterprise

     systems

    .

     I

     am

     a

     self

    -m

    ot

    ivated

     individual

     who

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     Haut

    s

    -de

    -F

    rance

     region

     in

     northern

     France

    .

     The

     city

     is

     the

     largest

     city

     and

     the

     most

     populous

     administrative

     and

     political

     capital

     of

     France

    .

     It

     is

     known

     for

     its

     classical

     architecture

    ,

     world

    -ren

    owned

     museums

    ,

     and

     historic

     sites

    .

     Paris

     is

     known

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     lively

     nightlife

    ,

     and

     is

     home

     to

     the

     Lou

    vre

     Museum

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     unique

     food

     culture

    ,

     including

     its

     famous

     cro

    iss

    ants

     and

     past

    ries

    .

     The

     city

     is

     also

     a

     major

     transportation

     hub

    ,

     known

     for

     its

     efficient

     public

     transportation

     system

    .

     
    


    Paris

     is

     a

     major

     cultural

     center

    ,

     with

     its

     many

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     exciting

     and

     disruptive

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

     Integration

    :

     AI

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     voice

     assistants

     like

     Amazon

    's

     Alexa

     and

     Google

     Assistant

     to

     self

    -driving

     cars

     and

     robots

     in

     manufacturing

    .

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     it

     will

     become

     more

     accessible

     and

     affordable

    .
    


    2

    .

     Enhanced

     Privacy

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     greater

     concerns

     about

     privacy

     and

     data

     security

    .

     There

     will

     be

     increasing

     efforts

     to

     protect

     the

     privacy

     of

     individuals

     as

     AI

     becomes

     more

     prevalent

     in

     our

     lives

    .
    


    3

    .

     Adv

    ancements

     in

     Human

    -

    Computer

     Interaction

    :

     AI

     will

     continue

    



```python
llm.shutdown()
```
