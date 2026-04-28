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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.22it/s]


    2026-04-28 00:03:41,264 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 00:03:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:41,  4.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.63it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 20.43it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.33 GB):   7%|▋         | 4/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   7%|▋         | 4/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):   7%|▋         | 4/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.31 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.99it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.29it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.29it/s] Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  38%|███▊      | 22/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 44.83it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 44.83it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 44.83it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 44.83it/s]Capturing num tokens (num_tokens=416 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 44.83it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 44.83it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 44.83it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 47.12it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 47.12it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 47.12it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 47.12it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:00<00:00, 47.12it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:00<00:00, 47.12it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:00<00:00, 47.12it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  69%|██████▉   | 40/58 [00:00<00:00, 48.90it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  69%|██████▉   | 40/58 [00:00<00:00, 48.90it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.90it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.90it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.90it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.90it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.90it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 49.98it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=32 avail_mem=72.19 GB):  79%|███████▉  | 46/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=32 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.97it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.97it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.97it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.97it/s]

    Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.97it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.97it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 49.97it/s] Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  98%|█████████▊| 57/58 [00:01<00:00, 50.51it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 50.51it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 43.13it/s]


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
    Generated text:  Glenn and I'm a physical therapist working with adults and children. I've been doing this for about 16 years. My physical therapists work to help people who have injuries, diseases and conditions like arthritis, brain injuries, stroke, and depression. I also work with individuals with cognitive, emotional, and psychological issues. My therapy focus is to help people become more active, work out, and live more fulfilling lives. I also encourage people to be kind and compassionate to others and to work hard and take care of themselves. My goal is to help people live in a healthier, happier, and more productive life. I have a specialty in
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with rotating terms. What is the correct term for this position? The term for the position of president of the United States is "president." The correct term for the position is "president," which is a term of office with a fixed term of four years and is not a political office in the sense of having rotating terms. The President of the United States has a term of office that allows them to be re-elected for a fixed number of terms. However, the term is not a political office, but rather a term of office with a fixed term of four years. The President is elected by the people and has
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Brussels C. Lyon D. Nice
    Answer: A
    
    Which of the following statements is incorrect?
    A. A blockchain is a technology that allows transactions to be made on a large scale without centralization.
    B. Bitcoin is a form of digital currency.
    C. The Bitcoin blockchain is the same as the blockchain.
    D. The blockchain is a type of digital currency.
    Answer: C
    
    Which of the following statements is incorrect?
    A. A blockchain is a technology that allows transactions to be made on a large scale without centralization.
    B. Bitcoin is a form of digital currency.
    C. The blockchain
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting and diverse, with robots becoming more and more intelligent, and devices becoming more and more adaptable. While we're excited to see how these technologies can be used to improve human lives, it's important to consider the potential ethical implications of these developments.
    One potential concern with AI is that it can lead to job displacement. With robots becoming more advanced, it's likely that some jobs that are currently automated will become less necessary in the future. This could lead to a ripple effect that can be difficult to predict and is therefore a concern for policymakers.
    Another potential ethical concern is the potential for AI to be used for surveillance and data collection. This


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I have always been passionate about [Your Passion], and I am always looking for new ways to [Your Goal]. I am always eager to learn and grow, and I am always open to new experiences. I am a [Your Personality] person, and I am always looking for ways to [Your Personality Trait]. I am always looking for new challenges and opportunities to [Your Personality Trait]. I am always looking for ways to [Your Personality Trait]. I am always looking for ways to [Your Personality Trait]. I am always looking for ways to [Your Personality
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in the European Union and the third-largest city in the world by population. Paris is known for its rich history, art, and culture, and is a major tourist destination. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination for its beautiful architecture, vibrant nightlife, and delicious cuisine. It is also a major center for business and finance, with many international companies and institutions headquartered there. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. The city is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to customer service. We may see more automation in areas like manufacturing, where machines can perform tasks that were previously done by humans, such as assembly line work or repetitive tasks.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, we may see more privacy and security concerns. We may
    


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
    Generated text:  Alex. I am a [insert your profession] who has been working for [insert your company name] for [insert your duration of employment] years. I have always been passionate about [insert something you enjoy doing] and have always been a true team player. I am always eager to learn and continue to grow as a professional. If you have any questions or if there is anything I can do for you, don't hesitate to reach out. I am always ready to help! What is your profession and what is your current position with [insert company name]?
    [Your Name] is a [insert your profession] who has been
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    In addition, provide at least two other facts about Paris:
    
    1. The city is located on the Île de France, which is an island off the coast of northern France. This makes it one of the five largest cities in France by area.
    2. The city is home to the Eiffel Tower, the world's tallest and most iconic building, which is situated on a cliff overlooking the Seine River. The tower was commissioned by Napoleon Bonaparte and completed in 1889. 
    
    Please note that the information provided is factual, and any facts that you add should be specific and well-supported with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, and there are many possible trends that could shape the technology's development. Here are some potential future trends in artificial intelligence:
    
    1. Increased use of AI in healthcare: AI is already being used in the healthcare industry to diagnose and treat diseases. We may see more widespread use of AI in the future as it becomes more widely accepted and as the technology becomes more sophisticated.
    
    2. Integration with human decision-making: AI is already being used to help humans make decisions. We may see even more integration of AI into human decision-making in the future as AI becomes more sophisticated and better able to make accurate predictions and decisions.
    
    3. Use of


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

     a

     [

    Age

    ]

     year

     old

    ,

     [

    Occup

    ation

    ],

     with

     [

    You

    th

    ful

     Traits

    ]

     personality

     type

    .

     I

     love

     spending

     my

     days

     exploring

     the

     world

     and

     learning

     new

     things

    ,

     whether

     it

    's

     reading

    ,

     writing

    ,

     or

     solving

     puzzles

    .

     I

    'm

     always

     looking

     for

     a

     challenge

    ,

     and

     I

     love

     sharing

     my

     knowledge

     and

     experiences

     with

     others

    .

     I

    'm

     an

     [

    Your

     Specialty

    ],

     and

     I

     would

     love

     to

     connect

     with

     anyone

     who

     wants

     to

     learn

     from

     me

    .

     Thank

     you

     for

     considering

     me

     for

     a

     character

     to

     introduce

    !

     

    🌟

    📚

    📚

    ✨

    


    Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ],

     and

     I

     am

     a

     [

    Age

    ]

     year

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     cuisine

    .
    


    That

    's

     great

    !

     Can

     you

     provide

     some

     additional

     information

     about

     Paris

     that

     I

     might

     find

     helpful

    ?

     Sure

    ,

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     and

     it

     is

     a

     city

     of

     incredible

     beauty

    ,

     culture

    ,

     and

     history

    .

     It

    's

     the

     largest

     city

     in

     Europe

    ,

     with

     a

     population

     of

     over

     

    2

     million

     people

    .

     Paris

     is

     famous

     for

     its

     iconic

     landmarks

    ,

     such

     as

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

     Arc

     de

     Tri

    omp

    he

    .

     It

    's

     also

     known

     for

     its

     vibrant

     food

     scene

    ,

     including

     fancy

     bist

    ros

    ,

     gourmet

     cuisine

    ,

     and

     some

     of

     the

     best

     cheese

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     combination

     of

     emerging

     technologies

     and

     evolving

     patterns

     of

     usage

    .

     Some

     potential

     trends

     that

     are

     currently

     on

     the

     horizon

     and

     could

     lead

     to

     significant

     changes

     in

     the

     field

     include

    :
    


    1

    .

     Increased

     Use

     of

     Explain

    able

     AI

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     their

     behavior

     and

     decision

    -making

     processes

     are

     likely

     to

     become

     more

     understandable

     to

     humans

    .

     This

     could

     lead

     to

     the

     development

     of

     more

     accessible

     AI

     tools

     that

     allow

     users

     to

     understand

     how

     AI

     systems

     work

     and

     make

     decisions

     based

     on

     the

     results

    .
    


    2

    .

     Greater

     Integration

     of

     AI

     with

     Other

     Technologies

    :

     As

     AI

     continues

     to

     become

     more

     integrated

     into

     various

     fields

    ,

     its

     capabilities

     and

     applications

     are

     likely

     to

     expand

    .

     For

     example

    ,

     AI

     may

    



```python
llm.shutdown()
```
