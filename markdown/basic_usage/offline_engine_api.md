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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.39it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.37it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.37it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.37it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.37it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.37it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 14.37it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 14.37it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:02, 14.37it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 18.99it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 18.99it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.99it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.99it/s]

    Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.99it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.99it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.99it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 23.61it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 23.61it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 23.61it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 23.61it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 23.61it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 23.61it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 23.61it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 23.61it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.09 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.54it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.34 GB):  21%|██        | 12/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.34 GB):  21%|██        | 12/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.34 GB):  21%|██        | 12/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.33 GB):  21%|██        | 12/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.33 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.14 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.16it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.31 GB):  31%|███       | 18/58 [00:00<00:01, 22.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.30 GB):  31%|███       | 18/58 [00:00<00:01, 22.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.30 GB):  31%|███       | 18/58 [00:00<00:01, 22.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.27 GB):  31%|███       | 18/58 [00:00<00:01, 22.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.27 GB):  36%|███▌      | 21/58 [00:00<00:01, 23.97it/s]Capturing num tokens (num_tokens=960 avail_mem=74.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 23.97it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 23.97it/s]Capturing num tokens (num_tokens=832 avail_mem=74.27 GB):  36%|███▌      | 21/58 [00:00<00:01, 23.97it/s]Capturing num tokens (num_tokens=768 avail_mem=74.26 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.97it/s]Capturing num tokens (num_tokens=768 avail_mem=74.26 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.21it/s]Capturing num tokens (num_tokens=704 avail_mem=74.26 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.21it/s]Capturing num tokens (num_tokens=640 avail_mem=74.25 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.21it/s]Capturing num tokens (num_tokens=576 avail_mem=74.25 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.21it/s]Capturing num tokens (num_tokens=512 avail_mem=74.23 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.21it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.23 GB):  50%|█████     | 29/58 [00:01<00:01, 28.85it/s]Capturing num tokens (num_tokens=480 avail_mem=74.24 GB):  50%|█████     | 29/58 [00:01<00:01, 28.85it/s]Capturing num tokens (num_tokens=448 avail_mem=74.22 GB):  50%|█████     | 29/58 [00:01<00:01, 28.85it/s]Capturing num tokens (num_tokens=416 avail_mem=74.24 GB):  50%|█████     | 29/58 [00:01<00:01, 28.85it/s]Capturing num tokens (num_tokens=384 avail_mem=74.23 GB):  50%|█████     | 29/58 [00:01<00:01, 28.85it/s]Capturing num tokens (num_tokens=384 avail_mem=74.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=352 avail_mem=74.22 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=320 avail_mem=74.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=288 avail_mem=74.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=256 avail_mem=74.20 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.30it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.61it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.61it/s]Capturing num tokens (num_tokens=224 avail_mem=74.18 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.61it/s]Capturing num tokens (num_tokens=208 avail_mem=74.17 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.61it/s]Capturing num tokens (num_tokens=192 avail_mem=74.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.61it/s]Capturing num tokens (num_tokens=192 avail_mem=74.16 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=176 avail_mem=74.16 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=160 avail_mem=74.16 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=144 avail_mem=74.16 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.16 GB):  71%|███████   | 41/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=128 avail_mem=74.16 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.15it/s]Capturing num tokens (num_tokens=112 avail_mem=74.15 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.15it/s]Capturing num tokens (num_tokens=96 avail_mem=74.15 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.15it/s] Capturing num tokens (num_tokens=80 avail_mem=74.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.15it/s]Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.15it/s]Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.47it/s]Capturing num tokens (num_tokens=48 avail_mem=74.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.47it/s]Capturing num tokens (num_tokens=32 avail_mem=74.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.47it/s]Capturing num tokens (num_tokens=28 avail_mem=74.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.47it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.11 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.47it/s]Capturing num tokens (num_tokens=24 avail_mem=74.11 GB):  91%|█████████▏| 53/58 [00:02<00:00, 25.61it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  91%|█████████▏| 53/58 [00:02<00:00, 25.61it/s]Capturing num tokens (num_tokens=16 avail_mem=74.10 GB):  91%|█████████▏| 53/58 [00:02<00:00, 25.61it/s]Capturing num tokens (num_tokens=12 avail_mem=74.09 GB):  91%|█████████▏| 53/58 [00:02<00:00, 25.61it/s]Capturing num tokens (num_tokens=8 avail_mem=74.08 GB):  91%|█████████▏| 53/58 [00:02<00:00, 25.61it/s] Capturing num tokens (num_tokens=4 avail_mem=74.08 GB):  91%|█████████▏| 53/58 [00:02<00:00, 25.61it/s]Capturing num tokens (num_tokens=4 avail_mem=74.08 GB): 100%|██████████| 58/58 [00:02<00:00, 29.38it/s]Capturing num tokens (num_tokens=4 avail_mem=74.08 GB): 100%|██████████| 58/58 [00:02<00:00, 26.65it/s]


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
    Generated text:  Sera. I am a college student from New York. I'm interested in taking on a job. I am applying to pursue a Bachelor of Science in Biology from the University of Pittsburgh. I am planning to apply for a job in the field of Biotechnology. I have read many interviews, and I have had a chance to learn about the work of some of the top scientists and business people in the field. I have also done research and attended classes on biotechnology. I am currently applying to be a research assistant in the lab. I want to apply for a job position that requires working with the latest technology and equipment in the field
    ===============================
    Prompt: The president of the United States is
    Generated text:  a kind of ____.
    A. Administrative organization
    B. Political party
    C. Political movement
    D. Council
    Answer: B
    
    Among the following, the method suitable for the project cost preparation of a construction company is ____
    A. Single-item Project Costing
    B. Planned Costing
    C. Comprehensive Project Costing
    D. Manual Costing
    Answer: B
    
    The clinical application of ultrasonic therapy is ____
    A. Rheumatism
    B. Joint inflammation
    C. Carpal joint pain
    D. Hip and knee joint pain
    E. Osteoporosis
    Answer: C
    
    The most suitable
    ===============================
    Prompt: The capital of France is
    Generated text:  __________.
    A. Paris
    B. London
    C. New York
    D. Tokyo
    Answer: A
    
    The school has more students than ____.
    A. There
    B. One
    C. No
    D. None
    Answer: B
    
    Please, please, please.
    A. Please
    B. Thank you
    C. Excuse me
    D. Goodbye
    Answer: B
    
    Which of the following options is the first step in the process of selecting a topic?
    A. Determine the purpose of the topic
    B. Set the date
    C. Choose the method of research
    D. Decide on the type
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s not just robots, it’s smart devices, smart cities, smart homes, smart healthcare, smart homes, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities, smart cities,


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


    Generated text:  [Name] and I'm a [occupation] with [number] years of experience in [field]. I'm a [type of person] who is always [positive trait]. I'm [age] years old and I'm [gender]. I'm [occupation] with [number] years of experience in [field]. I'm a [type of person] who is always [positive trait]. I'm [age] years old and I'm [gender]. I'm [occupation] with [number] years of experience in [field]. I'm a [type of person] who is always [positive trait]. I'm [age
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. The city is known for its fashion industry, art scene, and cuisine, and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical and social implications: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical and social implications. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased public awareness and engagement around the potential risks and benefits of AI.
    
    
    


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
    Generated text:  [insert name here]. I come from [insert country]. I am passionate about [insert hobby or interest]. I have always been curious about the world, and I enjoy learning new things and exploring different cultures. I have a good sense of humor and love to laugh. I am always looking for new experiences and adventures, and I am always eager to learn and grow. I am a social butterfly and love to hang out with friends and family. I enjoy music, art, and reading. I am an independent and goal-oriented person, and I am constantly striving to achieve my goals. I am a self-motivated person who is always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is the largest city in the country and is known for its rich cultural heritage, iconic landmarks, and world-class museums. The city is home to a variety of attractions, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many others. Paris has a rich history dating back to the Roman Empire and is famous for its bustling nightlife and art scene. The city is also known for its important role in French politics and culture, including the French Revolution and the rise of the French Republic. Paris is a popular tourist destination, with millions of visitors annually. It is a city that is steeped
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant advancements in areas such as deep learning, natural language processing, robotics, autonomous vehicles, and quantum computing. Here are some possible trends to watch for:
    
    1. Enhanced computer vision: As deep learning continues to advance, we may see even more sophisticated computer vision algorithms that can better identify and analyze objects and scenes in the world around us.
    
    2. Improved natural language processing: With the help of machine learning, natural language processing will become even more sophisticated, enabling AI to understand human language and emotions better.
    
    3. Increased autonomous vehicles: As autonomous vehicles become more advanced, we may see them in more and more places


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

     I

    'm

     a

     [

    Role

    /

    Title

    ]

     at

     [

    Organization

    /O

    rganization

     Name

    ].

     I

    'm

     a

     software

     engineer

     with

     a

     strong

     background

     in

     [

    Technology

    /

    Field

    ].

     I

     love

     coding

     and

     working

     with

     tools

     like

     [

    Technology

    /

    Tool

    ].

     I

     am

     passionate

     about

     [

    Why

     you

    're

     passionate

     about

     your

     job

    ].


    My

     journey

     has

     been

     full

     of

     challenges

     and

     successes

    .

     I

     have

     always

     been

     [

    Positive

     Qual

    ities

    /

    Personal

     Traits

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     grow

     and

     improve

    .

     What

     exc

    ites

     me

     most

     about

     my

     role

     at

     [

    Organization

    /O

    rganization

     Name

    ]

     is

     [

    What

     exc

    ites

     you

     most

     about

     your

     job

    ].


    I

     am

     always

     open

     to

     learning

     and

     growing

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     also

     known

     as

     "

    La

     Grande

     N

    uit

    "

     in

     French

    ,

     is

     the

     largest

     city

     in

     France

     by

     area

     and

     population

    ,

     and

     the

     capital

     of

     the

     L

    angu

    ed

    oc

     region

    .

     It

     is

     located

     in

     the

     Î

    le

    -de

    -F

    rance

     region

     and

     is

     the

     seat

     of

     government

    ,

     politics

    ,

     culture

    ,

     media

    ,

     and

     diplomacy

     in

     France

    .

     It

     is

     also

     the

     headquarters

     of

     France

    's

     foreign

     ministry

     and

     home

     to

     the

     French

     Academy

     of

     Sciences

    .

     Paris

     is

     famous

     for

     its

     rich

     history

    ,

     art

    ,

     and

     architecture

    ,

     including

     its

     famous

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

     The

     city

     is

     also

     known

     for

     its

     festivals

    ,

     food

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

    ,

     and

     there

     are

     several

     trends

     that

     are

     likely

     to

     shape

     its

     development

     in

     the

     coming

     years

    .

     Here

     are

     some

     of

     the

     most

     promising

     areas

    :
    


    1

    .

     Super

    intelligence

    :

     Super

    intelligence

     is

     a

     term

     that

     describes

     a

     future

     scenario

     where

     machines

     are

     capable

     of

     thinking

     and

     learning

     like

     humans

    .

     Some

     experts

     think

     that

     super

    intelligence

     could

     revolution

    ize

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     each

     other

    .
    


    2

    .

     Rob

    otic

     augmentation

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

    's

     likely

     that

     robotic

     augment

    ations

     will

     become

     more

     prevalent

    .

     These

     augment

    ations

     could

     include

     things

     like

     prost

    hetic

     limbs

    ,

     robotic

     cars

    ,

     and

     even

     robots

     that

     can

     assist

     humans

     in

     everyday

     tasks

    .
    


    



```python
llm.shutdown()
```
