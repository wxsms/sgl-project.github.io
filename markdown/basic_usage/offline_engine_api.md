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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.91it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.46it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.18it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.68it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.68it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.68it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.68it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.68it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.68it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.68it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:04, 13.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:04, 13.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.67 GB):   7%|▋         | 4/58 [00:00<00:04, 13.09it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.67 GB):  10%|█         | 6/58 [00:00<00:03, 13.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.65 GB):  10%|█         | 6/58 [00:00<00:03, 13.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:03, 13.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:03, 13.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.16 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.16 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.48it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.15 GB):  21%|██        | 12/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.15 GB):  21%|██        | 12/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.15 GB):  21%|██        | 12/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.14 GB):  21%|██        | 12/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.14 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.10 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.86 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.85 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.95it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=73.85 GB):  31%|███       | 18/58 [00:00<00:01, 23.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.85 GB):  31%|███       | 18/58 [00:00<00:01, 23.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.85 GB):  31%|███       | 18/58 [00:00<00:01, 23.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.83 GB):  31%|███       | 18/58 [00:00<00:01, 23.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.83 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.03it/s]Capturing num tokens (num_tokens=960 avail_mem=73.84 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.03it/s] Capturing num tokens (num_tokens=896 avail_mem=73.84 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.03it/s]Capturing num tokens (num_tokens=832 avail_mem=73.84 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.03it/s]

    Capturing num tokens (num_tokens=832 avail_mem=73.84 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.50it/s]Capturing num tokens (num_tokens=768 avail_mem=73.83 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.50it/s]Capturing num tokens (num_tokens=704 avail_mem=73.83 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.50it/s]Capturing num tokens (num_tokens=640 avail_mem=73.83 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.50it/s]Capturing num tokens (num_tokens=640 avail_mem=73.83 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.20it/s]Capturing num tokens (num_tokens=576 avail_mem=73.83 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.20it/s]Capturing num tokens (num_tokens=512 avail_mem=73.81 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.20it/s]

    Capturing num tokens (num_tokens=480 avail_mem=73.83 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.20it/s]Capturing num tokens (num_tokens=480 avail_mem=73.83 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.32it/s]Capturing num tokens (num_tokens=448 avail_mem=73.82 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.32it/s]Capturing num tokens (num_tokens=416 avail_mem=73.82 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.32it/s]Capturing num tokens (num_tokens=384 avail_mem=73.82 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.32it/s]Capturing num tokens (num_tokens=384 avail_mem=73.82 GB):  57%|█████▋    | 33/58 [00:01<00:01, 22.10it/s]Capturing num tokens (num_tokens=352 avail_mem=73.82 GB):  57%|█████▋    | 33/58 [00:01<00:01, 22.10it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.81 GB):  57%|█████▋    | 33/58 [00:01<00:01, 22.10it/s]Capturing num tokens (num_tokens=288 avail_mem=73.81 GB):  57%|█████▋    | 33/58 [00:01<00:01, 22.10it/s]Capturing num tokens (num_tokens=288 avail_mem=73.81 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.30it/s]Capturing num tokens (num_tokens=256 avail_mem=73.81 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.30it/s]Capturing num tokens (num_tokens=240 avail_mem=73.80 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.30it/s]Capturing num tokens (num_tokens=224 avail_mem=73.80 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.30it/s]Capturing num tokens (num_tokens=224 avail_mem=73.80 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.07it/s]Capturing num tokens (num_tokens=208 avail_mem=73.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.07it/s]

    Capturing num tokens (num_tokens=192 avail_mem=73.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.07it/s]Capturing num tokens (num_tokens=176 avail_mem=73.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.07it/s]Capturing num tokens (num_tokens=176 avail_mem=73.79 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.13it/s]Capturing num tokens (num_tokens=160 avail_mem=73.79 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.13it/s]Capturing num tokens (num_tokens=144 avail_mem=73.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.13it/s]Capturing num tokens (num_tokens=128 avail_mem=73.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.13it/s]

    Capturing num tokens (num_tokens=128 avail_mem=73.78 GB):  78%|███████▊  | 45/58 [00:02<00:00, 23.23it/s]Capturing num tokens (num_tokens=112 avail_mem=73.78 GB):  78%|███████▊  | 45/58 [00:02<00:00, 23.23it/s]Capturing num tokens (num_tokens=96 avail_mem=73.78 GB):  78%|███████▊  | 45/58 [00:02<00:00, 23.23it/s] Capturing num tokens (num_tokens=80 avail_mem=73.68 GB):  78%|███████▊  | 45/58 [00:02<00:00, 23.23it/s]Capturing num tokens (num_tokens=80 avail_mem=73.68 GB):  83%|████████▎ | 48/58 [00:02<00:00, 21.17it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  83%|████████▎ | 48/58 [00:02<00:00, 21.17it/s]Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:02<00:00, 21.17it/s]

    Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:02<00:00, 21.17it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:02<00:00, 21.17it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  83%|████████▎ | 48/58 [00:02<00:00, 21.17it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.39it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.39it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.39it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.39it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.39it/s] Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.39it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:02<00:00, 24.08it/s]


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
    Generated text:  Al Sweigart, and I am an avid programmer and computer hobbyist. I have been programming since I was a boy, and programming has become an important part of my life. The other day, I was playing with some basic programming concepts and I thought, "What if I gave this program a name? Can I put a name on something?" I had heard of the phrase "Software Name" (SNA) and wanted to try it out. It just so happened that the name "SNA" was already taken, so I took the name "Al" and I gave it a name. Here is what I wrote the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. His job is to make important decisions. He is the head of the government. He's a very important person because he's the one who controls all the big things. He's also the boss of the country. The president is the leader of the government. The president can't make changes to the laws or the business of the country. He has to work very hard to make sure he's a good leader. He is the only person who can make decisions about what the country does. He is called the "unlike" person. Everyone knows this. The president has to make important decisions. He has to
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. London
    C. Rome
    D. Washington D. C.
    Answer:
    
    A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Rome
    D. Washington D. C.
    Answer:
    
    A
    
    Which of the following is NOT a measure to improve the efficiency of food production?
    A. Improve storage and transportation technologies
    B. Increase production capacity
    C. Adopt new technologies
    D. Reduce waste
    Answer:
    
    B
    
    When the body is in a cold state, which type of blood vessel is dilated?
    A. Aorta
    B. Vein
    C
    ===============================
    Prompt: The future of AI is
    Generated text:  in the software, not hardware. Startups like Google and Facebook are creating the software for the next phase of AI, and the tools they are developing are pushing the boundaries of what the computer can do.
    But do we even need the software that Google, Facebook, and other companies are developing? Does it really matter where the computer sits in the system? Does the hardware matter? Is the computer a panacea for all our problems? That's the big question being debated by experts in the AI field.
    The software is the foundation, but the hardware is the real thing. The hardware makes the software work, and without a working computer,


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], and I'm [Appearance]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I enjoy [Favorite Hobby/Activity]. I'm always looking for ways to improve myself and make the world a better place. What's your dream job? I dream of [Dream Job], where I can use my skills and knowledge to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich cultural heritage and a vibrant nightlife. It is located in the south of France and is the largest city in the country. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also famous for its fashion industry, art scene, and its role as a center of science and technology. Paris is a popular tourist destination and is home to many museums, theaters, and restaurants. It is also a major economic hub and a major transportation hub. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to customer service. We can expect to see more automation in industries such as manufacturing, healthcare, and transportation.
    
    2. Improved privacy and security: As AI becomes more integrated into our lives, there will be increased concerns about privacy and security. We can expect to see more regulations and standards to protect people's data and prevent AI from being used for malicious purposes.
    
    3. Enhanced human-computer interaction: AI will continue to become more sophisticated, allowing for more natural and intuitive interactions between humans and
    


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
    Generated text: ... I don't have a name, but I'm a kind-hearted, empathetic person who works as a freelance graphic designer. I'm constantly learning and growing, and I'm always looking for new opportunities to make a difference in the world. I believe that creativity is the key to solving problems and creating meaningful connections. And I love spending time with my loyal team of graphic designers, as well as my loyal customers. I hope to continue working together to create amazing designs and experiences for our community.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as La Sainte-Victoire.
    
    Paris is the largest city in France and the capital of the country. It has a rich history and is known for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. The city is also home to many cultural institutions, including the Musée Rodin and the Musée d'Orsay. Paris is a vibrant and diverse city with a rich cultural heritage that is reflected in its architecture, food, and fashion. It is the second-largest city in Europe and is an important economic and cultural center for France. Paris has been recognized
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting and constantly evolving. Here are some possible trends in AI that could become more prevalent in the coming years:
    
    1. Increased emphasis on ethical considerations: As more AI technologies are developed, there will be a growing emphasis on ethical considerations. This will include topics such as data privacy, bias, and transparency in decision-making processes. There will also be a push towards designing AI systems that are transparent and accountable to users.
    
    2. More integration with human workers: With the rise of automation and AI, there will be a growing interest in integrating AI systems with human workers. This will require a shift in how we design and develop AI systems,


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

     [

    Career

    ]

     who

     has

     been

     active

     in

     [

    Field

    /

    Industry

    ]

     for

     [

    Number

     of

     Years

    ].

     I

     bring

     a

     unique

     blend

     of

     [

    Background

     Skills

    ]

     and

     [

    Professional

     Experience

    ].

     I

     strive

     to

     [

    Personal

     Goal

    ]

     and

     am

     always

     looking

     to

     [

    Attr

    actions

    ],

     [

    Skills

    ],

     or

     [

    Adv

    antages

    ]

     that

     make

     me

     a

     great

     fit

     for

     this

     role

    .

     Please

     let

     me

     know

     what

     you

     would

     like

     me

     to

     say

     when

     I

     introduce

     myself

    .

     Hi

     there

    !

     My

     name

     is

     [

    Name

    ]

     and

     I

     am

     a

     [

    Career

    ]

     with

     [

    Number

     of

     Years

    ]

     of

     experience

    .

     I

     bring

     a

     unique

     blend

     of

     [

    Background

     Skills

    ]

     and

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    To

     provide

     additional

     context

    ,

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     largest

     metropolitan

     area

     in

     the

     world

    .

     It

     is

     home

     to

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     iconic

     landmarks

     and

     attractions

    .

     Paris

     is

     also

     known

     for

     its

     rich

     cultural

     heritage

    ,

     including

     French

     cuisine

    ,

     museums

    ,

     and

     fashion

    .

     The

     city

     is

     a

     hub

     of

     commerce

    ,

     industry

    ,

     and

     entertainment

    ,

     and

     is

     a

     major

     international

     center

     for

     education

    ,

     diplomacy

    ,

     and

     government

    .

     As

     the

     seat

     of

     government

     and

     the

     capital

     of

     France

    ,

     Paris

     plays

     a

     central

     role

     in

     the

     country

    's

     political

     and

     economic

     life

    .

     Its

     influence

     extends

     beyond

     France

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     varied

    ,

     with

     many

     trends

     and

     potential

     developments

     shaping

     how

     it

     will

     evolve

     in

     the

     coming

     years

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

     integration

     with

     human

     intelligence

    :

     As

     AI

     becomes

     more

     capable

    ,

     it

     will

     likely

     become

     even

     more

     integrated

     with

     human

     intelligence

    .

     This

     could

     lead

     to

     a

     more

     complex

     and

     multi

    -f

    ac

    eted

     AI

     system

     that

     can

     perform

     tasks

     that

     are

     beyond

     the

     capabilities

     of

     any

     single

     AI

     system

    .
    


    2

    .

     Better

     ethical

     and

     legal

     considerations

    :

     As

     more

     people

     become

     aware

     of

     the

     potential

     risks

     and

     ethical

     issues

     associated

     with

     AI

    ,

     there

     will

     be

     an

     increased

     emphasis

     on

     developing

     better

     ethical

     and

     legal

     standards

     for

     AI

    .

     This

     could

     lead

     to

     new

    



```python
llm.shutdown()
```
