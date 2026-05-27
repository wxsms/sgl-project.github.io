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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:48,  4.01s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:48,  4.01s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:48,  4.01s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:48,  4.01s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:48,  4.01s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03, 10.10it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.53it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.95it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 34.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.21 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.21 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.21 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.20 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.20 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.20 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.19 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.17 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]Capturing num tokens (num_tokens=960 avail_mem=72.19 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s] Capturing num tokens (num_tokens=896 avail_mem=72.19 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]

    Capturing num tokens (num_tokens=896 avail_mem=72.19 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.70it/s]Capturing num tokens (num_tokens=832 avail_mem=72.18 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.70it/s]Capturing num tokens (num_tokens=768 avail_mem=72.18 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.70it/s]Capturing num tokens (num_tokens=704 avail_mem=72.18 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.70it/s]Capturing num tokens (num_tokens=640 avail_mem=72.17 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.70it/s]Capturing num tokens (num_tokens=576 avail_mem=72.17 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.70it/s]Capturing num tokens (num_tokens=576 avail_mem=72.17 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=512 avail_mem=72.16 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=480 avail_mem=72.17 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=448 avail_mem=72.17 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=416 avail_mem=72.17 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=384 avail_mem=72.16 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.31it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.16 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.24it/s]Capturing num tokens (num_tokens=352 avail_mem=72.16 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.24it/s]Capturing num tokens (num_tokens=320 avail_mem=72.15 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.24it/s]Capturing num tokens (num_tokens=288 avail_mem=72.15 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.24it/s]Capturing num tokens (num_tokens=256 avail_mem=72.15 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.24it/s]Capturing num tokens (num_tokens=240 avail_mem=72.15 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.24it/s]Capturing num tokens (num_tokens=240 avail_mem=72.15 GB):  66%|██████▌   | 38/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=224 avail_mem=72.14 GB):  66%|██████▌   | 38/58 [00:00<00:00, 45.65it/s]Capturing num tokens (num_tokens=208 avail_mem=72.14 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=192 avail_mem=72.14 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.65it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.13 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=160 avail_mem=72.13 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=160 avail_mem=72.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=144 avail_mem=72.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=128 avail_mem=72.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=112 avail_mem=72.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.63it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.63it/s] Capturing num tokens (num_tokens=80 avail_mem=72.08 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=80 avail_mem=72.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=64 avail_mem=72.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=48 avail_mem=72.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=32 avail_mem=72.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=28 avail_mem=72.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=28 avail_mem=72.07 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=24 avail_mem=72.07 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.37it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=16 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=12 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=12 avail_mem=72.06 GB):  97%|█████████▋| 56/58 [00:01<00:00, 33.19it/s]Capturing num tokens (num_tokens=8 avail_mem=72.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 33.19it/s] Capturing num tokens (num_tokens=4 avail_mem=72.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 33.19it/s]Capturing num tokens (num_tokens=4 avail_mem=72.05 GB): 100%|██████████| 58/58 [00:01<00:00, 35.19it/s]


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
    Generated text:  Derek.
    I am a scientist at the Stanford University School of Medicine.
    I specialize in developing new treatments for neurological disorders and are currently working on a new intervention that I call Nerve Nook.
    Here's the process of Nerve Nook:
    1. A human participant is given a nerve stimulator.
    2. The stimulator sends a current through the participant's nerves.
    3. This current stimulates the nerves and helps them to regenerate.
    4. After about 30 minutes, the participant receives a set of powerful magnet blocks.
    5. As the participant holds the magnet blocks, the stimulator sends out a very strong current.
    6
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. ____ ( )
    A. Correct
    B. Incorrect
    C. Uncertain
    D. No information provided
    Answer:
    
    A
    
    A 50-year-old male patient was admitted to the emergency department due to a sudden head injury. His blood pressure was found to be 80/60mmHg. What is the most likely diagnosis?
    A. Hypertensive crisis
    B. Hypovolemic shock
    C. Hypertensive encephalopathy
    D. Renal failure
    E. Shock
    Answer:
    
    A
    
    According to the classification of the Chinese Dietary Guidelines (2016),
    ===============================
    Prompt: The capital of France is
    Generated text:  located in ________.
    A. Montmartre
    B. Paris
    C. Montmartessur
    D. Paris
    Which of the following options is the correct answer to this question? B. Paris. 
    
    The capital of France is Paris, which is located in the northeast of the country and is the seat of government and the most important city in the country. Montmartre is a popular neighborhood in Paris, but it is not the capital of France. Montmartessur, as given, is not the capital of France either. Therefore, the correct answer is B. Paris. However, since the question asks for the
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but it is not a future to be feared. One of the most important lessons of the past few decades has been the creation of the Internet. The Internet allows millions of people to share and communicate with each other without being physically present. This makes the world more connected and more diverse. The Internet is also a great tool for improving the quality of life and making the world more open and less restricted. The development of AI (Artificial Intelligence) has been one of the most important developments of the past several decades. The AI technology is on the verge of revolutionizing the way we think, work, and live. The importance of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and landmarks. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its diverse cuisine, including French cuisine, and its vibrant nightlife. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is home to many famous artists, writers, and musicians, and its cultural heritage continues to inspire and influence the city and its residents
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased automation and artificial intelligence: As automation and AI become more prevalent, we are likely to see more jobs being automated, which could lead to a decline in wages and a rise in unemployment. However, this could also create new opportunities for people to work in areas such as data analysis, machine learning, and software development.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be a
    


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
    Generated text:  [Name], and I am [Age]. I recently graduated with my Bachelor of Science in [major]. My passion is [major], and I am currently studying hard to achieve my goals. I have always been a natural leader, and I thrive on challenges and making a difference in the world. I am a team player, and I enjoy working with people from all walks of life. I am always looking for new opportunities to learn and grow. Thank you for taking the time to meet me, and I look forward to having the opportunity to learn more about you. [Your Name] 
    (Author's note: This self-introduction should
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    A summary of the answer is Paris is the capital city of France. Paris is the capital of France, and it is where the French government and the nation's official language are located. Paris is the most populous city in France, with an estimated population of around 2. 3 million as of 2021. The city is also the third largest in the world by land area, and the second-largest in population, with a land area of approximately 782 square kilometers. Paris is a UNESCO World Heritage site, and it is home to numerous cultural landmarks and museums, including the Louvre, Notre-D
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly complex and constantly evolving. Here are some possible trends that are expected to shape the technology:
    
    1. Increased integration of AI into various industries: As AI becomes more integrated into various industries, we are likely to see more automation and improvements in productivity. This will lead to greater efficiency and lower costs.
    
    2. AI becoming more accessible to everyone: With the increasing availability of AI technology, we are likely to see AI becoming more accessible to everyone, including individuals with disabilities. This will lead to a more inclusive and equal society.
    
    3. AI becoming more ethical and transparent: As AI systems become more complex and sophisticated, it is important that we


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

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

     or

     role

    ].

     My

     [

    Favorite

     Food

    ]

     is

     a

     favorite

     thing

     to

     do

     whenever

     I

    'm

     on

     my

     [

    Travel

     Destination

    ],

     [

    City

    /

    State

    ].

     I

     enjoy

     [

    My

     Personal

     Hobby

    ],

     [

    My

     Hobby

    ],

     and

     [

    My

     Last

     Adventure

    ].

     As

     a

     [

    Future

     Goal

    /

    Project

    ],

     [

    Future

     Goal

    /

    Project

    ].

     I

     hope

     to

     [

    What

     I

     Wish

     to

     Achie

    ve

    ],

     [

    What

     I

     Wish

     to

     Achie

    ve

    ],

     and

     [

    What

     I

     Wish

     to

     Achie

    ve

    ].

     I

     am

     excited

     to

     meet

     you

     and

     see

     what

     you

     have

     to

     say

     about

     me

    !

     (

    End

     of

     Self

    -

    Introduction

    )

     
    


    Please

     note

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    .
    


    France

    ’s

     capital

     city

    ,

     Paris

    ,

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

     million

     people

    .

     The

     city

     is

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

     and

     is

     home

     to

     several

     world

    -ren

    owned

     landmarks

    ,

     including

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

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

     the

     famous

     E

    iff

    el

     Tower

     and

     the

     historic

     Lou

    vre

     Museum

    ,

     which

     house

     a

     vast

     collection

     of

     art

    ,

     architecture

    ,

     and

     historical

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     exciting

     and

     there

     are

     several

     trends

     that

     are

     predicted

     to

     shape

     the

     direction

     of

     AI

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

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     AI

     will

     become

     more

     widely

     available

    :

     In

     the

     past

    ,

     AI

     has

     been

     considered

     a

     luxury

     technology

     that

     is

     only

     accessible

     to

     a

     select

     few

    .

     However

    ,

     as

     the

     technology

     improves

     and

     becomes

     more

     accessible

    ,

     we

     can

     expect

     to

     see

     more

     widespread

     adoption

     of

     AI

     in

     various

     industries

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     with

     human

     decision

    -making

    :

     As

     AI

     technology

     improves

    ,

     we

     can

     expect

     to

     see

     more

     integration

     of

     AI

     with

     human

     decision

    -making

     processes

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     decision

    



```python
llm.shutdown()
```
