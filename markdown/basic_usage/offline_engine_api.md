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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.88it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 20.37it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:03, 17.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:03, 17.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:03, 17.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   7%|▋         | 4/58 [00:00<00:03, 17.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.08it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.40it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.55it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 42.33it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.34it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.34it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.54it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.67it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.18it/s]


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
    Generated text:  Mary and I'm a science fiction writer who loves to think of characters. I have always loved fantasy and science fiction, but I've always wanted to explore the world of literature. I'm currently working on my first novel and I've been writing for many years. My first novel is called "Riddles" and it's about a young girl who discovers she has the power to communicate with the dead. My main goal is to create a world that is different from the one I've been used to, and I want my readers to be able to connect with the world of fantasy and science fiction. Can you suggest some books that I should
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. ( )
    A. Correct
    B. Incorrect
    C. Uncertain
    To determine whether the statement "The president of the United States is a person" is correct, we need to understand the role of the president in the United States government.
    
    1. **Definition of President**: The president of the United States is the head of the executive branch of the federal government. It is the commander-in-chief of the armed forces, the director of the Central Intelligence Agency, and other important roles. The president is responsible for appointing federal judges, other federal officers, and other important appointments.
    
    2. **Nature of a Person**:
    ===============================
    Prompt: The capital of France is
    Generated text:  _____. A. Paris
    B. Paris
    C. Paris
    D. Paris
    Answer:
    D
    
    [Multiple Choice Question] In a certain region, 500 companies were surveyed, and the average wage per person was 5,000 yuan. What is the sample size for this survey? 
    A. 500 
    B. 50 
    C. 5000 
    D. 50000
    Answer:
    A. 500
    
    Which of the following statements about the three levels of people's courts are correct?
    A. The third level of people's
    ===============================
    Prompt: The future of AI is
    Generated text:  not in the tech industry, but in the history of the world. It’s a fascinating subject, and one that I believe you’ll find interesting in this article. In the world of AI, the field is always changing and evolving. This article will explore the future of AI, including the ways in which it can change society and the world around us. The article will also discuss the ways in which AI is already changing the world around us and how it will continue to do so in the years ahead.
    One of the most promising areas for AI in the future is in healthcare. AI has the potential to revolutionize the way we diagnose and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age], [gender], [nationality], [occupation]. I have a [job title] at [company name], and I'm always looking for ways to [describe your job or passion]. I enjoy [describe your hobbies or interests]. I'm always looking for ways to [describe your goals or aspirations]. I'm a [describe your personality or character]. I'm always looking for ways to [describe your strengths or weaknesses]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also the country's largest city and the second most populous. Paris is a cultural and economic center, known for its art, music, and fashion. It is also home to many famous landmarks and museums, including the Louvre and the Notre-Dame Cathedral. The city is known for its romantic atmosphere and is a popular tourist destination. Paris is a vibrant and dynamic city with a rich history and a unique blend of old and new. It is a city that has played a significant role in French culture and history.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk
    


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
    Generated text:  _____. I am a/an _____. I’m an AI language model created by _____. I was created by _____. I was created at _____. I am a/an _____. I am a/an _____. I was created for _____. I was created for _____. I am a/an _____. I was created at _____. I am a/an _____. I was created for _____. I am a/an _____. I was created at _____. I am a/an _____. I was created for _____. I am a/an _____. I was created at _____. I am a/an _____. I was created for _____. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the city-state of France and the largest city in the European Union. It was founded in 787 AD by Notre Dame de Paris and is the third largest city in the world by population. It is also one of the world’s cultural and intellectual centers. Paris is known for its world-class museums, art galleries, and operas. It has a long history dating back to Roman times, and has been the capital of France since 1358. It is also the birthplace of the French Revolution. Paris is a large and diverse city with a rich history, beautiful architecture, and a lively atmosphere
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but it is likely to continue to evolve and develop rapidly over the coming years. Here are some possible future trends in artificial intelligence:
    
    1. Increased reliance on AI in personal and professional settings: AI is increasingly being used in areas such as healthcare, finance, transportation, and customer service. As AI becomes more accessible and affordable, its use in these industries is likely to increase.
    
    2. Improved personalization and efficiency: AI is likely to continue improving as it learns to understand and tailor its responses to the individual user's preferences and needs. This could lead to more efficient and personalized service, as well as improved efficiency for businesses.
    
    


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

     a

     [

    profession

     or

     person

    ]

     with

     a

     passion

     for

     [

    specific

     interest

     or

     hobby

    ].

     I

     am

     always

     eager

     to

     learn

    ,

     to

     grow

    ,

     and

     to

     challenge

     myself

    .

     I

     have

     a

     keen

     eye

     for

     detail

    ,

     and

     I

     love

     to

     take

     things

     one

     step

     at

     a

     time

    .

     I

     am

     always

     looking

     for

     opportunities

     to

     learn

     from

     others

    ,

     and

     I

     am

     always

     willing

     to

     take

     risks

    .

     I

     am

     passionate

     about

     providing

     a

     positive

     and

     helpful

     experience

     to

     those

     I

     interact

     with

    ,

     and

     I

     strive

     to

     make

     a

     positive

     impact

     on

     the

     world

    .

     What

     is

     your

     profession

     or

     interest

    ?

     [

    What

     is

     your

     profession

     or

     interest

    ?

    ]


    [

    Your

     name

    ]

     is

     a

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     European

     Union

     and

     the

     most

     populous

     city

     in

     the

     European

     Union

    .

     It

     has

     a

     population

     of

     over

     

    1

    0

     million

     people

    .

     Paris

     is

     known

     as

     the

     City

     of

     Love

     and

     the

     City

     of

     Light

    ,

     and

     is

     also

     the

     birth

    place

     of

     many

     of

     France

    's

     most

     famous

     musicians

    .

     It

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     most

     visited

     museum

     in

     the

     world

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     The

     city

     is

     also

     home

     to

     many

     historical

     and

     cultural

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     Ch

    amps

    -

    É

    lys

    ées

    ,

     and

     Mont

    mart

    re

    .

     Paris

     is

     known

     for

     its

     fashion

     industry

     and

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

     and

     is

     expected

     to

     evolve

     rapidly

    .

     Here

     are

     some

     possible

     trends

     that

     could

     emerge

     in

     the

     future

    :
    


    1

    .

     Enhanced

     Privacy

     and

     Ethics

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    ,

     we

     will

     need

     to

     ensure

     that

     AI

     systems

     are

     ethical

    ,

     transparent

    ,

     and

     compliant

     with

     human

     values

    .

     The

     need

     for

     privacy

     protection

     will

     increase

    ,

     and

     there

     will

     be

     a

     push

     to

     improve

     data

     protection

     and

     privacy

     policies

    .
    


    2

    .

     AI

     Self

    -

    Impro

    vement

    :

     AI

     will

     become

     more

     self

    -learning

     and

     self

    -im

    pro

    ving

    ,

     leading

     to

     new

     breakthrough

    s

     and

     capabilities

    .

     Self

    -re

    p

    lication

     and

     self

    -aware

    ness

     are

     among

     the

     possibilities

    .
    


    3

    .

     AI

     with

     Natural

     Language

     Processing

    :

     AI

     will

    



```python
llm.shutdown()
```
