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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.55it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:47,  1.14it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:04<00:21,  2.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:10,  4.40it/s]

    Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:05,  7.73it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:05,  7.73it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:05,  7.73it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:05,  7.73it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:05,  7.73it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:05,  7.73it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:05<00:05,  7.73it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:05<00:05,  7.73it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]

    Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 21.22it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]

    Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 40.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.71 GB):   3%|▎         | 2/58 [00:00<00:05, 11.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.71 GB):   3%|▎         | 2/58 [00:00<00:05, 11.06it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=73.70 GB):   3%|▎         | 2/58 [00:00<00:05, 11.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.70 GB):   7%|▋         | 4/58 [00:00<00:04, 12.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.70 GB):   7%|▋         | 4/58 [00:00<00:04, 12.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.68 GB):   7%|▋         | 4/58 [00:00<00:04, 12.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.68 GB):  10%|█         | 6/58 [00:00<00:03, 14.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.67 GB):  10%|█         | 6/58 [00:00<00:03, 14.06it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=73.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.66 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.66 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.65 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.48it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.63 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.62 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.61 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.61 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.00it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=73.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.08it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.57 GB):  29%|██▉       | 17/58 [00:01<00:01, 23.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.57 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.09it/s]Capturing num tokens (num_tokens=960 avail_mem=73.58 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.09it/s] Capturing num tokens (num_tokens=896 avail_mem=73.57 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.09it/s]Capturing num tokens (num_tokens=832 avail_mem=73.57 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.09it/s]Capturing num tokens (num_tokens=768 avail_mem=73.56 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.09it/s]

    Capturing num tokens (num_tokens=704 avail_mem=73.56 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.09it/s]Capturing num tokens (num_tokens=704 avail_mem=73.56 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=640 avail_mem=73.55 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=576 avail_mem=73.54 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=512 avail_mem=73.54 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=480 avail_mem=73.55 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=448 avail_mem=73.55 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=448 avail_mem=73.55 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=416 avail_mem=73.54 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=384 avail_mem=73.53 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=352 avail_mem=73.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.16it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=288 avail_mem=73.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=288 avail_mem=73.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=256 avail_mem=73.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=240 avail_mem=73.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=224 avail_mem=73.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=208 avail_mem=73.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=192 avail_mem=73.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=192 avail_mem=73.48 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=176 avail_mem=73.47 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=160 avail_mem=73.47 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.46 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=128 avail_mem=73.46 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=112 avail_mem=73.45 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=112 avail_mem=73.45 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=96 avail_mem=73.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.51it/s] Capturing num tokens (num_tokens=80 avail_mem=73.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=64 avail_mem=73.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=48 avail_mem=73.42 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=32 avail_mem=73.42 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=32 avail_mem=73.42 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=28 avail_mem=73.41 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.80it/s]

    Capturing num tokens (num_tokens=24 avail_mem=73.40 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=20 avail_mem=73.40 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=16 avail_mem=73.39 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=12 avail_mem=73.38 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=12 avail_mem=73.38 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=8 avail_mem=73.38 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.76it/s] Capturing num tokens (num_tokens=4 avail_mem=73.37 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB): 100%|██████████| 58/58 [00:01<00:00, 30.23it/s]


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
    Generated text:  Charlie and I am a native speaker of English. I'm a character in the story "The Heart of the Phoenix" by John Green. The plot summary states that Charlie is a member of the "Red Rose Club" which is a fictional club in the story. I don't know anything about Charlie's past but I would love for you to tell me about Charlie's life or what happened to him. I want to know how the club was formed, who was the founders, what kind of people joined the club, and how they became successful. I would also like to know what happened to the club after the founders left and how it
    ===============================
    Prompt: The president of the United States is
    Generated text:  expected to attend which of the following?
    A) The International Labor Organization
    B) The United Nations
    C) The World Trade Organization
    D) The United Nations Security Council
    E) The United Nations General Assembly
    Answer: E) The United Nations General Assembly
    
    The president of the United States typically attends the United Nations General Assembly to deliver a speech or address the nation, discuss international affairs, and engage with leaders from various countries. The other options are not directly related to the president's role in the United States. 
    
    - The International Labor Organization (ILO) focuses on worker rights and labor standards worldwide.
    - The United Nations
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Berlin
    C. Moscow
    D. Tokyo
    Answer:
    
    A
    
    Which of the following statements about the characteristics of the Chinese national bourgeoisie is incorrect?
    A. The Chinese national bourgeoisie is a bourgeois class that colludes with foreign imperialism.
    B. The Chinese national bourgeoisie belongs to the reformist class.
    C. The Chinese national bourgeoisie has a certain degree of class character and revolutionary qualities.
    D. The Chinese national bourgeoisie has a certain degree of class character and lacks revolutionary qualities.
    Answer:
    
    B
    
    The safety risk level for the day shift is determined by the ___ based on the intensity of the shift work
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s here for all of us. AI is not just about cutting-edge technologies like machine learning, deep learning, and neural networks; it’s also about personalizing the technology and its applications to solve the biggest challenges facing humanity. Here’s a look at some of the most promising areas for AI and how they’re transforming industries and our lives.
    The future of AI is here, and it’s here for all of us. AI is not just about cutting-edge technologies like machine learning, deep learning, and neural networks; it’s also about personalizing the technology and its applications to solve the biggest challenges facing humanity. From healthcare


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Blanche" and the "City of Light". It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, art, and culture, and is a major tourist destination. It is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its cuisine, including its famous French fries and its traditional French wine. Paris is a vibrant and dynamic city with a rich cultural and historical heritage. It is a popular tourist destination and a major economic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more capable of performing tasks that were previously done by humans. This could lead to a significant increase in automation in various industries, from manufacturing to transportation to healthcare.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be a growing concern about its impact on society. This includes issues such as bias in AI algorithms, privacy concerns, and the potential for AI to
    


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
    Generated text:  [Name]. I am a [career] with [years of experience] years of experience in [field of interest]. I have always been [born, raised, or developed] in [location]. I am a [favorite hobby]. My background is in [add any relevant background information here]. I am always [positive or positive] about my work and enjoy [job or hobbies]. I am passionate about [occupation]. I am a [job title]. I believe that [reason why I started this career]. I am [elevator pitch] about myself. Thank you. 
    
    Why is your introduction so neutral? It appears to have a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the French capital, with a population of around 2.7 million. It is the cultural and economic center of France and home to the French Parliament, the Supreme Court, and the National Library. The city is known for its iconic landmarks such as the Eiffel Tower, Notre Dame Cathedral, and Louvre Museum, as well as its gastronomic and cultural scene. Paris is also the birthplace of many famous French artists, writers, and intellectuals, including Michelangelo, Leonardo da Vinci, and Oscar Wilde. The city has a rich history and culture dating back to the medieval era, and it continues to be a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and uncertain, and it will likely continue to evolve in complex ways. Some potential future trends include:
    
    1. Increased AI transparency and accountability: As AI systems become more sophisticated, we may see increased transparency and accountability for their decisions. This could include public access to code, data, and training sets used to train AI models.
    
    2. AI-powered personal assistants: We may see the development of AI-powered personal assistants that can help us perform tasks and tasks that require intelligence and discretion. For example, an AI-powered chatbot could help us manage our finances, schedule appointments, and even drive cars.
    
    3. AI-driven healthcare: AI could


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

    Job

     Title

    ]

     who

     is

     passionate

     about

     [

    Why

     I

     Love

     My

     Profession

    ].

     I

     have

     always

     been

     motivated

     by

     [

    Mot

    ivation

    ],

     and

     my

     experience

     has

     allowed

     me

     to

     gain

     valuable

     [

    Skill

    ].

     I

    'm

     eager

     to

     learn

     more

     and

     expand

     my

     knowledge

     base

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     contribute

     to

     the

     world

    .

     I

    'm

     excited

     to

     work

     with

     you

     and

     contribute

     to

     the

     success

     of

     our

     team

    .

     How

     can

     I

     become

     a

     better

     employee

     at

     [

    Company

     Name

    ]?

     Let

     me

     know

    !

     [

    Name

    ]

     [

    Job

     Title

    ]


    This

     template

     looks

     great

    !

     Can

     you

     provide

     some

     more

     information

     on

     how

     I

     can

     improve

     my

     knowledge

     base

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     Dame

     Cathedral

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Additionally

    ,

     Paris

     is

     home

     to

     many

     notable

     museums

    ,

     including

     the

     Lou

    vre

     and

     Mus

    ée

     Rod

    in

    .

     The

     city

    's

     cultural

     and

     artistic

     scene

     is

     also

     thriving

    ,

     with

     many

     famous

     artists

     and

     artists

    '

     residences

     located

     there

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     It

    's

     a

     city

     of

     contrasts

    ,

     with

     its

     rich

     history

     and

     modern

     culture

    ,

     making

     it

     a

     fascinating

     and

     dynamic

     place

     to

     explore

    .

     It

    's

     important

     to

     note

     that

     while

     Paris

     is

     a

     major

     city

     in

     France

    ,

     it

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     promises

     to

     bring

     significant

     changes

     in

     various

     aspects

     of

     our

     lives

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     we

     can

     expect

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     automation

     and

     artificial

     general

     intelligence

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     increasing

     automation

     of

     tasks

     and

     the

     development

     of

     AI

     that

     can

     perform

     tasks

     that

     require

     human

    -like

     intelligence

    .

     This

     could

     lead

     to

     the

     creation

     of

     more

     efficient

     and

     productive

     systems

    ,

     as

     well

     as

     the

     creation

     of

     new

     applications

     that

     require

     greater

     autonomy

     and

     decision

    -making

    .
    


    2

    .

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

     improve

     the

     accuracy

     and

     speed

     of

     diagnoses

     and

     treatment

     plans

    .

     In

     the

     future

    ,

     we

     may

     see

     even

    



```python
llm.shutdown()
```
