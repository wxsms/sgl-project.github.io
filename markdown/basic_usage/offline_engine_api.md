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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]


    2026-04-27 23:46:31,400 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 23:46:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  3.99it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.01it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]

    Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 31.62it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 31.62it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 31.62it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 31.62it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 31.62it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 31.62it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 31.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   7%|▋         | 4/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   7%|▋         | 4/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):   7%|▋         | 4/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   7%|▋         | 4/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.19it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.77it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.75it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.75it/s]Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.75it/s] Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=640 avail_mem=72.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.19it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.70it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=32 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.63it/s]Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.63it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.63it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 40.57it/s]


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
    Generated text:  Miki. I’m a member of the Cultural Workers' Group. I'm 16 years old and I'm from Japan. I want to be a doctor when I grow up. I have a brother, who is now a university student. We will go to the hospital every week. I'm afraid that I will be too sick to go to school. I also have to clean up my room and feed the dog. What about you, Miki? What kind of job do you have? Are you afraid that you won't be able to go to school if you don't have a job? I think I would like to
    ===============================
    Prompt: The president of the United States is
    Generated text:  in New York. To make a telephone call to his home in Washington D. C., the president must go by plane. There are four possible routes to choose from. The first route will take 8 hours; the second route will take 5 hours; the third route will take 6 hours; and the fourth route will take 3 hours. 
    
    What is the average travel time for a round trip to Washington D. C. by plane? 
    (A) 2.75 hours 
    (B) 2.97 hours 
    (C) 3.15 hours 
    (D) 3.33 hours 
    
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) Marseille
    
    C) Lyon
    
    D) Nice
    The capital of France is:
    
    A) Paris
    
    Paris, also known as "La Grande Paris" (The Great Paris), is the capital city of France. It is located in the south of the country and is the largest city in Europe. Paris is known for its medieval architecture, fashion industry, and many of the world's major museums and theaters. The city is also famous for its famous Eiffel Tower, as well as its status as one of the world's most popular tourist destinations. 
    
    Among the options provided, Paris is the correct answer
    ===============================
    Prompt: The future of AI is
    Generated text:  very uncertain and the predictions for the future of AI are not very reliable. We can only hope that the future of AI is good, and that we can use it to make the world better and more equitable.
    
    As a teenager, I remember the excitement of a new technology and how it would change the world. I was particularly excited about the potential of artificial intelligence (AI) and how it could revolutionize the way we live our lives. With the rise of AI, we can make the world more efficient, personalized, and fairer.
    
    The potential of AI is vast and it has the power to solve complex problems and make the world better


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love to Do], and I'm always looking for ways to [What I Want to Improve]. I'm always eager to learn and grow, and I'm always looking for new challenges and opportunities. I'm a [Personality] who is [What I Like to Do]. I'm always ready to help others, and I'm always willing to learn from others. I'm a [Personality] who is [What I Like
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic and cultural center with a rich history dating back to the Roman Empire. It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination and is home to many world-renowned museums, art galleries
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption.
    
    2. AI becoming more autonomous: As AI becomes more integrated into our daily lives, we can expect to see more autonomous vehicles and robots that can perform tasks without human intervention.
    
    3. AI becoming more sophisticated: As AI continues to advance,
    


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
    Generated text:  [Your Name]. I'm a [age] year old girl who loves to explore the world and have a passion for learning new things. I'm an active person and enjoy trying new things. I'm always looking for new challenges and trying to learn as much as I can. What excites me the most is the chance to make new friends and share my love of learning with them. I'm always striving to improve myself and my knowledge of the world. What brings you to this position? I feel that I can help others in my community and offer a voice for those who might not have one. Thank you for taking the time to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. The city is known for its rich history, cultural attractions, and romantic ambiance, particularly the Eiffel Tower and the Louvre Museum. It is a popular tourist destination and the seat of government for the nation. 
    
    Please format the response in an Excel table:
    |City Name| Capital City|
    |---|---|
    |Paris| |
    |---|---|
    |City Name| |
    |---|---|
    |Country| France|
    |Latitude| |
    |Longitude| |
    |Population| |
    |GDP| |
    |Population Density| | Here is a concise factual statement about France's capital city in French:
    
    "
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising and varied, with many potential developments and trends shaping its evolution. Here are some of the most likely trends we can expect to see in the coming years:
    
    1. Increased automation and AI integration: As automation continues to become more widespread, AI will become more integrated into everyday life. We can expect to see more automation in industries like manufacturing, healthcare, and transportation, as well as more AI-driven solutions in areas like finance and energy.
    
    2. AI-powered healthcare advancements: With the growing need for personalized medicine and treatment, we can expect to see significant advancements in AI-powered healthcare. For example, AI algorithms could be used to analyze


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

    profession

    ].

     I

    've

     been

     coding

     for

     [

    Company

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     In

     my

     spare

     time

    ,

     I

     enjoy

     [

    爱好

    或

    兴趣

    ]

     and

     travel

    .

     Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    .


    As

     an

     AI

     language

     model

    ,

     I

     do

     not

     have

     a

     physical

     presence

     or

     profession

    .

     However

    ,

     I

     can

     be

     programmed

     to

     assist

     in

     various

     tasks

     and

     answer

     questions

     to

     the

     best

     of

     my

     abilities

    .

     Please

     let

     me

     know

     if

     you

     have

     any

     specific

     questions

     or

     needs

    .

     I

    'm

     here

     to

     help

    !

     


    Let

     me

     know

     if

     you

     would

     like

     me

     to

     interact

     with

     you

     using

     a

     specific

     language

     or

     model

    .

     How

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     iconic

     city

     known

     for

     its

     medieval

     architecture

    ,

     vibrant

     neighborhoods

    ,

     and

     rich

     history

    .

     The

     city

     is

     home

     to

     many

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     for

     its

     annual

     festivals

     and

     cultural

     events

    ,

     including

     the

     M

    esse

     de

     la

     Culture

     and

     the

     Mont

    mart

    re

     Art

     District

    .

     The

     city

    's

     climate

     is

     temper

    ate

     with

     mild

     winters

     and

     hot

     summers

    ,

     which

     attracts

     many

     tourists

     from

     around

     the

     world

    .

     Overall

    ,

     Paris

     is

     a

     bustling

    ,

     vibrant

     city

     with

     a

     rich

     history

     and

     a

     unique

     culture

     that

     draws

     visitors

     from

     all

     over

     the

     world

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     key

     trends

     that

     will

     shape

     the

     way

     we

     interact

     with

     technology

     and

     the

     world

     around

     us

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

    :
    


    1

    .

     Increased

     automation

    :

     AI

     is

     becoming

     more

     integrated

     into

     various

     industries

    ,

     including

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

    .

     As

     this

     technology

     becomes

     more

     widespread

    ,

     it

     is

     likely

     that

     we

     will

     see

     even

     more

     automation

    ,

     leading

     to

     increased

     efficiency

     and

     productivity

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     cameras

    ,

     and

     voice

     recognition

    .

     We

     may

     see

     further

     integration

     in

     the

     coming

     years

     as

     these

     technologies

     become

     more

     sophisticated

    .
    


    3

    .

     Increased

     AI

     research

    



```python
llm.shutdown()
```
