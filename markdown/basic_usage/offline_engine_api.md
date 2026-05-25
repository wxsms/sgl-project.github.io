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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.48it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.48it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.48it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.48it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:05<00:01, 20.59it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.73it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 39.83it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 39.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 16.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 16.47it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 16.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.42 GB):   7%|▋         | 4/58 [00:00<00:04, 11.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.42 GB):   7%|▋         | 4/58 [00:00<00:04, 11.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:04, 11.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   7%|▋         | 4/58 [00:00<00:04, 11.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.60it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.51it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.50it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.50it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.50it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.50it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.50it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:01, 30.03it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:01, 30.03it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:01, 30.03it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.03it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.03it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:01, 26.95it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:01<00:01, 26.95it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:01<00:01, 26.95it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:01, 26.95it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:01, 26.95it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.98it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.98it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.98it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.98it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.98it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.57it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.57it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.57it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.57it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.57it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 31.11it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.07it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.20it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.20it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.20it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.20it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.20it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.34it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 29.36it/s]


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
    Generated text:  Maha. My name is Maha. And so, my name is Maha. As you can see, I am speaking from a single point of view. And so I am trying to portray my view of things, but I am only speaking for myself. I'm not trying to imply that anyone else has the same view. I don't think that we are all on the same page. And so I would encourage people to work on their own views, and it's okay to disagree. But that doesn't mean that I don't have a different view and that I'm trying to impress on you that I have a different view
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very busy person. He gets up very early in the morning, as he is supposed to be in Washington D. C. for the next several days. He usually starts his work in Washington D. C. at about 7 o'clock in the morning. His secretary (秘书) goes to work at 8: 30 o'clock in the morning. He stays at his desk for the rest of the day. When it is time for dinner, he walks home at 5:00 p. m. to the hotel. On the other hand, the president of the European Commission (欧洲委员会) spends much more
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The symbol of France is the Eiffel Tower. The capital of France is the capital of France. The symbol of France is the Eiffel Tower. The capital of France is Paris.
    
    What is the problem with the given statement?
    
    A) The sentence is grammatically correct.
    B) The sentence is clear and complete.
    C) The sentence is grammatically correct and clear and complete.
    D) The sentence is grammatically correct, but not clear and complete.
    
    B) The sentence is clear and complete. 
    
    The statement "The capital of France is Paris. The symbol of France is the Eiffel Tower. The
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but the vast number of experts and the numerous resources available for the professional to provide insightful advice on AI are exciting. As a web developer, there are many benefits to knowing how to deal with AI. If you are a web developer, you should know that the future of AI is uncertain, but the vast number of experts and the numerous resources available for the professional to provide insightful advice on AI are exciting.
    On the other hand, some people believe that the future of AI is bleak. This is a common misconception. The future of AI is uncertain, but the vast number of experts and the numerous resources available for the professional to provide


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [majority] of [interests or hobbies]. I'm always looking for new experiences and learning new things, and I'm always eager to share my knowledge with others. What's your favorite hobby or activity? I love [mention a hobby or activity]. What's your favorite book or movie? I love [mention a book or movie]. What's your favorite place to go? I love
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Paris is also known for its fashion industry, with many famous designers and fashion houses located in the city. The city is also home to many other notable landmarks, including the Champs-Élysées, the Eiffel Tower, and the Arc de
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, reduce costs, and improve quality. As AI technology continues to improve, we can expect to see even more widespread use of AI in manufacturing.
    
    3. AI in finance
    


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
    Generated text:  [insert name], and I'm a professional [insert professional title] with a background in [insert relevant field or industry]. I'm a [insert age or current age] year-old who majored in [insert major] at [insert university or college]. My personal background includes [insert relevant experience or education], and I've been [insert relevant skills or qualifications] in my field. I'm a [insert neutral superlative or adjective] professional with [insert relevant experience or education] in [insert relevant field or industry]. I'm a [insert neutral superlative or adjective] professional with [insert relevant experience or education]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a city with a rich history, famous for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its beautiful architecture, including the Notre-Dame Cathedral and the Palais de Justice. French cuisine is also famous, with many regional dishes to explore. Overall, Paris is a vibrant and diverse city with a rich cultural heritage. According to the passage, Paris is a city with a rich history and famous landmarks. It is also a city with a diverse food scene, with a variety of regional dishes to explore. The Eiffel Tower, Notre-D
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by increasing integration with other technologies, including broader adoption of machine learning, deep learning, and quantum computing. It may also see greater focus on developing new AI technologies and applications, such as robotics and autonomous vehicles. AI may also become more integrated with human decision-making, as more complex AI systems learn from and adapt to human behavior and preferences. Finally, the development of AI could also focus on improving the ethical and societal implications of AI, as well as developing AI safety protocols and best practices for responsible use. Overall, the future of AI is likely to be characterized by continued innovation and growth in its capabilities and applications. It


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

     am

     [

    Age

    ],

     [

    Occup

    ation

    ]

     ([

    Occup

    ation

     describes

     the

     character

    's

     current

     job

     or

     role

    ]).

     I

     have

     always

     been

     passionate

     about

     [

    Career

     Goal

    ],

     and

     I

     believe

     that

     my

     background

     and

     skills

     in

     [

    Professional

     Experience

    ]

     have

     equipped

     me

     to

     make

     a

     significant

     impact

     in

     [

    Industry

    ].

     I

     am

     always

     eager

     to

     learn

     and

     grow

    ,

     always

     looking

     for

     ways

     to

     improve

     myself

     and

     achieve

     my

     goals

    .

     I

     am

     a

     reliable

    ,

     dependable

    ,

     and

     hard

    working

     individual

     who

     is

     always

     ready

     to

     help

     others

    .

     I

     am

     passionate

     about

     [

    Char

    ity

    /

    Environmental

    /S

    ocial

     Good

    ],

     and

     I

     am

     dedicated

     to

     making

     a

     positive

     impact

     on

     the

     world

    .

     Thank

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     about

     the

     capital

     city

     of

     France

     is

    :

     Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    .

     It

     is

     located

     in

     the

     south

    -central

     part

     of

     the

     country

     and

     is

     the

     capital

     of

     France

    .

     The

     city

     covers

     an

     area

     of

     

    6

    0

    2

     square

     kilometers

     and

     has

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

     known

     for

     its

     historic

     architecture

    ,

     cultural

     heritage

    ,

     and

     modern

     landmarks

    ,

     including

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

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     a

     global

     cultural

     and

     business

     center

    ,

     with

     a

     large

     presence

     of

     French

     restaurants

    ,

     bars

    ,

     and

     museums

    .

     Paris

     is

     known

     as

     "

    la

     ville

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     and

     there

     are

     several

     areas

     where

     we

     can

     expect

     significant

     advancements

     and

     trends

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

     Improved

     Explain

    ability

     and

     Transparency

    :

     With

     the

     emergence

     of

     machine

     learning

     algorithms

    ,

     we

     are

     seeing

     improvements

     in

     how

     AI

     systems

     can

     be

     explained

     and

     understood

     by

     humans

    .

     This

     can

     lead

     to

     more

     effective

     AI

     systems

     that

     can

     be

     used

     in

     areas

     such

     as

     fraud

     detection

     and

     cybersecurity

    .
    


    2

    .

     Better

     Security

     and

     Privacy

    :

     As

     AI

     systems

     become

     more

     complex

     and

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increased

     concerns

     about

     security

     and

     privacy

    .

     Governments

     and

     organizations

     will

     need

     to

     work

     to

     develop

     new

     technologies

     and

     regulations

     to

     protect

     users

    '

     privacy

     and

     ensure

    



```python
llm.shutdown()
```
