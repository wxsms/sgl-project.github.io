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


    2026-05-20 22:44:43,927 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 22:44:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.43it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.95it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.82it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.60 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.59 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.59 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.59 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.59 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.58 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=75.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=960 avail_mem=74.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s] Capturing num tokens (num_tokens=896 avail_mem=74.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=832 avail_mem=74.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=768 avail_mem=74.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=704 avail_mem=74.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=640 avail_mem=74.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=576 avail_mem=74.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=512 avail_mem=74.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=512 avail_mem=74.52 GB):  50%|█████     | 29/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=480 avail_mem=74.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=448 avail_mem=74.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=416 avail_mem=74.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=384 avail_mem=74.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.83it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.83it/s]Capturing num tokens (num_tokens=352 avail_mem=74.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.34it/s]Capturing num tokens (num_tokens=320 avail_mem=74.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.34it/s]Capturing num tokens (num_tokens=288 avail_mem=74.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.34it/s]Capturing num tokens (num_tokens=256 avail_mem=74.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.34it/s]Capturing num tokens (num_tokens=240 avail_mem=74.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.34it/s]Capturing num tokens (num_tokens=224 avail_mem=74.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.34it/s]Capturing num tokens (num_tokens=224 avail_mem=74.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.36it/s]Capturing num tokens (num_tokens=208 avail_mem=74.50 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.36it/s]Capturing num tokens (num_tokens=192 avail_mem=74.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.36it/s]Capturing num tokens (num_tokens=176 avail_mem=74.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.36it/s]Capturing num tokens (num_tokens=160 avail_mem=74.34 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.36it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.33 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.36it/s]Capturing num tokens (num_tokens=144 avail_mem=74.33 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.22it/s]Capturing num tokens (num_tokens=128 avail_mem=74.33 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.22it/s]Capturing num tokens (num_tokens=112 avail_mem=74.33 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.22it/s]Capturing num tokens (num_tokens=96 avail_mem=74.33 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.22it/s] Capturing num tokens (num_tokens=80 avail_mem=74.32 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.22it/s]Capturing num tokens (num_tokens=64 avail_mem=74.32 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.22it/s]Capturing num tokens (num_tokens=64 avail_mem=74.32 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=32 avail_mem=73.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.43it/s]

    Capturing num tokens (num_tokens=20 avail_mem=73.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=20 avail_mem=73.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=8 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.37it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 42.03it/s]


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
    Generated text:  Michael and I am 16 years old. I have a question about my son who is 12 years old. Should I take him to the doctor or go to the school?
    
    I have no idea how much the school's needs are or if they will help him. To me, it seems like the school should be the first stop for all parents to see their child. I have no idea what kind of tests they would do.
    
    Should I ask him what he thinks about the doctor?
    
    What should I do to find out what the best option is?
    
    Multi-choice problem: What are your thoughts on going to the doctor for a
    ===============================
    Prompt: The president of the United States is
    Generated text:  attempting to make a statement to the country. He says he is going to make a speech. When he is speaking, the speaker will not make any gestures. In fact, he will not smile. The speaker will not make any of the other gestures that we use in our everyday speech. He will not even speak, he will just stand, look at the audience and begin speaking. There will be no pauses. He will speak so quickly that the audience will probably have to call out the speakers name. He will then begin to move, using his arms and legs to demonstrate what he has said. The audience will be surprised to see a
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the: 
    
    A. Mediterranean Sea
    
    B. Atlantic Ocean
    
    C. North Sea
    
    D. Indus River
    
    1. The capital of France is not located on any of the five major waterways that flow through the country. The Atlantic Ocean, the Mediterranean Sea, the North Sea, and the Indus River are all major waterways that flow through France.
       
    2. Therefore, the capital of France is most likely located in the Atlantic Ocean.
    
    Thus, the capital of France is most likely located in:
    
    B. Atlantic Ocean
    
    The Atlantic Ocean is the largest body of water in the world and is an inland body
    ===============================
    Prompt: The future of AI is
    Generated text:  looking bright in the tech world. Many AI researchers are looking at artificial intelligence in a new way. They are looking at it from a new perspective, one that is closer to the human mind and uses more data. The new AI is able to learn from examples, which allows it to solve problems that a human cannot. It's also better able to make decisions based on the data that it receives.
    This new AI looks a lot like the human brain. It has many parts, including the cortex, which is the center of the brain. The cortex is made up of neural networks, which are like tiny computers that process information. The human


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] who has always been [Positive Trait]. I'm passionate about [What I Love to Do]. I'm [What I Do For Fun]. I'm [What I Do For Work]. I'm [What I Do For Life]. I'm [What I Do For Life]. I'm [What I Do For Life]. I'm [What I Do For Life]. I'm [What I Do For Life]. I'm [What I Do For Life]. I'm [What I Do For Life]. I'm [What I Do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third largest in the world, with a population of over 2. 5 million people. Paris is a cultural and historical center, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major financial and business center, with many of the world's major financial institutions and companies headquartered there. Paris is a popular tourist destination, with millions of visitors each year. It is also a major center for the arts, with many museums,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use of AI
    


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
    Generated text:  [Name], and I am a [Job Title] with over [number] years of experience in [Related Field]. I bring a unique blend of creativity, problem-solving skills, and a willingness to learn, making me an ideal fit for any team or project. As an [Responsibility/Job], I thrive on pushing boundaries, embracing change, and always striving to improve myself and others. I am a passionate and persistent individual, with a strong sense of responsibility and a positive attitude towards every opportunity and challenge I face. I believe that with hard work and dedication, anyone can achieve their goals, and I am confident in my ability to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic Eiffel Tower, Notre-Dame Cathedral, and numerous museums and cultural institutions. France's capital is Paris. It is the largest and most populous city in the European Union and is the capital of France and the second-most populous city in the world after Beijing. Paris is home to numerous landmarks, including the Louvre Museum and the Notre-Dame Cathedral, and is a UNESCO World Heritage site. The city is also known for its fashion and art scene, with the Paris Museum of Modern Art being one of the largest in the world. Paris is a cultural and economic center that is home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, and there are several trends that are likely to shape it in the coming years. Here are some possible trends:
    
    1. Increased Intelligence: AI is expected to continue growing in intelligence, with more advanced algorithms and models that can analyze and understand more complex information.
    
    2. AI Integration: AI is likely to become more integrated into everyday life, from personal assistants like Siri and Alexa to autonomous vehicles and drones. This will lead to a shift towards a more connected world where AI is a fundamental part of everyday life.
    
    3. AI Ethics and Responsibility: As AI becomes more integrated into society, there will be growing concerns about its impact on


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

     __

    ________

    ,

     and

     I

     am

     a

    /an

     __

    ________

    __

    _.

     I

     came

     here

     from

     a

    /an

     __

    ________

    __

    _.

     I

     have

     been

     attending

     this

     school

     since

     I

     __

    ________

    __.

     I

     love

     __

    ________

    __.

     I

    'm

     a

    /an

     __

    ________

    _.

     I

     enjoy

     __

    ________

    __.

     I

     look

     forward

     to

     __

    ________

    __

    _.

     I

    'm

     optimistic

     and

     optimistic

    .

     I

    'm

     patient

     and

     patient

    .

     I

    'm

     a

    /an

     __

    ________

    _.

     I

    'm

     confident

     and

     confident

    .

     I

    'm

     a

    /an

     __

    ________

    _.

     I

    'm

     __

    ________

    _.

     I

    'm

     ______

    _

    .


    I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    _.

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     beautiful

     architecture

    ,

     charming

     historical

     landmarks

    ,

     and

     vibrant

     cultural

     scene

    .

     It

     is

     a

     city

     that

     has

     been

     an

     important

     cultural

    ,

     religious

    ,

     and

     economic

     hub

     for

     centuries

    .

     Paris

     has

     a

     rich

     history

     that

     includes

     ancient

     Roman

     and

     Greek

     influences

    ,

     as

     well

     as

     the

     influence

     of

     various

     European

     cultures

    .

     It

     is

     also

     home

     to

     some

     of

     the

     world

    ’s

     most

     famous

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .

     In

     addition

     to

     its

     historical

     and

     cultural

     significance

    ,

     Paris

     is

     also

     known

     for

     its

     lively

     nightlife

     and

     rich

     cuisine

    .

     With

     its

     numerous

     museums

    ,

     theaters

    ,

     and

     cultural

     venues

    ,

     Paris

     is

     an

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

     that

     will

     shape

     how

     it

     is

     used

     and

     developed

    .

     Here

     are

     some

     potential

     future

     trends

     that

     could

     impact

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

    :

     As

     AI

     technologies

     continue

     to

     advance

    ,

     we

     are

     likely

     to

     see

     more

     sophisticated

     algorithms

     that

     can

     learn

     and

     adapt

     to

     new

     data

    .

     This

     could

     lead

     to

     new

     breakthrough

    s

     in

     areas

     like

     image

     and

     speech

     recognition

    ,

     natural

     language

     processing

    ,

     and

     robotics

    .
    


    2

    .

     Increased

     reliance

     on

     AI

     in

     healthcare

    :

     With

     the

     increasing

     availability

     of

     large

     datasets

     and

     the

     ability

     to

     analyze

     them

     in

     real

    -time

    ,

     AI

     is

     likely

     to

     play

     a

     more

     prominent

     role

     in

     healthcare

    .

     This

     could

     lead

    



```python
llm.shutdown()
```
