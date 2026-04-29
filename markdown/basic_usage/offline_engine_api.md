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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.95it/s]


    2026-04-29 00:30:36,953 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 00:30:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=2816):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:05<00:09,  4.77it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:09,  4.77it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:09,  4.77it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:09,  4.77it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:05<00:09,  4.77it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]

    Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 15.06it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 21.83it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:02, 19.12it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.37it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.37it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.37it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.37it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=768 avail_mem=137.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  50%|█████     | 29/58 [00:00<00:00, 40.06it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 40.06it/s]

    Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 40.06it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 40.06it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 40.06it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 40.06it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.18it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.18it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.18it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.18it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.18it/s]

    Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.94it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.94it/s]

    Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.78it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 38.96it/s]


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
    Generated text:  Stephanie. I have no idea what I’m doing, I’m a full time student in college. My major is Psychology and I am enrolled in a program to do research. I am taking a course on the effects of social media on children. The teacher is very nice and helps me learn, but I am not learning much. What should I do?
    
    There are a number of things that you could do to become successful in your studies, but you could also do a number of things to improve your learning experience. For example, you could try to take notes during class or review notes after class. You could also try taking notes while you
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. What can be inferred about the man?  A. not enough information  B. He is not interested in running for office.  C. He is not a good candidate.  D. He is well aware of the importance of being a good candidate.  E. He is inexperienced in running for office.  Given a list of categories:  A. all animals B. famous people C. native plants D. competition E. organs E. organs The best category to fill in the blank is E. organs. 
    Answer the above question by determining which category best fits the given information. Based on
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    A
    
    Which of the following options is the capital of France?
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    A
    
    The capital of Japan is ____
    A. Tokyo
    B. Osaka
    C. Kyoto
    D. Yokohama
    Answer:
    A
    
    What is the capital of the country with the highest natural population growth rate in the world?
    A. Japan
    B. United States
    C. China
    D. Brazil
    Answer:
    A
    
    What is the capital of
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. But the future of AI with a compelling one is more clear. The future of AI is being defined by the needs of the present.
    
    The need for AI
    
    Today, artificial intelligence is becoming a critical enabler of technology and innovation. More and more companies are beginning to adopt AI for better productivity and better customer experience.
    
    The development of AI has led to a significant reduction in human labor, resulting in productivity gains. More and more companies have started to adopt AI for better collaboration and communication.
    
    AI has been defined as the ability to perform tasks that require intelligence, by imitating human intelligence. It is the technology that enables computers and


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] with [Number of Wheels] wheels. I'm [Favorite Color] and [Favorite Food]. I'm [Favorite Book] and [Favorite Movie]. I'm [Favorite Sport]. I'm [Favorite Music]. I'm [Favorite Movie]. I'm [Favorite Book]. I'm [Favorite Movie]. I'm [Favorite Book]. I'm [Favorite Movie]. I'm [Favorite Book]. I'm [Favorite Movie]. I'm [Favorite Book]. I'm [Favorite Movie]. I'm [Favorite Book]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and historical center with a rich history dating back to ancient times and a modern city with a diverse population. It is a major financial center and a major tourist destination. The city is known for its cuisine, fashion, and art, and is home to many famous museums, theaters, and other cultural institutions. Paris is a vibrant and dynamic city that continues to be a major center of global
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, including in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management and fraud detection. As
    


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
    Generated text:  [Your Name]. I'm an 18-year-old art student who enjoys exploring new and diverse art forms, including painting, sculpture, and performance art. I'm a natural visual storyteller and enjoy capturing the essence of the subject through my art. I'm also passionate about sharing my artistic creations with others and have been participating in various art exhibitions and art festivals. I enjoy meeting new people, learning from them, and exploring different cultures. I'm excited to continue growing and learning in my field of art. 
    
    [Your Name] [Your Position] - Artistic Journalist
    
    This is a great self-introduction! Can you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known as the "City of Light" for its vibrant cultural life, and is a UNESCO World Heritage Site. The city is located on the Seine River, overlooking the River Seine, and is home to numerous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its fashion industry, art scene, and delicious cuisine. The city is a popular tourist destination, drawing millions of visitors each year. The capital of France is located in the center of the country, with its population concentrated in the north and central regions. It is a major transportation hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and transformative, and it is set to shape nearly every aspect of our lives. Here are some of the possible future trends in artificial intelligence:
    
    1. Increased Personalization: AI will continue to improve our ability to understand and respond to individuals' needs, preferences, and behaviors. This will lead to more personalized experiences and better service delivery.
    
    2. Autonomous Vehicles: AI will become more advanced and integrated into our daily lives, leading to safer, more efficient, and more convenient transportation.
    
    3. Smart Cities: AI will be used to improve the efficiency and sustainability of cities, from energy management to traffic control to public services.
    
    4. Medical


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

     fictional

     character

    's

     name

    ].

     I

    'm

     a

     [

    insert

     fictional

     character

    's

     role

     or

     profession

    ].

     I

    'm

     a

     [

    insert

     fictional

     character

    's

     age

    ].

     I

    'm

     [

    insert

     fictional

     character

    's

     appearance

     or

     characteristic

    ].

     I

    'm

     [

    insert

     fictional

     character

    's

     personality

     trait

     or

     unique

     personality

    ].

     I

    'm

     [

    insert

     fictional

     character

    's

     hobbies

    ,

     interests

    ,

     or

     passions

    ].

     I

    'm

     [

    insert

     fictional

     character

    's

     favorite

     food

     or

     drink

    ].

     I

    'm

     [

    insert

     fictional

     character

    's

     profession

     or

     occupation

    ].

     I

    'm

     [

    insert

     fictional

     character

    's

     profession

     or

     occupation

    ].

     I

    'm

     [

    insert

     fictional

     character

    's

     profession

     or

     occupation

    ].

     I

    'm

     [

    insert

     fictional

     character

    's

     profession

     or

     occupation

    ].

     I

    'm

     [

    insert

     fictional

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     

    3

    rd

    -largest

     in

     the

     European

     Union

    ,

     with

     an

     estimated

     population

     of

     over

     

    1

    1

     million

     people

     as

     of

     

    2

    0

    2

    1

    .

     The

     city

     is

     home

     to

     many

     iconic

     landmarks

    ,

     including

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

    -D

    ame

     Cathedral

    ,

     and

     the

     Mus

    ée

     d

    ’

    Or

    say

    .

     Paris

     is

     also

     a

     major

     hub

     for

     cultural

    ,

     economic

    ,

     and

     political

     activity

    ,

     making

     it

     a

     major

     center

     for

     the

     arts

    ,

     education

    ,

     and

     international

     diplomacy

    .

     It

     is

     considered

     one

     of

     the

     world

    's

     top

     cities

     for

     innovation

     and

     creativity

    .

     As

     a

     major

     urban

     center

    ,

     Paris

     plays

     a

     significant

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     number

     of

     potential

     trends

     that

     could

     shape

     the

     direction

     of

     development

     in

     the

     field

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

     automation

    :

     AI

     will

     become

     even

     more

     sophisticated

     and

     integrated

     into

     everyday

     life

    ,

     with

     automation

     becoming

     more

     prevalent

    .

     This

     could

     lead

     to

     increased

     efficiency

    ,

     productivity

    ,

     and

     quality

     of

     life

    .
    


    2

    .

     Enhanced

     ethical

     concerns

    :

     As

     AI

     continues

     to

     advance

    ,

     there

     will

     be

     increased

     ethical

     concerns

     about

     its

     impact

     on

     society

    .

     There

     will

     likely

     be

     debates

     about

     the

     use

     of

     AI

     in

     healthcare

    ,

     education

    ,

     and

     other

     areas

    ,

     as

     well

     as

     concerns

     about

     the

     potential

     misuse

     or

     abuse

     of

     AI

    .
    


    3

    .

     AI

     will

     continue

    



```python
llm.shutdown()
```
