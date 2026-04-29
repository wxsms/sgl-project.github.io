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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.17it/s]


    2026-04-29 19:27:44,367 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 19:27:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.10it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.21it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 22.51it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.06it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.06it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.06it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.06it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.06it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.06it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.06it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 19.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 19.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 19.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 19.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 19.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.52it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.73it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.73it/s] Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  43%|████▎     | 25/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  43%|████▎     | 25/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  43%|████▎     | 25/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.39it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.39it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.39it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.39it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.39it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.39it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.39it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.68it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.68it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.68it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.68it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.12it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.54it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.54it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.54it/s]Capturing num tokens (num_tokens=32 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.54it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.54it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.54it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.39it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 41.25it/s]


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
    Generated text:  Mary. I am a girl. My parents are very nice to me. I like to play tennis, and I like to play the piano. I can play the piano well. I can play tennis very well, too. One day, I got an ice cream and a chocolate. What a wonderful day! I can't wait to play the piano and tennis. My father and my mother are both doctors. They are very busy. They work very hard. And they always try to help others. I love my parents and I love my father, my mother, and the doctors. How do they do it? They are kind and they
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, holding the most powerful position in the United States government. The president is also the head of the executive branch of the government. The president represents the United States in the United Nations. The president is elected by the people in an election.
    What is the highest level of government in the United States? The highest level of government in the United States is the federal government. The federal government is made up of three branches: the executive, the legislative, and the judicial. The president is the head of the executive branch and is elected by the people to represent the United States in the federal government. The legislative branch is made
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Rome
    C. London
    D. Berlin
    Answer:
    A
    
    When using the 'refine' function to filter data, which of the following statements is true?
    A. The filter criteria can be a single field name, and the field name can be set as a wildcard;
    B. The filter criteria can be a single field name, and the field name can only be set as a regular expression;
    C. The filter criteria can be a multiple field names, and each field name can be a regular expression;
    D. The filter criteria can be a single field name, but the field name
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, with the proliferation of apps on mobile devices. From the one-minute "success" recommendation on Amazon to the 15-second "quick fix" Twitter message to the 15-second video featuring a 24-hour fix, it is easy to see the transformative power of AI in our lives. But the power of AI is not limited to our mobile devices and email. A 2018 study by the Pew Research Center found that 28% of American adults in one-way conversations (sending messages to send messages) and 27% in two-way conversations (sending messages to reply to messages) were


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] [Vehicle Name]. I'm [Favorite Hobby] and I enjoy [Favorite Activity]. I'm [Favorite Food] and I love [Favorite Movie]. I'm [Favorite Book] and I read [Number of Books] books a year. I'm [Favorite Sport] and I play [Favorite Sport]. I'm [Favorite Music] and I listen to [Favorite Album] and [Favorite Song]. I'm [Favorite Movie] and I watch [Number of Movies] movies a year. I'm [Favorite Book]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. The city is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its fashion industry, with many famous fashion houses and designers operating in the area. Paris is a popular tourist destination and a major economic center in France. It is the seat of the French government and the headquarters of the European Union. The city is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as
    


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
    Generated text:  [First Name] and I am an [Primary Role or Career] with [Number of Years in Industry] years of experience in the [Industry]. I am known for my [Unique Trait or Skill] and my ability to [Benefit to the Company or Industry]. I look forward to the chance to get to know you and learn more about your career goals. How would you like to meet you? [First Name] | [Phone Number] | [Email Address] | [LinkedIn Profile URL] | [Company Name] | [Company Website URL] [First Name] | [Phone Number] | [Email Address] | [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre Dame Cathedral, and the Palace of Versailles. The city is also renowned for its vibrant French culture, including its cuisine, music, and dance. It's a cultural and economic hub, with a population of over 6.6 million people. Paris has played a pivotal role in shaping French identity and politics throughout history. Today, it remains a cultural and economic center of France and a major tourist destination. The city is also famous for its historical architecture, such as the Louvre and the Notre Dame Cathedral. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising. Here are some potential future trends:
    
    1. Deep Learning: Deep learning is one of the most popular AI algorithms. It is capable of learning from very large datasets and extracting patterns and features that are difficult for humans to discern. It is already being used in a wide range of applications, such as image and speech recognition, natural language processing, and autonomous driving.
    
    2. Automation: AI is becoming more and more advanced, and it is already being used to automate repetitive tasks. This trend is expected to continue as more and more industries adopt AI-powered automation solutions.
    
    3. Autonomous Vehicles: Autonomous vehicles are one of the most exciting


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

     [

    Your

     Age

    ]

     years

     old

    .

     I

     grew

     up

     in

     [

    Your

     Place

     of

     Origin

    ],

     where

     I

     always

     had

     a

     thirst

     for

     learning

     and

     a

     love

     for

     adventure

    .

     I

     always

     wanted

     to

     explore

     new

     places

     and

     meet

     new

     people

    .

     My

     favorite

     hobby

     is

     [

    Your

     Hobby

    /

    Interest

    ].

     I

     am

     passionate

     about

     [

    Your

     Personal

     Interest

    /

    Interest

    ],

     and

     I

     believe

     that

     everyone

     should

     have

     the

     opportunity

     to

     learn

     and

     grow

    .

     How

     can

     I

     help

     you

    ?

     As

     a

     beginner

     in

     the

     field

     of

     [

    Your

     Field

     of

     Interest

    ],

     I

     am

     always

     eager

     to

     learn

     and

     share

     my

     knowledge

     with

     anyone

     who

     has

     the

     desire

     to

     learn

     too

    .

     I

     am

     always

     looking

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     vibrant

     culture

    ,

     stunning

     architecture

    ,

     and

     rich

     history

    .

     Its

     famous

     landmarks

     include

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

     Paris

     is

     also

     home

     to

     numerous

     museums

     and

     galleries

    ,

     including

     the

     Mus

    ée

     de

     l

    '

    Or

    anger

    ie

     and

     the

     Mus

    ée

     national

     d

    '

    histoire

     nature

    lle

    .

     It

    's

     a

     city

     that

    's

     steep

    ed

     in

     French

     culture

     and

     cuisine

    ,

     and

     it

    's

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     the

     French

     capital

    .

      


    Answer

     this

     question

    :

     Which

     of

     the

     following

     cities

     is

     known

     for

     its

     architectural

     style

     and

     cultural

     significance

    ?

     Paris

    ,

     Edinburgh

    ,

     Paris

    ,

     Berlin

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     marked

     by

     increased

     collaboration

     between

     humans

     and

     machines

    ,

     as

     well

     as

     continued

     development

     of

     new

     technologies

     and

     algorithms

    .

     AI

     is

     expected

     to

     continue

     evolving

     at

     an

     accelerated

     pace

    ,

     with

     new

     applications

     and

     capabilities

     emerging

     in

     many

     fields

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     AI

     integration

     with

     traditional

     industries

    :

     AI

     is

     already

     being

     integrated

     into

     various

     industries

    ,

     such

     as

     healthcare

    ,

     manufacturing

    ,

     and

     transportation

    ,

     but

     more

     integration

     is

     expected

     as

     AI

     becomes

     more

     prevalent

     and

     useful

    .
    


    2

    .

     AI

    -driven

     personalized

     medicine

    :

     AI

     is

     being

     used

     to

     create

     more

     accurate

     and

     personalized

     diagnoses

     and

     treatments

     for

     diseases

    ,

     and

     AI

    -powered

     health

     monitoring

     devices

     are

     becoming

     more

     widely

     available

    .
    


    3

    



```python
llm.shutdown()
```
