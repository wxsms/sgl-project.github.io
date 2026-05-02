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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.73it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.72it/s]


    2026-05-02 23:18:18,933 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 23:18:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:03,  4.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.51it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]

    Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=512 avail_mem=74.94 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=512 avail_mem=74.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=480 avail_mem=74.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=448 avail_mem=74.93 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=416 avail_mem=67.51 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=384 avail_mem=61.68 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]

    Capturing num tokens (num_tokens=352 avail_mem=61.68 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=352 avail_mem=61.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=320 avail_mem=61.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=288 avail_mem=61.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=256 avail_mem=61.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=240 avail_mem=61.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=224 avail_mem=61.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.06it/s]Capturing num tokens (num_tokens=224 avail_mem=61.66 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.41it/s]Capturing num tokens (num_tokens=208 avail_mem=61.66 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.41it/s]Capturing num tokens (num_tokens=192 avail_mem=61.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=176 avail_mem=61.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=160 avail_mem=61.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.41it/s]

    Capturing num tokens (num_tokens=144 avail_mem=61.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=144 avail_mem=61.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=128 avail_mem=61.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=112 avail_mem=61.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=96 avail_mem=61.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.33it/s] Capturing num tokens (num_tokens=80 avail_mem=61.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=64 avail_mem=61.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=64 avail_mem=61.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=48 avail_mem=61.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=32 avail_mem=61.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=28 avail_mem=61.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=24 avail_mem=61.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.05it/s]

    Capturing num tokens (num_tokens=20 avail_mem=61.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=20 avail_mem=61.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=16 avail_mem=61.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=12 avail_mem=61.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=8 avail_mem=61.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.33it/s] Capturing num tokens (num_tokens=4 avail_mem=61.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=4 avail_mem=61.60 GB): 100%|██████████| 58/58 [00:01<00:00, 42.04it/s]


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
    Generated text:  Alex and I'm 14 years old. I'm very talented in sports and I love to run and play soccer. I want to join the army when I grow up. My friends think I'm crazy and they think I should go to university instead. I'm also interested in politics. I want to study something related to it. I know about basic politics and I'm interested in its language and structure. I'm also interested in using the world to make a difference. What do you think about this? What do you think about this? I hope you could help me decide.
    Dear Alex,
    Congratulations on your passion for sports and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. If the president were to die, who would fill that office? Is there anyone in the world with the ability to become the president of the United States?
    
    The answer is that the president of the United States would not have to have any special abilities or qualifications to fill the office. In fact, the current president has had to work extremely hard and put in many years of effort to become the president of the United States.
    
    The president is elected by the people of the United States and is the leader of the country. As such, they are responsible for making decisions on behalf of the nation and ensuring that the country is
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A. Paris B. Frankfurt C. London D. Frankfurt
    Answer:
    
    A
    
    The capital of France is ______.
    A. Paris
    B. Frankfurt
    C. London
    D. Frankfurt
    Answer:
    
    A
    
    What is the capital of France?
    A. Paris
    B. Frankfurt
    C. London
    D. Frankfurt
    Answer:
    
    A
    
    Which of the following cities is the capital of France?
    A. Paris
    B. Frankfurt
    C. London
    D. Frankfurt
    Answer:
    
    A
    
    In a company's organizational structure, which entity's department is typically responsible for day-to-day operations?
    A. Project Management
    ===============================
    Prompt: The future of AI is
    Generated text:  so revolutionary that some people think they can predict the future of AI with the help of quantum computers and neural networks. However, it is necessary to understand that quantum computers and neural networks are all computer algorithms, so they cannot predict the future of AI in the same way. AI will continue to evolve, but we should not be overly optimistic about the future. What does this imply?
    A. The future of AI is uncertain
    B. AI will never evolve
    C. We should be overly optimistic about the future of AI
    D. The future of AI will be uncertain
    Answer:
    
    A
    
    There is a new study about a rare disease


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


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The city is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and economic center of France and a major tourist destination. It is home to many world-renowned museums, theaters, and restaurants. The city is also known for its fashion industry, with many famous fashion houses and boutiques. Paris is a vibrant and dynamic city with a rich history and a thriving
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential future trends in AI:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare to transportation. This will lead to increased efficiency, productivity, and cost savings for businesses and individuals alike.
    
    2. Enhanced cognitive capabilities: AI is likely to continue to advance in areas such as natural language processing, machine learning, and deep learning, which will allow AI to perform tasks that were previously thought to be beyond
    


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
    Generated text:  [Your Name] and I am a [insert your profession or background here, e.g. [e.g. Student, Student at Harvard University, YouTuber, etc.]].
    Hello there! My name is [Your Name] and I am a [insert your profession or background here, e.g. [e.g. Student, Student at Harvard University, YouTuber, etc.]]. My passion is [insert what excites you, e.g. storytelling, music, writing, sports, etc.]. And I believe in [insert a quote or statement that reflects your values, e.g. "The future belongs to those
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    What is the answer? Paris is the capital of:
    
    A) Germany
    B) Italy
    C) Russia
    D) France
    
    The answer is D) France. Paris is the capital city of France. 
    
    To explain it simply:
    - Paris, also known as "La République" (The Republic), is the capital of France.
    - It is located in the south of the country, near the Mediterranean Sea.
    - The city has a rich history and culture, known for its iconic Eiffel Tower and famous landmarks like Notre-Dame Cathedral and the Louvre Museum. 
    - Paris is one of the most
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve and change, with several trends expected to emerge and influence the technology in the coming years.
    
    1. Increased focus on ethical AI: As concerns about the potential for AI to be used for harmful purposes increase, there is a growing push for ethical guidelines and regulations to govern the development and use of AI. This may lead to more rigorous testing and validation of AI systems, and a greater emphasis on transparency and accountability in their development and deployment.
    
    2. Better understanding of AI: As AI becomes more integrated into our daily lives, there is a growing need for a deeper understanding of how it works and how it can be used


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

    .

     I

     am

     a

    /an

     __

    ________

    .

     I

     am

     here

     to

     __

    ________

    .

     My

     interests

     are

     __

    ________

    .

     I

     am

     __________________

     (

    your

     favorite

     hobby

     or

     interest

    ).

     I

     look

     forward

     to

     __

    ________

    .
    


    What

     is

     your

     favorite

     thing

     to

     do

     in

     your

     free

     time

    ?

     How

     about

     __

    ________

    ?

     What

     does

     your

     favorite

     hobby

     or

     interest

     help

     you

     achieve

    ?

     What

     are

     your

     goals

     for

     the

     future

    ?

     What

     are

     you

     looking

     forward

     to

    ?

     What

     is

     your

     next

     big

     project

    ?
    


    I

     will

     ask

     you

     a

     few

     questions

     about

     your

     life

     and

     interests

    .

     I

     can

     listen

     to

     you

     talk

     and

     ask

     you

     questions

    .

     I

     hope

     this

     helps

    .
    


    My

     name

     is

     [

    insert

     name

    ].

     I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     and

     it

     is

     the

     largest

     city

     and

     most

     populous

     met

    ropolis

     in

     the

     country

    .

     It

     is

     located

     on

     the

     banks

     of

     the

     River

     Se

    ine

    ,

     in

     the

     heart

     of

     the

     Lo

    ire

     Valley

    ,

     and

     is

     known

     for

     its

     architecture

    ,

     cuisine

    ,

     and

     museums

    .

     Paris

     is

     a

     hub

     for

     culture

    ,

     arts

    ,

     and

     fashion

    ,

     and

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     numerous

     other

     notable

     landmarks

    .

     The

     city

     is

     also

     home

     to

     the

     French

     National

     Library

    ,

     the

     National

     Opera

    ,

     and

     the

     Opera

     House

    ,

     and

     is

     known

     for

     its

     annual

     Bast

    ille

     Day

     celebrations

    ,

     including

     the

     iconic

     par

    ades

     and

     fireworks

     display

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     there

     is

     no

     single

     clear

     path

     for

     growth

    .

     However

    ,

     there

     are

     several

     trends

     that

     are

     likely

     to

     shape

     AI

     in

     the

     years

     ahead

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

     other

     technologies

    :

     AI

     is

     already

     becoming

     more

     integrated

     with

     other

     technologies

     such

     as

     machine

     learning

    ,

     blockchain

    ,

     and

     cloud

     computing

    .

     The

     integration

     of

     AI

     with

     these

     other

     technologies

     is

     likely

     to

     continue

    ,

     further

     advancing

     the

     capabilities

     of

     AI

     systems

    .
    


    2

    .

     Increased

     privacy

     concerns

    :

     With

     the

     increasing

     use

     of

     AI

     in

     various

     industries

    ,

     there

     is

     a

     growing

     concern

     about

     privacy

    .

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     may

     need

     to

     collect

     more

     personal

     data

     to

     function

     effectively

    



```python
llm.shutdown()
```
