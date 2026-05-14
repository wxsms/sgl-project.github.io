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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.83it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.83it/s]


    2026-05-14 05:25:25,354 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 05:25:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:49,  5.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:49,  5.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:49,  5.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:49,  5.07s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:49,  5.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  3.86it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.11it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.18 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.18 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.18 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.18 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.18 GB):   9%|▊         | 5/58 [00:00<00:02, 21.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.16 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.16 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.12 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.07it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.10 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.10 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.10 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.10 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.44it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=59.09 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.09 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.73it/s]Capturing num tokens (num_tokens=960 avail_mem=59.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.73it/s] Capturing num tokens (num_tokens=896 avail_mem=59.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.73it/s]Capturing num tokens (num_tokens=896 avail_mem=59.08 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=832 avail_mem=59.08 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=768 avail_mem=59.07 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.54it/s]

    Capturing num tokens (num_tokens=704 avail_mem=59.07 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=640 avail_mem=59.07 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=640 avail_mem=59.07 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.25it/s]Capturing num tokens (num_tokens=576 avail_mem=59.07 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.25it/s]Capturing num tokens (num_tokens=512 avail_mem=59.05 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.25it/s]Capturing num tokens (num_tokens=480 avail_mem=59.07 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.25it/s]Capturing num tokens (num_tokens=448 avail_mem=59.07 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.25it/s]Capturing num tokens (num_tokens=448 avail_mem=59.07 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.27it/s]Capturing num tokens (num_tokens=416 avail_mem=59.06 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.27it/s]Capturing num tokens (num_tokens=384 avail_mem=59.06 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.27it/s]

    Capturing num tokens (num_tokens=352 avail_mem=59.06 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.27it/s]Capturing num tokens (num_tokens=320 avail_mem=59.05 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.27it/s]Capturing num tokens (num_tokens=320 avail_mem=59.05 GB):  60%|██████    | 35/58 [00:01<00:00, 34.28it/s]Capturing num tokens (num_tokens=288 avail_mem=59.05 GB):  60%|██████    | 35/58 [00:01<00:00, 34.28it/s]Capturing num tokens (num_tokens=256 avail_mem=59.05 GB):  60%|██████    | 35/58 [00:01<00:00, 34.28it/s]Capturing num tokens (num_tokens=240 avail_mem=59.04 GB):  60%|██████    | 35/58 [00:01<00:00, 34.28it/s]Capturing num tokens (num_tokens=224 avail_mem=59.04 GB):  60%|██████    | 35/58 [00:01<00:00, 34.28it/s]Capturing num tokens (num_tokens=224 avail_mem=59.04 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=208 avail_mem=59.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=192 avail_mem=59.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=176 avail_mem=59.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.07it/s]

    Capturing num tokens (num_tokens=160 avail_mem=59.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=144 avail_mem=59.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=144 avail_mem=59.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.42it/s]Capturing num tokens (num_tokens=128 avail_mem=59.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.42it/s]Capturing num tokens (num_tokens=112 avail_mem=59.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.42it/s]Capturing num tokens (num_tokens=96 avail_mem=59.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.42it/s] Capturing num tokens (num_tokens=80 avail_mem=59.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.42it/s]Capturing num tokens (num_tokens=64 avail_mem=59.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.42it/s]Capturing num tokens (num_tokens=64 avail_mem=59.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=48 avail_mem=59.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=32 avail_mem=59.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=28 avail_mem=59.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.76it/s]

    Capturing num tokens (num_tokens=24 avail_mem=59.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=20 avail_mem=58.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=20 avail_mem=58.99 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=16 avail_mem=58.99 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=12 avail_mem=58.99 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=8 avail_mem=58.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.31it/s] Capturing num tokens (num_tokens=4 avail_mem=58.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=4 avail_mem=58.98 GB): 100%|██████████| 58/58 [00:01<00:00, 33.80it/s]


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
    Generated text:  Jeff White, I’m a 24 year old grad student in a few different fields. I’m a fan of science, engineering, and technology, and I like to work with others, especially people who have a sense of humor and a passion for science. I think that when we all work together, we can create something amazing. I think that we need to share ideas with each other in order to make a positive impact, and I believe that sharing is the key to success.
    I’m also the co-founder of the Endeavour Green Bank, a company that uses a solar-powered, green energy store, it is entirely reusable and
    ===============================
    Prompt: The president of the United States is
    Generated text:  now trying to reconstruct the United States after the 2001 attacks. In 2001, the president witnessed the events of September 11, 2001, and was deeply affected. This event caused the president to have to reflect on his political philosophy and become more cautious. He decided to substitute "al-Qaeda" as the name for the terrorist group responsible for the attacks. The president also changed his stance on the war on terrorism and changed the name of the federal government. 
    
    If the president's reflection on his political philosophy led to him substituting "al-Qaeda" for the name of the terrorist
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which of the following is NOT one of Paris' famous landmarks? A. Eiffel Tower B. Arc de Triomphe C. Louvre Museum D. Notre-Dame Cathedral E. The Eiffel Tower
    Answer:
    
    D
    
    In the 1840s, the first imperial examination system was established in China, and the term for the exam was called 'simplified imperial examination'.
    A. Correct
    B. Incorrect
    Answer:
    
    A
    
    Which of the following statements about the characteristics of the 'Sui and Tang Dynasties' is incorrect? ____ 
    A. The Sui Dynasty was a
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the people. This is the opinion of the author. Among the following, which one does NOT reflect the author's view on the future of AI?
    A. Artificial Intelligence, like a wild beast, cannot be domesticated.
    B. The goal of AI development is to ensure the safety of humanity.
    C. Artificial intelligence will ultimately solve all problems, making the world more harmonious.
    D. Artificial intelligence is no different from humans in terms of creation, development, and application.
    Answer:
    
    C
    
    Which of the following statements about frictional unemployment is true?
    A. It occurs during periods of economic growth.
    B.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. Let's chat! [Name] [Job Title] [Company Name] [Company Address] [City, State, Zip Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub of the country and is a popular tourist destination. It is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a vibrant and dynamic city that is a must-visit for anyone interested in French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a greater emphasis on privacy and security, with more stringent regulations and controls in place to protect user data and prevent misuse of AI systems.
    
    3. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI becomes more advanced, it is likely to be used
    


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
    Generated text:  [Your Name]. I am a [occupation] who has been passionate about [objective] for [number of years] years now. I am known for my [tool or hobby] and for my [achievement or skill]. I enjoy [activities or interests] outside of work. I am [age], [gender] and [physical appearance]. I am [occupation] and have [number of years] years of experience in [specific field]. I have been in this field for [number of years] years and have [objective] for [number of years]. I am fluent in [language(s)] and am a [degree]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the Eiffel Tower, Notre-Dame Cathedral, and Montmartre are located. It has a diverse population of around 2.5 million people, and is known for its history, culture, fashion, and food. The city is a popular tourist destination and has a rich literary and artistic history, with notable figures such as Victor Hugo and Pablo Picasso. Paris has a reputation for being a romantic city, with many beautiful parks and gardens that are also popular tourist destinations. The city is home to the Louvre, the National Museum of France, and the Eiffel Tower, which have been a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be influenced by several potential trends, including:
    
    1. Increased Robotic Interventions: The integration of AI into healthcare systems could lead to robotic assistants that can help diagnose and treat diseases, reduce human error, and improve patient outcomes.
    
    2. Autonomous Vehicles: The use of AI and machine learning could revolutionize the transportation sector, with self-driving cars becoming more common. Autonomous vehicles could also reduce accidents and improve traffic flow.
    
    3. Quantum Computing: The ability of quantum computers to process information in a different way than classical computers could have a significant impact on AI and computing, potentially leading to breakthroughs in fields like natural language processing


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

    ].

     I

    'm

     a

     human

     with

     an

     intelligence

     that

     is

     unparalleled

    .

     I

     have

     a

     natural

     ability

     to

     understand

     people

     and

     their

     emotions

    ,

     which

     has

     allowed

     me

     to

     make

     a

     living

     as

     a

     private

     detective

    .

     I

     have

     a

     keen

     eye

     for

     detail

     and

     have

     a

     knack

     for

     unravel

    ing

     complex

     cases

    .

     Whether

     it

    's

     solving

     crimes

     or

     uncover

    ing

     secrets

    ,

     I

    'm

     always

     on

     the

     lookout

     for

     the

     truth

    ,

     no

     matter

     how

     big

     or

     small

    .

     How

     would

     you

     like

     to

     meet

     me

    ?

     [

    Name

    ]?

     This

     is

     just

     a

     thought

     to

     get

     you

     started

    .

     Please

     let

     me

     know

     if

     there

     are

     any

     questions

     or

     if

     there

     is

     anything

     else

     I

     can

     do

    .

     Hello

    ,

     [

    Name

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     known

     for

     its

     rich

     history

    ,

     iconic

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     Its

     enchant

    ing

     streets

    ,

     picturesque

     neighborhoods

    ,

     and

     beautiful

     views

     make

     it

     a

     must

    -

    visit

     destination

     for

     tourists

     and

     locals

     alike

    .

     Paris

     is

     also

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     cultural

     institutions

    ,

     providing

     a

     wide

     range

     of

     experiences

     for

     visitors

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     beauty

    ,

     culture

    ,

     and

     endless

     opportunities

     for

     exploration

    .

     When

     you

     visit

    ,

     you

     can

     enjoy

     the

     sounds

     of

     the

     E

    iff

    el

     Tower

    ,

     the

     hustle

     and

     bust

    le

     of

     the

     French

     Quarter

    ,

     and

     the

     aromatic

     scent

     of

     bou

    ill

    ab

    ais

    se

    .

     The

     city

     offers

     something

     for

     everyone

    ,

     from

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     technological

     and

     social

     developments

    ,

     including

    :
    


    1

    .

     Increased

     precision

     in

     image

     and

     speech

     recognition

    :

     AI

     systems

     will

     continue

     to

     become

     more

     accurate

     and

     capable

     of

     understanding

     and

     interpreting

     human

     speech

     and

     language

    ,

     allowing

     for

     more

     natural

     language

     processing

     and

     language

     translation

    .
    


    2

    .

     Greater

     integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     be

     integrated

     into

     a

     wider

     range

     of

     technologies

    ,

     from

     smart

     homes

     and

     devices

     to

     self

    -driving

     cars

     and

     autonomous

     weapons

     systems

    .
    


    3

    .

     Increased

     reliance

     on

     data

    :

     AI

     systems

     will

     continue

     to

     rely

     on

     large

     amounts

     of

     data

     to

     learn

     and

     improve

    ,

     and

     there

     will

     be

     an

     increased

     focus

     on

     data

     privacy

     and

     security

    .
    


    4

    .

     More

    



```python
llm.shutdown()
```
