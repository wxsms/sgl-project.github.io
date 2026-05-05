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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.20it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.19it/s]


    2026-05-05 23:00:40,298 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 23:00:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.26it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.57it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.29it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.99it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  50%|█████     | 29/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 39.99it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.86it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.86it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.86it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.86it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.86it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.86it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.82it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.82it/s]

    Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.35it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.35it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.35it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.35it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.35it/s]

    Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.71it/s]


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
    Generated text:  Sam and I want to be a veterinarian. Can you give me advice on how to become one? Starting a career as a veterinarian can be exciting and fulfilling, but it requires dedication, hard work, and a lot of education. Here are some steps to get started on your journey as a veterinarian:
    
    1. Obtain a bachelor's degree: To become a veterinarian, you'll need to obtain a bachelor's degree in veterinary medicine or related fields. This can be obtained through a private college, a medical school, or a veterinary school. Many universities and community colleges offer this degree.
    
    2. Complete a veterinary training program: After obtaining your bachelor
    ===============================
    Prompt: The president of the United States is
    Generated text:  from the ____.
    A. Asia
    B. Europe
    C. North America
    D. Africa
    Answer:
    C
    
    Which of the following statements about the structure of the human body is correct?
    A. The digestive system can only digest food.
    B. The digestive system can only absorb substances.
    C. The digestive system can only break down food.
    D. The digestive system can only digest, absorb, and metabolize substances.
    Answer:
    D
    
    Based on the following information, answer questions 88 to 92: Chen Mou, female, 28 years old. I, female, 13 years old.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is known as a city of the nation and of the world. From the earliest days of the city of Paris, it was the capital of France. However, it is a city with a rich history, which has influenced the other cities of the world. Paris is a city of the people, the world's capital of learning. It is a world city and a global city.
    History of Paris
    In the eighth century, the ancient Romans set up the city of Rome, and the first king of France, Louis the长寿, named the city of Paris after his own name. This name was later changed to Paris. In
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the research community, which is an important step towards realizing the full potential of AI. This is especially true in the field of computational biology, where researchers are working on developing algorithms that can identify the most important genes for disease and cancer treatments.
    
    One of the key challenges in computational biology is identifying the most important genes for disease and cancer treatments. Researchers have developed various techniques and methods to identify the most important genes, such as the use of machine learning algorithms and bioinformatics tools.
    
    However, the identification of the most important genes is not always straightforward. This is because the genes that are most important for a particular disease or cancer


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I'm always looking for new opportunities to grow and learn, and I'm always eager to share my knowledge and experience with others. What's your background? I have a [insert a short, positive, enthusiastic statement about your background or education]. I'm always looking for new challenges and opportunities to grow and learn, and I'm always eager to share my knowledge and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major transportation hub, with many major highways and rail lines connecting it to other parts of France and the world. The city is known for its fashion industry, with many famous fashion designers and boutiques. Paris is a popular tourist destination, with millions of visitors each year. It is also a cultural hub, with many museums,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to automate many of the tasks that humans currently perform, such as data analysis, decision-making, and routine maintenance. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. Improved privacy and security: As AI becomes more sophisticated, it is likely to require more data to function effectively. This could lead to increased privacy concerns, as companies may
    


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
    Generated text:  [Name]. I am a [Job Title] and I am currently [Job Title]. I am passionate about [My Passion], and I enjoy [My Passion]. I have a keen interest in [My Passion], and I would love to be a [My Passion]. I am a [My Passion] and I believe in [My Passion]. I am confident in my abilities and I believe that I can achieve [My Goal]. I am determined to [My Goal] and I am looking forward to [My Goal]. I am eager to [My Goal], and I am ready to take on any challenge. I am excited to be a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France, with a population of approximately 1.22 million. It is located in the centre of the country and is one of the most cosmopolitan cities in the world. Paris is known for its rich history, art, architecture, and cuisine. The city is home to many world-renowned landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. Paris is also one of the most popular tourist destinations in the world, with millions of visitors each year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be full of exciting developments and changes. Here are some potential future trends in AI:
    
    1. Deep Learning: Deep learning is the next major area of AI research, with the aim of achieving higher levels of accuracy and efficiency in solving complex problems. Deep learning models will continue to improve, with the ability to learn and adapt to new data types and patterns.
    
    2. Self-Driving Cars: Self-driving cars are expected to become more widespread in the future as technology advances. AI will enable self-driving cars to navigate complex traffic environments, recognize traffic signs and hazards, and make safe and efficient decisions.
    
    3. Virtual and Augmented Reality


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

     [

    Age

    ]

     year

    -old

     [

    Occup

    ation

    ].

     I

    'm

     here

     to

     take

     you

     to

     [

    Your

     City

    /C

    ountry

    ],

     [

    Your

     Country

    's

     Capital

     City

    ].

     I

    'm

     also

     a

     [

    Your

     Special

    ity

    /

    Interest

    ].

     Can

     you

     tell

     me

     a

     bit

     about

     yourself

    ?


    Sure

    ,

     how

     about

     you

    ?

     Tell

     me

     a

     little

     bit

     about

     yourself

    .

     I

    'm

     [

    Age

    ],

     [

    Occup

    ation

    ],

     and

     I

    'm

     here

     to

     take

     you

     to

     [

    Your

     City

    /C

    ountry

    ],

     [

    Your

     Country

    's

     Capital

     City

    ].

     I

     also

     have

     a

     [

    Your

     Special

    ity

    /

    Interest

    ].

     What

    's

     your

     story

    ?

     How

     did

     you

     get

     here

     and

     what

    's

     your

     special

     interest

    ?


    I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historical

     landmarks

    ,

     iconic

     architecture

    ,

     and

     diverse

     cultural

     scene

    .

     It

    's

     a

     city

     rich

     in

     art

    ,

     literature

    ,

     and

     gastr

    onomy

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     also

     home

     to

     several

     museums

     and

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

    .

     Paris

     has

     a

     long

     and

     stor

    ied

     history

    ,

     with

     its

     origins

     dating

     back

     to

     the

     Roman

     Empire

     and

     being

     a

     major

     center

     for

     trade

     and

     commerce

     in

     the

     Middle

     Ages

    .

     Today

    ,

     it

     continues

     to

     be

     a

     vibrant

     hub

     of

     creativity

    ,

     culture

    ,

     and

     international

     exchange

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     not

     limited

     to

     one

     single

     direction

    ,

     as

     it

     can

     change

     based

     on

     a

     variety

     of

     factors

    .

     Here

     are

     some

     of

     the

     most

     likely

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

     human

     consciousness

    :

     The

     integration

     of

     AI

     with

     human

     consciousness

     could

     lead

     to

     a

     new

     understanding

     of

     the

     human

     brain

     and

     its

     relationship

     to

     AI

    .

     This

     could

     lead

     to

     breakthrough

    s

     in

     areas

     such

     as

     AI

     ethics

    ,

     ethics

    ,

     and

     artificial

     general

     intelligence

    .
    


    2

    .

     Greater

     privacy

     concerns

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     increasing

     concerns

     about

     the

     privacy

     of

     data

     and

     the

     use

     of

     AI

     for

     surveillance

     and

     monitoring

    .

     There

     will

     be

     a

     need

     to

     establish

     regulations

     and

     guidelines

     to

     ensure

     the

     ethical

     and

    



```python
llm.shutdown()
```
