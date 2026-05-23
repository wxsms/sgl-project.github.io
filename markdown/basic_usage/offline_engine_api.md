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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.84it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.90it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.90it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.29it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.27it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.46it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.46it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.46it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.46it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.46it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.46it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.31it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.31it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.02it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.86it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.21it/s] Capturing num tokens (num_tokens=4 avail_mem=72.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.21it/s]Capturing num tokens (num_tokens=4 avail_mem=72.41 GB): 100%|██████████| 58/58 [00:01<00:00, 40.77it/s]


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
    Generated text:  _____ and I _____ my pen.
    A. I'm, use
    B. I, use
    C. My, use
    D. Me, use
    Answer: C
    
    Can a single power supply be connected in parallel with multiple power supplies for load balancing?
    A. Correct
    B. Incorrect
    Answer: B
    
    On the 15th, after the expiration of the supervision period, the supervision unit shall, in accordance with the supervision contract, handle the supervision work, organize the completion of the supervision project, and issue a supervision work completion notification.
    A. Correct
    B. Incorrect
    Answer: B
    
    In most cases
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize a deal with the European Union to bring about a temporary suspension of sanctions on the United States and the European Union. The president believes that sanctions will be lifted when the $1.2 trillion debt is paid off. However, the president believes that if the debt is not paid off within a year, the sanctions will be lifted.
    
    The president initially had $1.5 trillion in debt. The total interest paid on the debt is $600 billion. If the president has no other expenses, what is the new total amount of debt that the president will have in the long term?
    
    Let's calculate the new total amount
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In the first month of 2014, the price of a common house in Paris was 3000 Euros. In the second month, the price of a common house in Paris was 4000 Euros. In the third month, the price of a common house in Paris was 5000 Euros, and in the fourth month, the price of a common house in Paris was 6000 Euros.
    
    What is the arithmetic mean of the prices of the houses in Paris in the first four months of 2014?
    
    To find the arithmetic mean of the prices of the
    ===============================
    Prompt: The future of AI is
    Generated text:  looking more and more complex. To be able to address the biggest challenges in AI, we need a new approach that focuses on improving the quality of AI research and development. This will lead to the creation of a new kind of AI, which is more efficient, reliable and predictable. It will also improve the quality of AI applications and enable the development of new forms of AI that are better suited to the needs of humans.
    What type of AI is more complex than other AI? (If the question is unanswerable, say "unanswerable) unanswerable


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. Let's chat! [Name] [Job Title] [Company Name] [Company Address] [City, State, ZIP Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance within the country. However, it could be expanded to include additional information about Paris's cultural, historical, or economic importance, depending on the context of the question. For example, the statement could be expanded to mention that Paris is the birthplace of the French Revolution and the capital of the French Riviera. 
    
    Overall, the statement is a good starting point for discussing the capital city of France, but it could be further elaborated to provide a more comprehensive and accurate representation of the city's significance. 
    
    Therefore,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for workers.
    
    2. AI-powered healthcare: AI will play a key role in improving the accuracy and efficiency of medical diagnosis and treatment. This will likely lead to more personalized and effective treatments, as well as reduced
    


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
    Generated text:  [Name]. I'm a [noun] who enjoys [verb] things, [job title]ing me, and maintaining [thing].
    
    I'm a [noun] who enjoys [verb] things, [job title]ing me, and maintaining [thing]. I'm a [noun] who enjoys [verb] things, [job title]ing me, and maintaining [thing]. I'm a [noun] who enjoys [verb] things, [job title]ing me, and maintaining [thing]. I'm a [noun] who enjoys [verb] things, [job title]ing me, and maintaining [thing
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of lights and the French capital of Paris, located in the central part of the country, is the largest and most populous city in France, and is home to a diverse population of about 3 million people. It is the heart of Paris’ art, culture, and intellectual life, and is home to some of the most famous landmarks in the world, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It has a rich history dating back to the Middle Ages, and is known for its lively nightlife, gourmet dining, and culinary tradition, as well as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting and varied, and it is likely that we will see many interesting trends in the years to come. Here are some possible future trends in AI:
    
    1. Personalization and personalization: With more and more data available, we can expect AI to be able to learn more about individual users and provide more personalized experiences. This will not only increase efficiency but also help businesses to stand out in the market.
    
    2. Autonomous vehicles: The use of AI in autonomous vehicles is already being explored, and in the future, we may see vehicles that can drive themselves or operate on roads without human intervention. This will change the way we travel and


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

    Occup

    ation

    ]

    !
    


    As

     someone

     who

     has

     worked

     in

     the

     field

     for

     many

     years

    ,

     I

     have

     a

     deep

     understanding

     of

     the

     various

     challenges

     and

     opportunities

     that

     come

     with

     my

     profession

    .

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    ,

     and

     I

     enjoy

     sharing

     my

     knowledge

     with

     others

     in

     a

     way

     that

     is

     both

     informative

     and

     engaging

    .
    


    My

     work

     ethic

     is

     a

     key

     aspect

     of

     my

     character

    ,

     and

     I

     strive

     to

     always

     put

     my

     colleagues

    '

     needs

     and

     the

     company

    's

     goals

     first

    .

     I

    'm

     always

     willing

     to

     go

     above

     and

     beyond

     to

     ensure

     that

     the

     work

     is

     done

     to

     the

     best

     of

     my

     ability

    ,

     and

     I

    'm

     proud

     to

     have

     made

     a

     difference

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     city

     is

     known

     for

     its

     grand

     architecture

    ,

     rich

     history

    ,

     and

     lively

     culture

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     fifth

    -largest

     city

     in

     the

     world

     by

     population

    .

     Paris

     has

     a

     diverse

     population

     of

     over

     

    2

     million

     people

     and

     is

     home

     to

     many

     cultural

     institutions

    ,

     museums

    ,

     and

     art

     galleries

    .

     The

     city

     is

     also

     a

     major

     tourist

     destination

    ,

     with

     over

     

    1

    0

     million

     visitors

     annually

    .

     Paris

     is

     known

     for

     its

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

     the

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

     The

     city

     is

     also

     known

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     music

    .

     It

     is

     one

     of

     the

     most

     visited

     cities

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

     and

     promising

    ,

     with

     many

     potential

     areas

     of

     advancement

    .

     Some

     of

     the

     key

     trends

     that

     have

     the

     potential

     to

     shape

     the

     future

     of

     AI

     include

    :
    


    1

    .

     Improved

     privacy

     and

     security

    :

     AI

     is

     likely

     to

     become

     more

     personalized

     and

     require

     more

     advanced

     data

     analysis

     to

     understand

     and

     respond

     to

     individual

     users

    .

     This

     will

     require

     significant

     improvements

     in

     privacy

     and

     security

    ,

     particularly

     in

     the

     handling

     of

     sensitive

     data

    .
    


    2

    .

     Increased

     automation

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     to

     automate

     many

     tasks

     that

     are

     currently

     performed

     manually

    ,

     such

     as

     routine

     maintenance

    ,

     data

     analysis

    ,

     and

     customer

     support

    .

     This

     will

     require

     significant

     investment

     in

     AI

     development

     and

     infrastructure

    .
    


    3

    .

     Human

     oversight

    :

     As

     AI

    



```python
llm.shutdown()
```
