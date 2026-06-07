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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.44it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.18it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.18it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.18it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.18it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.18it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.18it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.18it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.74 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.73 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.73 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=56.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=960 avail_mem=56.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=56.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=832 avail_mem=56.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=832 avail_mem=56.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=768 avail_mem=56.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=704 avail_mem=55.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=640 avail_mem=55.31 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=576 avail_mem=55.31 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=512 avail_mem=55.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=512 avail_mem=55.30 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=480 avail_mem=55.31 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=448 avail_mem=55.31 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=416 avail_mem=55.31 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]

    Capturing num tokens (num_tokens=384 avail_mem=55.30 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=352 avail_mem=55.30 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=352 avail_mem=55.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=320 avail_mem=55.29 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=288 avail_mem=55.29 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=256 avail_mem=55.29 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=240 avail_mem=55.28 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=224 avail_mem=55.28 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=224 avail_mem=55.28 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.27it/s]Capturing num tokens (num_tokens=208 avail_mem=55.28 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.27it/s]Capturing num tokens (num_tokens=192 avail_mem=55.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=176 avail_mem=55.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.27it/s]

    Capturing num tokens (num_tokens=160 avail_mem=55.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=144 avail_mem=55.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=144 avail_mem=55.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=128 avail_mem=55.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=112 avail_mem=55.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=96 avail_mem=55.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s] Capturing num tokens (num_tokens=80 avail_mem=55.25 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=64 avail_mem=55.25 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=64 avail_mem=55.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=48 avail_mem=55.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=32 avail_mem=55.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=28 avail_mem=55.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.88it/s]

    Capturing num tokens (num_tokens=24 avail_mem=55.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=20 avail_mem=55.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=20 avail_mem=55.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=16 avail_mem=55.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=12 avail_mem=55.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=8 avail_mem=55.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.18it/s] Capturing num tokens (num_tokens=4 avail_mem=55.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=4 avail_mem=55.22 GB): 100%|██████████| 58/58 [00:01<00:00, 41.66it/s]


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
    Generated text:  Shana!
    
    I am a Junior at Chicago State University majoring in Applied Mathematics. My interests are in the fields of numerical methods, functional analysis, and computational neuroscience.
    
    I have a passion for learning and sharing what I have learned, and I strive to do so through my work with the Computational Neuroscience Research Group (CNRG) at Chicago State University.
    
    I am excited to see how I can make a positive impact in the field of computational neuroscience! If you have any questions or topics you'd like to discuss, feel free to reach out to me.
    
    --- 
    
    **Disclaimer:**
    
    I am not a medical professional. This is purely
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government official who holds the title of ( ). A. President of the United States B. Head of State C. Prime Minister D. Chancellor
    A. The president of the United States is a high-ranking government official who holds the title of ().
    A. President of the United States
    B. Head of State
    C. Prime Minister
    D. Chancellor
    Answer: B
    
    Which of the following is a function of the Reducibility and Reliability of Information (RRI) index? 
    A. It reflects the information system's ability to resist information errors. 
    B. It reflects the information system's ability
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Washington D. C. The capital of France is Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  based on the development of the cutting edge, and the field of AI research is the leading race in today’s world. But how to get involved in AI research? Here is an overview of the best ways of getting involved in AI research.
    The major obstacle to AI research is the lack of talent and funding. The only way to get involved in AI research is by demonstrating one’s interest and talent.
    To get involved in AI research, you can begin by studying the basics of AI, which may include:
    - Programming languages
    - Machine learning algorithms
    - Computer hardware
    - Artificial intelligence
    - Machine learning
    - Computer systems
    - Programming


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I love [job title] because [reason for passion]. What do you do for a living? I'm a [job title] at [company name], and I'm passionate about [job title] and I love [job title] because [reason for passion]. What do you enjoy doing? I enjoy [job title] because [reason for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance in French culture and politics. It is a widely recognized and well-known fact that Paris is the capital city of France, and this statement accurately reflects this fact. 
    
    However, it is worth noting that the statement could be more precise by specifying that Paris is the capital city of France, rather than just referring to it as the capital. This would provide a more complete and accurate description of the city's status and importance. 
    
    Overall, the statement is a straightforward and accurate representation of the capital city's location and significance in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Greater integration with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. As these technologies continue to evolve, we can expect to see even more integration
    


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
    Generated text:  [Name] and I'm a [Age] year old [Background] person. I've always been [What motivates you? What motivates you? What would you like to learn more about? What is your greatest strength? What is your weakest point? What challenges do you face? What is your favorite thing to do? What is your favorite hobby? What are your biggest fears? What are your dreams? What is your biggest regret? What is your favorite quote? What is your career path? What is your career goal? What is your biggest challenge? What is your biggest achievement? What is your biggest regret? What is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France. It is known for its rich history, stunning architecture, and vibrant culture. The city is a major tourist destination, famous for its landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major transportation hub, with many train lines and roads that connect it to other parts of France and beyond. Its importance in French politics and economy also makes it a significant city.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by the following trends:
    
    1. Increased Integration: AI is expected to be more integrated into our daily lives, from the smart homes and automated machines to the medical and healthcare applications. We can expect a greater emphasis on AI as a tool for decision-making and decision-support, rather than just a source of work.
    
    2. AI-Driven Personalization: AI will continue to play a larger role in shaping our personal experience. We will see more personalized experiences, where the AI will be able to learn and adapt based on our interactions and preferences.
    
    3. AI Ethics and Privacy Concerns: As AI becomes more integrated into our


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

    Name

    ].

     I

     am

     a

     [

    current

     occupation

     or

     profession

    ]

     with

     [

    number

     of

     years

     in

     that

     field

    ].

     I

     love

     [

    reason

     why

     I

     love

     my

     current

     occupation

    ],

     and

     I

     am

     always

     seeking

     out

     new

     opportunities

     to

     [

    something

     specific

    ].

     I

     am

     [

    number

     of

     years

     in

     this

     profession

    ],

     and

     I

     am

     always

     [

    something

     specific

    ].

     I

     am

     a

     [

    name

    ]

     at

     heart

    ,

     and

     I

     am

     always

     [

    something

     specific

    ].

     I

     love

     [

    reason

     why

    ]

     and

     I

     am

     always

     [

    something

     specific

    ].

     I

     am

     a

     [

    name

    ]

     and

     I

     am

     always

     [

    something

     specific

    ].

     How

     about

     you

    ?

     Can

     you

     provide

     me

     with

     more

     information

     about

     yourself

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     often

     referred

     to

     as

     the

     "

    City

     of

     Love

    "

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

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     known

     for

     its

     historical

     significance

    ,

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     France

    's

     capital

    ,

     Paris

    ,

     is

     renowned

     for

     its

     romantic

     atmosphere

     and

     has

     a

     rich

     cultural

     and

     historical

     significance

    .

     The

     city

     is

     also

     the

     country

    's

     largest

     city

     and

     the

     heart

     of

     its

     economy

    .

     It

     is

     a

     vibrant

     and

     diverse

     city

    ,

     known

     for

     its

     café

     culture

     and

     delicious

     cuisine

    .

     Paris

     is

     famous

     for

     its

     art

    ,

     culture

    ,

     and

     tourism

    ,

     and

     continues

     to

     be

     a

     popular

     destination

     for

     travelers

    .

     The

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     will

     continue

     to

     evolve

     rapidly

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     field

    :
    


    1

    .

     Increased

     automation

     and

     integration

    :

     With

     advancements

     in

     machine

     learning

    ,

     AI

     will

     become

     more

     integrated

     with

     various

     industries

    ,

     making

     it

     easier

     for

     machines

     to

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

    .
    


    2

    .

     Greater

     reliance

     on

     AI

     in

     healthcare

    :

     AI

     will

     play

     an

     increasingly

     important

     role

     in

     healthcare

     by

     improving

     diagnosis

    ,

     personalized

     treatment

    ,

     and

     predicting

     disease

     outbreaks

    .
    


    3

    .

     Enhanced

     creativity

     and

     creativity

    :

     AI

     will

     be

     able

     to

     create

     and

     generate

     art

    ,

     music

    ,

     and

     other

     forms

     of

     creative

     content

     in

     ways

     that

     are

     now

     possible

     only

     through

     human

     intervention

    .
    


    4

    .

     Greater

     privacy

     and

     security

    



```python
llm.shutdown()
```
