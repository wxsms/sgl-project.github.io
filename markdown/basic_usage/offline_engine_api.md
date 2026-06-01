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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 14.99it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 14.99it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 14.99it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 14.99it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 14.99it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 14.99it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.99it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.99it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.99it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.99it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.99it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.05it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.03it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.03it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.03it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.03it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.03it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.03it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.31it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.31it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.31it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.31it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.31it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.31it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.94it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.85it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 41.67it/s]


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
    Generated text:  Bela. I'm a physician and a board-certified medical doctor (MD) with a specialization in cardiovascular disease. I practice at Cedars-Sinai Medical Center in Los Angeles, CA. My practice focuses on the prevention and treatment of cardiovascular diseases, including coronary artery disease, angina, heart failure, and stroke.
    
    My general approach is to use a multidisciplinary team of cardiologists, cardiologists, and other medical professionals to provide comprehensive, comprehensive, and comprehensive care for my patients. I believe that this multidisciplinary approach to care is the best way to improve the outcomes of heart disease patients, and the best way
    ===============================
    Prompt: The president of the United States is
    Generated text:  a holder of a high power and great responsibility. As the leader of the country, he or she is responsible for managing the country’s economy, solving issues in the nation, and all these efforts will be a result of the people’s support and the participation of the country’s citizens. That is why the United States is known as a country that values the participation of all its citizens in its government and its policies. The President’s role is to make decisions that affect all the citizens of the country. The president makes decisions regarding the national and the state budget and the national security. The president also ensures that the laws are being followed in the
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Nice
    B. Paris
    C. Lyon
    D. Strasbourg
    Answer:
    
    B
    
    The market value of an asset at the end of the period is 100,000 yuan, and the net profit at the end of the period is 15,000 yuan. The return rate of the stock market is 12%, and the return rate of the company's profit is 20%. The return rate of the asset is ____.
    A. 10%
    B. 20%
    C. 23%
    D. 24%
    Answer:
    
    D
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. In fact, we are likely to see a full-scale AI takeover. But what does that mean for the healthcare industry? The future of AI is bright. In fact, we are likely to see a full-scale AI takeover. But what does that mean for the healthcare industry? Here's a look at what's on the horizon.
    A look at what's on the horizon for healthcare AI.
    A look at what's on the horizon for healthcare AI.
    Biopharmaceutical companies, insurance companies, and governments are all working to create a holistic view of patient health. This means that all health-related data, from medical records to genetics


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? I'm a [insert a few key points about your personality, skills, or background]. I'm always looking for new opportunities to grow and learn, and I'm always eager to share my knowledge and experiences with others. Thank you for taking the time to meet me. [Name] [Company Name] [Contact Information] [LinkedIn Profile] [About Me] [Personal Blog] [Other Relevant Links] [Personal Interests] [Professional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history dating back to the Roman Empire. It is a major cultural, economic, and political center, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a vibrant and diverse city with a rich cultural heritage that continues to thrive today.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be increased scrutiny of its ethical implications, including issues such as bias, transparency, and accountability.
    
    3. Development of new AI technologies: AI is likely to continue to develop new technologies that can improve its performance and capabilities, such as machine learning, natural language processing, and quantum computing.
    
    4. Increased focus on privacy and security
    


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
    Generated text:  [Name]. I'm a [job title] at [company], and I'm a great [occupation]. I love [mention a hobby or skill that you're passionate about]. I'm very [personality trait], and I try to always [add a positive statement about myself]. I'm always [add a positive statement about myself], and I'm passionate about [mention a specific interest or area of interest]. What's your name? What's your occupation? What's your hobby or skill that you're passionate about? And what's your personality trait? And what's your positive statement about yourself? And what's your positive statement about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Fact: The city of Paris, located in the Île de la Cité on the Seine River, is the capital of France. It is the largest city in France by population, with an estimated population of around 2.1 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also home to many cultural institutions, including the Louvre Museum, the Musée d'Orsay, and the Musée d'Art Moderne. Paris is a hub of commerce, art
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  poised to be a dynamic and evolving field, with many possible paths and areas of development. Here are some potential trends in AI that could shape the industry in the years ahead:
    
    1. Advancements in machine learning and deep learning: Machine learning and deep learning are areas of AI that are at the forefront of innovation. Advancements in these areas could lead to more sophisticated algorithms, better training of models, and even the ability to learn and improve without being explicitly programmed.
    
    2. Increased focus on ethical and responsible AI: With concerns about AI's potential to be used for harmful or unethical purposes, there is growing pressure to develop AI that is more


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

     a

     [

    Occup

    ation

    ]

     who

     has

     always

     been

     fascinated

     by

     [

    Subject

    ],

     developing

     a

     deep

     understanding

     of

     [

    Specific

     Field

    ].

     I

     am

     a

     [

    Level

     of

     Expert

    ise

    ],

     having

     spent

     [

    Number

     of

     Years

    ]

     years

     hon

    ing

     my

     skills

     in

     [

    Your

     Expert

    ise

    ],

     and

     I

     am

     also

     a

     [

    Level

     of

     Expert

    ise

    ],

     having

     taken

     my

     studies

     to

     a

     whole

     new

     level

    .

     I

     am

     a

     [

    Level

     of

     Expert

    ise

    ]

     and

     have

     used

     my

     knowledge

     to

     [

    Achie

    vements

     or

     Contributions

    ]

     in

     [

    Your

     Field

     of

     Expert

    ise

    ].

     I

     am

     a

     [

    Level

     of

     Expert

    ise

    ]

     and

     have

     built

     a

     [

    Number

     of

     Clients

    ]

     reputation

     in

     the

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     "

    City

     of

     Light

    "

     for

     its

     vibrant

     culture

    ,

     rich

     history

    ,

     and

     impressive

     landmarks

    .

     Paris

     is

     a

     major

     city

     located

     on

     the

     banks

     of

     the

     Se

    ine

     river

    ,

     and

     it

     is

     the

     fourth

     most

     populous

     city

     in

     the

     world

    ,

     after

     New

     York

    ,

     Tokyo

    ,

     and

     Los

     Angeles

    .

     The

     city

     is

     home

     to

     many

     famous

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     as

     well

     as

     fashion

     and

     art

     institutions

    .

     Paris

     is

     also

     known

     for

     its

     delicious

     cuisine

    ,

     such

     as

     French

     cuisine

    ,

     and

     its

     annual

     cultural

     events

    ,

     such

     as

     the

     Cannes

     Film

     Festival

    .

     Paris

     is

     a

     beautiful

     city

     with

     its

     skyline

     dotted

     with

     tall

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     with

     exciting

     new

     possibilities

     on

     the

     horizon

    ,

     ranging

     from

     the

     development

     of

     more

     powerful

     hardware

     to

     the

     emergence

     of

     new

     AI

     applications

    .

     Here

     are

     a

     few

     possible

     future

     trends

     to

     keep

     an

     eye

     on

    :
    


    1

    .

     Increased

     AI

     efficiency

     and

     productivity

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     an

     increase

     in

     the

     efficiency

     and

     productivity

     of

     AI

     systems

    .

     This

     could

     result

     in

     new

     ways

     to

     automate

     tasks

    ,

     reduce

     costs

    ,

     and

     improve

     accuracy

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     prevalent

     in

     our

     lives

    ,

     there

     is

     a

     risk

     of

     increasing

     data

     breaches

     and

     privacy

     violations

    .

     This

     could

     be

     addressed

     by

     developing

     more

     robust

     privacy

     protections

     for

     AI

     systems

    



```python
llm.shutdown()
```
