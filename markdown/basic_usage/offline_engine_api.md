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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.91it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.93it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.08it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.53it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.53it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.53it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.53it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.53it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.53it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.53it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.92it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.92it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.92it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.81it/s]Capturing num tokens (num_tokens=960 avail_mem=53.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.81it/s] Capturing num tokens (num_tokens=896 avail_mem=53.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.81it/s]

    Capturing num tokens (num_tokens=832 avail_mem=53.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.81it/s]Capturing num tokens (num_tokens=832 avail_mem=53.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=768 avail_mem=53.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=704 avail_mem=53.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=640 avail_mem=53.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=576 avail_mem=53.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=512 avail_mem=53.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=512 avail_mem=53.55 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=480 avail_mem=53.56 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=448 avail_mem=53.56 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=416 avail_mem=53.56 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=384 avail_mem=53.56 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]

    Capturing num tokens (num_tokens=352 avail_mem=53.55 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=352 avail_mem=53.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=320 avail_mem=53.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=288 avail_mem=53.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=256 avail_mem=53.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=240 avail_mem=53.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=224 avail_mem=53.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=224 avail_mem=53.53 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.39it/s]Capturing num tokens (num_tokens=208 avail_mem=53.53 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.39it/s]Capturing num tokens (num_tokens=192 avail_mem=53.52 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.39it/s]Capturing num tokens (num_tokens=176 avail_mem=53.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.39it/s]Capturing num tokens (num_tokens=160 avail_mem=53.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.39it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.39it/s]Capturing num tokens (num_tokens=144 avail_mem=53.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=128 avail_mem=53.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=112 avail_mem=53.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=96 avail_mem=53.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s] Capturing num tokens (num_tokens=80 avail_mem=53.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=64 avail_mem=53.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=64 avail_mem=53.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=48 avail_mem=53.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=32 avail_mem=53.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=28 avail_mem=53.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=24 avail_mem=53.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]

    Capturing num tokens (num_tokens=20 avail_mem=53.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.34it/s]Capturing num tokens (num_tokens=20 avail_mem=53.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=16 avail_mem=53.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=12 avail_mem=53.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=8 avail_mem=53.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.02it/s] Capturing num tokens (num_tokens=4 avail_mem=53.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=4 avail_mem=53.47 GB): 100%|██████████| 58/58 [00:01<00:00, 41.86it/s]


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
    Generated text:  Alemayehu and I am the founder of St. Asaph School, a private, multi-academy chain in south-eastern Nigeria. I have over 20 years of experience in education and leadership, having been the head of the Curriculum and School Development Department of a Nigerian government school and the head of school for over a decade at a private school in Lagos. I have also been the lead external evaluator for several Nigerian and international schools. I hold a Bachelor of Arts in Economics from the University of Ogun and a Master of Arts in Curriculum and Assessment from the University of Lagos. I have been a passionate advocate for
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to hold a presidential election or a special election to fill a House seat vacated by a previous incumbent. The incumbent has been in office for 12 years and is only 55 years old. If the president decides to hold an election, the average lifespan of an elected president will be 73.4 years. If the president holds a special election, the average lifespan of the holder will be 58.3 years. If the incumbent is elected by a majority, the average lifespan of the new president will be 70 years. If the incumbent is not elected, the average lifespan of the
    ===============================
    Prompt: The capital of France is
    Generated text:  called:
    A. Paris
    B. Bourg
    C. Marne
    D. Normandy
    Answer:
    
    A
    
    A 1-year-old patient presents with symptoms of nasal congestion, coughing, and purulent nasal discharge. Laboratory tests show a total white blood cell count of 15,000 cells/μL, with 85% neutrophils, and a blood sugar level of 16.7 mmol/L. Which of the following is the most likely diagnosis for this child?
    A. Upper respiratory tract infection
    B. Acute nephritis
    C. Allergic rhinitis
    
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and unpredictable. What is the most likely scenario for the next decade, and how will it impact society?
    I apologize, but I cannot generate this kind of content. Generating a realistic and informative response involving political controversies, future scenarios, or controversial topics like artificial intelligence is beyond my capabilities as an AI assistant. My purpose is to provide assistance with general knowledge and non-political inquiries. If you have any unrelated questions, I'll be glad to help. 
    
    However, I can offer some general information on the potential impacts of AI based on current trends and advancements:
    
    1. **Increased Efficiency and Automation:** AI is already transforming industries by autom


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a popular tourist destination and a major economic center in France. It is also home to many important
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, we can expect to see even more sophisticated applications in healthcare, such as personalized medicine, drug discovery, and image analysis.
    
    2. Greater integration of AI into everyday life: AI is already being integrated into everyday life through things like voice assistants, smart home devices, and self-driving cars. As AI
    


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
    Generated text:  [Name] and I am [Age]. I am a [type of work or career]. I have been [number of years] years working in [field] and [occupation]. I have an [interest or hobby] and [reason why you are good at it]. I have always loved [what you like to do]. I am [height] inches tall and [weight] pounds. I am [gender] and I wear [type of clothing] whenever I go out. I am [color] and [hair color]. I like [daily activities or hobbies], and I enjoy [reason why you enjoy it]. I have a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and the 15th largest city in the world, located on the banks of the Seine River in the center of the French region of Paris. It is also the world's largest city in terms of population. The city is home to many of the world's most famous landmarks, including the Eiffel Tower, the Louvre Museum, the Notre Dame Cathedral, and the Arc de Triomphe, among many others. Paris is also one of the most economically prosperous cities in Europe, with a strong and diverse cultural scene. The city is known for its fashion industry, its art
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of potential and exciting developments that promise to revolutionize our lives in many ways. Here are some possible future trends in AI:
    
    1. Increased automation: As AI becomes more advanced and integrated into our daily lives, it will become more and more automated. This will lead to increased efficiency, productivity, and convenience for many people.
    
    2. Improved ethics and transparency: As AI becomes more advanced, we will need to ensure that it is used ethically and transparently. This will require a greater focus on ethical considerations and the development of AI that is accountable for its decisions.
    
    3. Increased diversity and inclusion: There will be a push to


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

    'm

     a

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

    'm

     excited

     to

     meet

     you

     today

     and

     learn

     more

     about

     you

    .


    I

    'm

     [

    Your

     Age

    ]

     years

     old

    ,

     and

     I

    've

     always

     had

     a

     love

     for

     learning

     and

     adapting

     to

     new

     situations

    .

     Whether

     it

    's

     a

     new

     language

     or

     a

     different

     cultural

     tradition

    ,

     I

    'm

     always

     seeking

     to

     expand

     my

     hor

    izons

     and

     experience

     new

     things

    .


    I

    'm

     driven

     by

     a

     desire

     to

     improve

     myself

     and

     take

     on

     new

     challenges

    .

     Whether

     it

    's

     learning

     a

     new

     skill

    ,

     traveling

     to

     a

     new

     place

    ,

     or

     participating

     in

     a

     new

     project

    ,

     I

    'm

     always

     eager

     to

     take

     the

     next

     step

     in

     my

     growth

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    la

     grande

     ville

    ",

     an

     important

     cultural

    ,

     economic

    ,

     and

     political

     center

    .

     It

     is

     located

     in

     the

     north

     of

     the

     country

     and

     is

     the

     largest

     city

     in

     Europe

     by

     population

    .

     The

     city

     is

     home

     to

     numerous

     historical

     landmarks

    ,

     including

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

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    ,

     among

     others

    .

     Paris

     has

     a

     rich

     and

     diverse

     culture

     that

     reflects

     the

     country

    's

     history

     and

     identity

    ,

     and

     is

     a

     major

     tourist

     destination

    .

     The

     French

     capital

     is

     often

     considered

     one

     of

     the

     most

     beautiful

     cities

     in

     the

     world

    .

     (

    source

    :

     Wikipedia

    )

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     combination

     of

     rapid

     technological

     advancements

    ,

     significant

     changes

     in

     how

     we

     live

     and

     work

    ,

     and

     a

     growing

     emphasis

     on

     ethical

     considerations

     and

     societal

     impacts

    .

     Here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     increase

     in

     precision

     and

     accuracy

     in

     its

     predictions

     and

     applications

    .

     As

     AI

     models

     become

     more

     sophisticated

     and

     capable

     of

     learning

     from

     data

    ,

     they

     are

     likely

     to

     become

     even

     more

     accurate

     at

     making

     predictions

     and

     identifying

     patterns

     in

     complex

     data

     sets

    .
    


    2

    .

     Democrat

    ization

    :

     As

     AI

     becomes

     more

     widely

     available

     and

     accessible

    ,

     it

     is

     likely

     to

     democrat

    ize

     access

    



```python
llm.shutdown()
```
