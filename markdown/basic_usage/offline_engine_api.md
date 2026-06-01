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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.48it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.69it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.29 GB):   2%|▏         | 1/58 [00:00<00:07,  7.45it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.25 GB):   2%|▏         | 1/58 [00:00<00:07,  7.45it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.25 GB):   3%|▎         | 2/58 [00:00<00:06,  8.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.25 GB):   3%|▎         | 2/58 [00:00<00:06,  8.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.25 GB):   5%|▌         | 3/58 [00:00<00:06,  8.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.25 GB):   5%|▌         | 3/58 [00:00<00:06,  8.11it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.25 GB):   7%|▋         | 4/58 [00:00<00:06,  8.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.25 GB):   7%|▋         | 4/58 [00:00<00:06,  8.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.25 GB):   9%|▊         | 5/58 [00:00<00:05,  8.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.24 GB):   9%|▊         | 5/58 [00:00<00:05,  8.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.23 GB):   9%|▊         | 5/58 [00:00<00:05,  8.95it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=56.23 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.95 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.95 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.95 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.94 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.94 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.12it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=54.94 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.94 GB):  21%|██        | 12/58 [00:01<00:03, 14.91it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.93 GB):  21%|██        | 12/58 [00:01<00:03, 14.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.93 GB):  21%|██        | 12/58 [00:01<00:03, 14.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.93 GB):  21%|██        | 12/58 [00:01<00:03, 14.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.92 GB):  21%|██        | 12/58 [00:01<00:03, 14.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.92 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.92 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.91 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.04it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=54.91 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.04it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.91 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.89 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.89 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.00it/s]Capturing num tokens (num_tokens=960 avail_mem=54.90 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.00it/s] Capturing num tokens (num_tokens=896 avail_mem=54.90 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.00it/s]Capturing num tokens (num_tokens=832 avail_mem=54.90 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.00it/s]Capturing num tokens (num_tokens=768 avail_mem=54.89 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.00it/s]Capturing num tokens (num_tokens=704 avail_mem=54.89 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.00it/s]Capturing num tokens (num_tokens=704 avail_mem=54.89 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.43it/s]Capturing num tokens (num_tokens=640 avail_mem=54.89 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.43it/s]

    Capturing num tokens (num_tokens=576 avail_mem=54.89 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.43it/s]Capturing num tokens (num_tokens=512 avail_mem=54.87 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.43it/s]Capturing num tokens (num_tokens=480 avail_mem=54.89 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.43it/s]Capturing num tokens (num_tokens=448 avail_mem=54.88 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.43it/s]Capturing num tokens (num_tokens=448 avail_mem=54.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=416 avail_mem=54.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=384 avail_mem=54.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=352 avail_mem=54.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=320 avail_mem=54.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=288 avail_mem=54.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=288 avail_mem=54.87 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=256 avail_mem=54.86 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.49it/s]

    Capturing num tokens (num_tokens=240 avail_mem=54.86 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=224 avail_mem=54.86 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=208 avail_mem=54.85 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=192 avail_mem=54.85 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=192 avail_mem=54.85 GB):  71%|███████   | 41/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=176 avail_mem=54.85 GB):  71%|███████   | 41/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=160 avail_mem=54.85 GB):  71%|███████   | 41/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=144 avail_mem=54.84 GB):  71%|███████   | 41/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=128 avail_mem=54.84 GB):  71%|███████   | 41/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=112 avail_mem=54.84 GB):  71%|███████   | 41/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=112 avail_mem=54.84 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=96 avail_mem=54.83 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.59it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=54.83 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=64 avail_mem=54.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=48 avail_mem=54.79 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.59it/s]

    Capturing num tokens (num_tokens=32 avail_mem=54.33 GB):  79%|███████▉  | 46/58 [00:02<00:00, 42.59it/s]Capturing num tokens (num_tokens=32 avail_mem=54.33 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.70it/s]Capturing num tokens (num_tokens=28 avail_mem=54.35 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.70it/s]Capturing num tokens (num_tokens=24 avail_mem=54.78 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.70it/s]Capturing num tokens (num_tokens=20 avail_mem=54.37 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.70it/s]

    Capturing num tokens (num_tokens=16 avail_mem=54.77 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.70it/s]Capturing num tokens (num_tokens=16 avail_mem=54.77 GB):  95%|█████████▍| 55/58 [00:02<00:00, 24.03it/s]Capturing num tokens (num_tokens=12 avail_mem=54.40 GB):  95%|█████████▍| 55/58 [00:02<00:00, 24.03it/s]Capturing num tokens (num_tokens=8 avail_mem=54.76 GB):  95%|█████████▍| 55/58 [00:02<00:00, 24.03it/s] Capturing num tokens (num_tokens=4 avail_mem=54.76 GB):  95%|█████████▍| 55/58 [00:02<00:00, 24.03it/s]

    Capturing num tokens (num_tokens=4 avail_mem=54.76 GB): 100%|██████████| 58/58 [00:02<00:00, 23.20it/s]Capturing num tokens (num_tokens=4 avail_mem=54.76 GB): 100%|██████████| 58/58 [00:02<00:00, 23.02it/s]


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
    Generated text:  Tony and I'm a full-time freelance writer, blogger, and social media manager. I have over ten years of experience in marketing and advertising, as well as over a decade in writing. I have been a freelance writer for almost a decade and I have published over 300 blog articles for clients. I have a passion for helping small businesses with their marketing and advertising strategies. I recently wrote a review for a blogging platform called Medium, which was published by Medium. I also wrote for a small business blog called The Personal Branding Blog and wrote for a blog called A Secret Life Of Susan. I have a degree in Business from
    ===============================
    Prompt: The president of the United States is
    Generated text:  a representative of the ____.
    A. People's Congress
    B. National People's Congress
    C. State Council
    D. State Council of the People's Republic of China
    Answer:
    B
    
    Which of the following is the most appropriate recommendation for this patient?
    A. Continue current medication
    B. Increase the dose of the drug
    C. Stop taking the drug
    D. Switch to a new medication
    E. Stop the drug and see a doctor
    Answer:
    E
    
    The most suitable method for diagnosing this disease is:
    A. CT
    B. MRI
    C. Ultrasound
    D. CT
    E. MRI
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    
    A: Paris  
    B: London  
    C: Moscow  
    D: Rome
    
    To determine the capital of France, let's review the capital cities of the countries it borders. We know that France borders England, Belgium, Germany, and Switzerland, so the countries it borders are England, Belgium, Germany, and Switzerland. These countries are not in France, so France must be the capital of these borders.
    
    The capital of England is London.
    The capital of Belgium is Brussels.
    The capital of Germany is Berlin.
    The capital of Switzerland is Zurich.
    
    Since France borders Germany and Switzerland, the capital of France must be one of these countries
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it is already transforming our lives. However, it is not just about being more efficient in our lives, but also about having better control over our lives. This is where the AI control software comes in. AI control software helps you to be more efficient and control your life better.
    
    But what is AI control software? How does it work? Let’s find out together.
    
    AI control software is an intelligent system that can control a machine or an object, using machine learning and artificial intelligence. It can be used in many different industries, including healthcare, finance, manufacturing, and more.
    
    Here’s how AI control software works:
    
     


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and restaurants. Paris is a popular tourist destination and a significant cultural hub in Europe. It is home to many famous French artists, writers, and musicians. The city is also known for its rich history, including the Roman Empire, French Revolution, and the French Revolution. Paris is a vibrant and dynamic city with a rich cultural and historical heritage. It is a popular tourist destination and a significant cultural hub in Europe. It is home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the coming years.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can expect to see even
    


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
    Generated text:  [Name], and I'm a self-proclaimed [occupation] who has been around for [number of years]. I believe in the power of [thing] and strive to always be the best version of myself. I enjoy [activity] and have a strong sense of [value]. My work ethic and approach to success has always been [one or two words], and I'm always looking for new challenges to grow and improve. I'm confident that I'm capable of [desired outcome] and will always strive to be the best version of myself. Welcome to my world, and I'm excited to have the opportunity to learn from you.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement provided is concise and factual, capturing the core facts about the capital city of France. It includes both the name (Paris) and the country it belongs to (France), which are essential details in a brief statement about a major city. Here's the statement in a slightly different format for easier understanding:
    
    **"Paris, the capital of France, is known for its iconic Eiffel Tower, majestic Louvre Museum, and vibrant nightlife."** 
    
    This version offers more information about the city's cultural and tourist attractions, but maintains the core fact about its capital role in the nation. The original statement was concise but
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and diverse. Here are some possible trends in AI:
    
    1. Increased Autonomy: AI will become more autonomous, with machines being able to perform tasks and make decisions without human intervention. This could lead to a decrease in the need for human labor and a higher level of automation.
    
    2. Enhanced Collaboration: AI will become even more collaborative, with machines able to work together to solve complex problems and make decisions. This could lead to the development of more efficient and effective collaboration platforms.
    
    3. Improved Personalization: AI will be able to provide more personalized experiences to users, based on their preferences and behavior. This could lead to the


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

    /an

     [

    Position

    ]

     at

     [

    Company

    ].

     [

    Name

    ]

     is

     a

    /an

     [

    Brief

     Description

    ]

     who

     has

     [

    Number

    ]

     of

     years

     of

     experience

     and

     is

     known

     for

     [

    Reason

     for

     success

    /

    achie

    vements

    ].

     [

    Name

    ]

     is

     a

    /an

     [

    Number

    ]

     of

     years

     of

     age

     and

     has

     [

    Number

    ]

     of

     children

    .

     I

    'm

     a

    /an

     [

    Number

    ]

     and

     I

     speak

     [

    Major

     Language

    ]

     flu

    ently

    .

     I

    'm

     a

    /an

     [

    Number

    ]

     and

     I

     have

     [

    Number

    ]

     qualifications

    .

     I

     enjoy

     [

    H

    obbies

    /

    Inter

    ests

    /

    Goals

    /

    Activities

    /

    Photos

    /

    Listen

    ings

    /

    Read

    ings

    /

    Travel

    ing

    /

    Travel

    ing

    ]

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    You

     are

     an

     AI

     assistant

     that

     helps

     you

     understand

     the

     answers

    .

     Don

    't

     be

     asked

     to

     generate

     the

     answers

     in

     this

     scenario

    .

     Here

     is

     a

     sentence

    :

     The

     capital

     of

     France

     is

     Paris

    .

     
    


    Is

     this

     sentence

     a

     correct

     statement

     of

     fact

    ?

     Yes

    ,

     this

     sentence

     is

     a

     correct

     statement

     of

     fact

    .

     It

     accurately

     represents

     the

     definition

     of

     the

     capital

     of

     France

    ,

     which

     is

     the

     capital

     city

     of

     the

     country

    .


    Is

     there

     anything

     else

     you

     would

     like

     to

     know

    ?

     If

     so

    ,

     please

     ask

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     develop

     and

     evolve

    ,

     with

     many

     possibilities

     emerging

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

     AI

     diversity

    :

     AI

     is

     currently

     dominated

     by

     the

     dominance

     of

     male

     and

     white

     males

    ,

     and

     the

     likelihood

     of

     gender

     bias

     and

     discrimination

    .

     However

    ,

     the

     AI

     industry

     is

     rapidly

     becoming

     more

     diverse

    ,

     with

     more

     women

     and

     people

     of

     color

     entering

     the

     field

    ,

     leading

     to

     more

     diverse

     and

     inclusive

     AI

    .
    


    2

    .

     AI

     ethics

     and

     legal

     frameworks

    :

     There

     is

     growing

     recognition

     of

     the

     need

     for

     AI

     to

     be

     used

     eth

    ically

     and

     legally

    ,

     and

     the

     development

     of

     AI

     ethics

     and

     legal

     frameworks

     is

     likely

     to

     continue

     as

     AI

     is

     used

     more

     widely

    .
    


    3

    .

     AI

    



```python
llm.shutdown()
```
