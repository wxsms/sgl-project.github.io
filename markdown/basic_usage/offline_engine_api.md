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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.13it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.39it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.39it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.39it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.39it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.39it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 14.39it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 14.39it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:02, 14.39it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.39it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 37.26it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.18 GB):   3%|▎         | 2/58 [00:00<00:05, 10.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.18 GB):   3%|▎         | 2/58 [00:00<00:05, 10.61it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.18 GB):   3%|▎         | 2/58 [00:00<00:05, 10.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.18 GB):   7%|▋         | 4/58 [00:00<00:04, 11.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.18 GB):   7%|▋         | 4/58 [00:00<00:04, 11.08it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.17 GB):   7%|▋         | 4/58 [00:00<00:04, 11.08it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.17 GB):  10%|█         | 6/58 [00:00<00:04, 11.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.16 GB):  10%|█         | 6/58 [00:00<00:04, 11.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.16 GB):  10%|█         | 6/58 [00:00<00:04, 11.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.16 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.16 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.15 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.72it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.15 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.15 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.15 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.15 GB):  21%|██        | 12/58 [00:00<00:03, 14.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.14 GB):  21%|██        | 12/58 [00:00<00:03, 14.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.14 GB):  21%|██        | 12/58 [00:00<00:03, 14.94it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.14 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.14 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.14 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.13 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.13 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.13 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.13 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.12 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.10 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=960 avail_mem=58.12 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.12it/s] Capturing num tokens (num_tokens=896 avail_mem=58.12 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.12it/s]

    Capturing num tokens (num_tokens=832 avail_mem=58.11 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=832 avail_mem=58.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=768 avail_mem=58.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=704 avail_mem=58.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=640 avail_mem=58.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=576 avail_mem=58.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=512 avail_mem=58.09 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=512 avail_mem=58.09 GB):  50%|█████     | 29/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=480 avail_mem=58.10 GB):  50%|█████     | 29/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=448 avail_mem=58.10 GB):  50%|█████     | 29/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=416 avail_mem=58.10 GB):  50%|█████     | 29/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=384 avail_mem=58.10 GB):  50%|█████     | 29/58 [00:01<00:00, 35.71it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.09 GB):  50%|█████     | 29/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=352 avail_mem=58.09 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=320 avail_mem=58.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=288 avail_mem=58.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=256 avail_mem=58.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=240 avail_mem=58.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=224 avail_mem=58.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=224 avail_mem=58.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=208 avail_mem=58.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=192 avail_mem=58.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=176 avail_mem=58.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=160 avail_mem=58.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.24it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=144 avail_mem=58.06 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=128 avail_mem=58.06 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=112 avail_mem=58.06 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=96 avail_mem=58.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.30it/s] Capturing num tokens (num_tokens=80 avail_mem=58.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=64 avail_mem=58.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=64 avail_mem=58.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=48 avail_mem=58.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=32 avail_mem=58.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=28 avail_mem=58.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.67it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=20 avail_mem=58.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=20 avail_mem=58.03 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=16 avail_mem=58.03 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=12 avail_mem=58.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=8 avail_mem=58.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.10it/s] Capturing num tokens (num_tokens=4 avail_mem=58.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.10it/s]Capturing num tokens (num_tokens=4 avail_mem=58.01 GB): 100%|██████████| 58/58 [00:01<00:00, 29.66it/s]


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
    Generated text:  Janice. I’m 18 years old and I’m studying in the University of Southampton and I’m writing my GCSE essay. The topic is on the effects of social media. I’m having problems writing about social media. I’m writing about my problems with my personal life. I have been working on this essay for a month. How do I write about my problems with my personal life? The topic is on the effects of social media. The essay is due in two weeks. I can’t seem to get any ideas. I’m just thinking of making it a “series of essays” and then a final essay on the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking official of the country, holding various important positions in government and public services. President Xi Jinping of China is the ____
    A. Chief Executive
    B. Premier
    C. Chairman
    D. President
    Answer:
    
    A
    
    The primary function of a router is to _______.
    A. exchange information between networks
    B. transmit data
    C. receive data
    D. connect networks
    Answer:
    
    D
    
    2. Which of the following is NOT a requirement for a good interpersonal relationship? 
    A. Being polite
    B. Being friendly
    C. Being sincere
    D. Being serious
    Answer:
    
    D
    
    Patient,
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Bordeaux
    C. Strasbourg
    D. Lyon
    Answer:
    A
    
    Which of the following statements about the Capital of France is incorrect?
    A. The capital of France is Paris.
    B. The capital of France is the French Riviera.
    C. The capital of France is Lyon.
    D. The capital of France is Bordeaux.
    Answer:
    D
    
    Which of the following statements about the capital of France is incorrect? 
    A. The capital of France is Paris.
    B. The capital of France is Bordeaux.
    C. The capital of France is Lyon.
    D. The capital of France is Nice
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it’s time for researchers to take a new approach to making it better. This is where the open-source software platform, Redfin, comes in.
    Redfin is an AI-based platform that will help you plan your perfect home, property, or rental. Based on the population data of the area, Redfin identifies the homes that are most likely to sell and recommends the best one. It also offers a range of real estate features and tools, including its own data-driven pricing strategy and an algorithm that’s based on human-like intelligence. But what makes Redfin so special is its open source foundation. Redfin is open, meaning


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French language. It is also a popular tourist destination, attracting millions of visitors each year. The city is home to many famous French artists, writers, and musicians, and is a major hub for the arts and culture industry. Paris is a city of contrasts, with its modern architecture and historical landmarks blending
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate routine tasks, freeing up human workers to focus on more complex and creative work. This will lead to a shift in the labor market, as more people will be able to focus on higher-value tasks.
    
    2. Enhanced human-machine collaboration: AI will continue to improve its ability to understand and interact with humans, leading to more effective collaboration between humans and machines. This will require a greater understanding of human emotions and motivations, as well as the ability to communicate effectively.
    
    3. AI ethics and privacy concerns: As AI becomes more integrated into our daily lives
    


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
    Generated text:  [Your Name]. I'm a [Your Age] year old [Your Name] and I'm an [Your Profession]. I've always been a [Your Hobby/Interest] person. I've also learned to [Your Character Trait or Personality] in [Your Profession]. I'm [Your Personality Type] and I'm always [Your Motivation for Pursuing Your Profession]. I'm passionate about [Your Profession] because [Your Passion for the Profession]. I'm [Your Personality Type] because [Your Personality Type]. I'm [Your Personality Type] and I enjoy [Your Hobby/Interest]. I'm always [Your Motivation
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Parisi." It is the largest city in France and is also the capital of the country. 
    
    (The Paris).
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and it is expected to continue to grow and change in many ways. Here are some possible trends in AI:
    
    1. Increased autonomous vehicles: Autonomous vehicles have the potential to significantly reduce traffic accidents and improve safety. AI can be used to develop algorithms that can optimize driving routes, predict road conditions, and assist with route planning.
    
    2. Improved natural language processing: With the increasing amount of data being generated every day, AI algorithms that can understand and interpret natural language will become more advanced. This will enable computers to provide more context-aware responses to users, such as asking questions, giving directions, and even responding to jokes.
    
    3


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

    'm

     a

     [

    Career

    ]

     at

     [

    Company

    ].

     I

    've

     always

     loved

     [

    Field

    ]

     and

     have

     always

     been

     eager

     to

     [

    Achie

    ve

    ].

     What

     brings

     you

     to

     [

    Company

    ]

     today

    ?

     I

    'm

     excited

     to

     be

     a

     part

     of

     your

     team

     and

     contribute

     to

     the

     success

     of

     [

    Company

    ].


    [

    Name

    ]

     will

     write

     a

     short

    ,

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    .

     I

     will

     suggest

     their

     name

     and

     industry

    ,

     then

     require

     them

     to

     write

     an

     introduction

     for

     a

     fictional

     character

    .

     I

     will

     also

     provide

     an

     example

    .

     Feel

     free

     to

     use

     any

     fictional

     character

     you

    'd

     like

     to

     use

     for

     a

     self

    -int

    roduction

    ,

     but

     please

     make

     sure

     the

     character

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

    ,

     the

     historic

     and

     world

    -ren

    owned

     city

    ,

     is

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

    ,

     an

     island

     located

     in

     the

     Mediterranean

     Sea

    ,

     and

     is

     the

     largest

     city

     in

     France

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     vibrant

     culture

    ,

     and

     thriving

     art

     scene

    .

     The

     city

     is

     also

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    ,

     as

     well

     as

     its

     many

     museums

    ,

     theaters

    ,

     and

     shopping

     districts

    .

     Paris

     is

     the

     cultural

     and

     economic

     center

     of

     France

    ,

     and

     is

     a

     major

     global

     hub

     for

     trade

    ,

     diplomacy

    ,

     and

     tourism

    .

     Its

     status

     as

     the

     capital

     has

     played

     a

     significant

     role

     in

     shaping

     French

     identity

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

    ,

     and

     it

     is

     likely

     to

     continue

     evolving

     and

     divers

    ifying

     in

     ways

     that

     will

     shape

     society

    ,

     work

    ,

     and

     our

     daily

     lives

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     integration

     with

     human

     society

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     may

     see

     a

     greater

     degree

     of

     collaboration

     between

     humans

     and

     AI

    .

     For

     example

    ,

     we

     may

     see

     more

     AI

    -powered

     personal

     assistants

     like

     Siri

     or

     Alexa

    ,

     as

     well

     as

     more

     AI

    -powered

     medical

     devices

     that

     can

     assist

     with

     diagnosis

     and

     treatment

    .
    


    2

    .

     Enhanced

     privacy

     and

     data

     protection

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     there

     will

     be

     greater

     pressure

     to

     protect

     the

     privacy

     of

     individuals

     and

     their

     data

    .

    



```python
llm.shutdown()
```
