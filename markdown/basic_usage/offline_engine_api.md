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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:42,  4.95s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:42,  4.95s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.55it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.46it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.46it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.46it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.46it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.46it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.46it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.46it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.46it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 16.68it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 16.68it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 16.68it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 16.68it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 16.68it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 16.68it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 16.68it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 16.68it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.28it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.28it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.28it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.28it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.28it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.28it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.28it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.28it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.18it/s]

    Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 34.84it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 34.84it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 34.84it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 34.84it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 34.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.10 GB):   3%|▎         | 2/58 [00:00<00:04, 12.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:04, 12.40it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:04, 12.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.09 GB):   7%|▋         | 4/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   7%|▋         | 4/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.08 GB):   7%|▋         | 4/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.08 GB):  10%|█         | 6/58 [00:00<00:03, 15.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.07 GB):  10%|█         | 6/58 [00:00<00:03, 15.34it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.07 GB):  10%|█         | 6/58 [00:00<00:03, 15.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):  10%|█         | 6/58 [00:00<00:03, 15.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.29it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  21%|██        | 12/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.04 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.04 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.69it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.78it/s]Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.78it/s] Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.81it/s]Capturing num tokens (num_tokens=896 avail_mem=72.03 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.81it/s]Capturing num tokens (num_tokens=832 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.81it/s]Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.81it/s]

    Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.81it/s]Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.30it/s]Capturing num tokens (num_tokens=640 avail_mem=72.01 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.30it/s]Capturing num tokens (num_tokens=576 avail_mem=72.01 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.30it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.30it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  50%|█████     | 29/58 [00:01<00:01, 28.43it/s]Capturing num tokens (num_tokens=480 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:01<00:01, 28.43it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:01<00:01, 28.43it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:01<00:01, 28.43it/s]Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.32it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.32it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.32it/s]Capturing num tokens (num_tokens=320 avail_mem=71.99 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.32it/s]Capturing num tokens (num_tokens=288 avail_mem=71.99 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.32it/s]Capturing num tokens (num_tokens=288 avail_mem=71.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.88it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.88it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.88it/s]

    Capturing num tokens (num_tokens=224 avail_mem=71.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.88it/s]Capturing num tokens (num_tokens=224 avail_mem=71.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.27it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.27it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.27it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.27it/s]Capturing num tokens (num_tokens=160 avail_mem=71.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.27it/s]Capturing num tokens (num_tokens=160 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.41it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.41it/s]

    Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.41it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.41it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.45it/s] Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=64 avail_mem=71.95 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.89it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.89it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.89it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  86%|████████▌ | 50/58 [00:02<00:00, 29.89it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.58it/s]Capturing num tokens (num_tokens=20 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.58it/s]Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.58it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.58it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.08it/s]Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.08it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=71.92 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.08it/s]Capturing num tokens (num_tokens=4 avail_mem=71.92 GB): 100%|██████████| 58/58 [00:02<00:00, 25.93it/s]


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
    Generated text:  Nelia. I am a bright young girl. I have dark brown hair, a big nose, and a big mouth. I have big eyes that are brown and big ears. I am a doctor. I have a big job but I am never tired. I am very kind. When I meet other people, I make them feel comfortable. I always say "Yes, please". I like to be gentle when I am helping people. And I always smile when I am talking with people. I also like to help people with their problems. I want to be a doctor like you. I'm very happy to be here. What do
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected by a majority of the popular vote but not a simple majority. If the president of a certain country is elected by a majority of the popular vote, what is the minimum number of candidates he must win to ensure that he has a majority of the popular vote?
    To determine the minimum number of candidates the president must win to ensure he has a majority of the popular vote, we need to understand the concept of a majority in a presidential election. A majority means more than half of the votes cast.
    
    In a presidential election, the number of votes that must be cast to win is approximately half of the total number of votes. This is because
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A. Paris B. Berlin C. London D. Moscow
    Answer:
    A
    
    Which of the following substances can react with both dilute sulfuric acid and sodium hydroxide solution?
    A. Sodium chloride
    B. Iron (III) oxide
    C. Potassium chloride
    D. Copper (II) hydroxide
    Answer:
    D
    
    Xiaohong took out a loan of 1000 yuan with a bank. The loan period was 1 year, with annual interest rates of 1.5% and 2.5% for the first and second years, respectively. After the loan
    ===============================
    Prompt: The future of AI is
    Generated text:  not so bright
    
    Posted on 2021-04-12 by EJ
    
    In the field of artificial intelligence, there are many people who have a positive attitude towards it. They believe that AI is the future of technology, and that AI will completely transform every aspect of our lives, from the way we work to the way we live. They see AI as a game-changer, and they are excited about the potential of it to make our lives easier, more efficient, and more comfortable.
    
    However, there are also people who are skeptical about the potential of AI, and they believe that it will not be able


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic, or neutral description of your personality or skills]. I'm always looking for new opportunities and I'm always eager to learn and grow. What are some of your favorite things to do? I love to travel, read, and explore new places. I'm always looking for new experiences and I'm always eager to learn new things. What's your favorite hobby? I love to cook and I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry and has a thriving art scene. It is a popular destination for tourists and locals alike, with many visitors coming to Paris to experience the city's culture and history. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more efficient and effective AI systems that can perform tasks that are currently performed by humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI systems that are used for harmful purposes.
    
    3. Increased use of AI in
    


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
    Generated text:  [Name], and I'm a [Age] year old. I'm a [Occupation/Personality] who enjoys [insert something related to hobbies or interests]. I have a deep appreciation for [insert something you like or dislike about yourself]. I have always been [insert something about your personality or something about you]. I have a great sense of humor, and I'm always looking for ways to make people smile. I'm excited to meet you and learn more about you, so please feel free to ask me any questions you have! [You're ready to say something positive or show your personality].
    **Name:** Alex  
    **
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement is factually accurate based on official information available from reliable sources. Paris is the largest city in France and serves as the political, cultural, and economic center of the country. It was founded in the 8th century and has been the capital of France since 1804. Paris is known for its beautiful architecture, rich cultural heritage, and iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Its status as the capital has been recognized by the European Union and the United Nations. 
    
    Some key facts about Paris include:
    - It has a population of over
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by several trends, including the increasing adoption of machine learning, the emergence of deep learning, and the development of smart algorithms. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent, there will be a greater emphasis on addressing ethical concerns related to AI, such as bias, transparency, and accountability.
    
    2. Enhanced cognitive functions: As AI becomes more advanced, there will be an increase in cognitive functions such as creativity, problem-solving, and decision-making, which will lead to new opportunities for AI to be used in various industries.
    
    3. Integration with other technologies:


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

    ].

     I

    'm

     a

     [

    Your

     Profession

     or

     Special

    ization

    ]

     who

     has

     been

     working

     in

     the

     field

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     enjoy

     [

    Your

     Passion

     or

     Hobby

    ].

     I

     have

     a

     strong

     work

     ethic

     and

     love

     to

     learn

     new

     things

    .

     I

    'm

     always

     up

     for

     a

     challenge

     and

     would

     love

     to

     get

     to

     know

     you

     better

    .

     How

     can

     I

     contact

     you

    ?

     Please

     feel

     free

     to

     call

     or

     email

     me

    .

     What

    's

     one

     thing

     you

    're

     looking

     forward

     to

     the

     most

     in

     the

     coming

     year

    ?

     Writing

    .

     I

     can

    't

     wait

     to

     see

     what

     you

     have

     planned

     for

     yourself

    .

     I

    'm

     excited

     to

     meet

     you

     and

     talk

     about

     my

     career

     and

     interests

    .

     Hello

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     with

     the

     iconic

     E

    iff

    el

     Tower

    .

     
    


    Please

     answer

     the

     following

     question

     based

     on

     the

     factual

     statement

     provided

    :


    What

     is

     the

     capital

     of

     France

    ?

     The

     capital

     of

     France

     is

     Paris

    .

     
    


    To

     verify

     this

     statement

    ,

     we

     can

     cross

    -reference

     it

     with

     other

     established

     facts

     about

     Paris

    .

     For

     example

    ,

     we

     know

     that

     Paris

     is

     the

     capital

     of

     France

     and

     that

     it

     is

     home

     to

     the

     E

    iff

    el

     Tower

    .

     However

    ,

     we

     cannot

     say

     for

     certain

     that

     Paris

     is

     the

     only

     capital

     of

     France

    ,

     as

     there

     are

     other

     cities

     with

     unique

     features

     and

     historical

     significance

    .

     For

     instance

    ,

     Nice

     in

     southwestern

     France

    ,

     Paris

    's

     neighbor

     on

     the

     north

    ,

     and

     Lyon

     in

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     significant

     changes

     as

     we

     move

     towards

     a

     more

     interconnected

     and

     global

    ized

     world

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     is

     expected

     to

     become

     more

     integrated

     into

     human

     work

    ,

     leading

     to

     more

     automation

     and

     less

     reliance

     on

     human

     labor

    .

     This

     could

     lead

     to

     job

     displacement

     but

     also

     create

     new

     opportunities

     for

     AI

    -driven

     solutions

    .
    


    2

    .

     Improved

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     sophisticated

     and

     capable

     of

     handling

     large

     amounts

     of

     data

    ,

     there

     will

     be

     increased

     concerns

     about

     privacy

     and

     security

    .

     As

     a

     result

    ,

     there

     may

     be

     more

     emphasis

     on

     creating

     robust

     data

     privacy

     and

     security

     policies

    .
    


    3

    .

     Enhanced

     creativity

    :

     AI

     is

     likely

     to

     become

    



```python
llm.shutdown()
```
