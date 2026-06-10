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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.14it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.45it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 27.02it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 36.61it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 36.61it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 36.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   7%|▋         | 4/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.35it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.35it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.13it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.91it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.91it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.91it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.91it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.97it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.97it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.97it/s]

    Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.97it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.97it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  48%|████▊     | 28/58 [00:07<00:16,  1.82it/s]Capturing num tokens (num_tokens=512 avail_mem=75.72 GB):  48%|████▊     | 28/58 [00:07<00:16,  1.82it/s]Capturing num tokens (num_tokens=480 avail_mem=75.46 GB):  48%|████▊     | 28/58 [00:07<00:16,  1.82it/s]Capturing num tokens (num_tokens=448 avail_mem=75.05 GB):  48%|████▊     | 28/58 [00:07<00:16,  1.82it/s]Capturing num tokens (num_tokens=448 avail_mem=75.05 GB):  53%|█████▎    | 31/58 [00:07<00:11,  2.39it/s]Capturing num tokens (num_tokens=416 avail_mem=75.05 GB):  53%|█████▎    | 31/58 [00:07<00:11,  2.39it/s]Capturing num tokens (num_tokens=384 avail_mem=75.04 GB):  53%|█████▎    | 31/58 [00:07<00:11,  2.39it/s]Capturing num tokens (num_tokens=352 avail_mem=75.04 GB):  53%|█████▎    | 31/58 [00:07<00:11,  2.39it/s]Capturing num tokens (num_tokens=320 avail_mem=75.03 GB):  53%|█████▎    | 31/58 [00:07<00:11,  2.39it/s]

    Capturing num tokens (num_tokens=320 avail_mem=75.03 GB):  60%|██████    | 35/58 [00:07<00:06,  3.44it/s]Capturing num tokens (num_tokens=288 avail_mem=75.03 GB):  60%|██████    | 35/58 [00:07<00:06,  3.44it/s]Capturing num tokens (num_tokens=256 avail_mem=75.03 GB):  60%|██████    | 35/58 [00:07<00:06,  3.44it/s]Capturing num tokens (num_tokens=240 avail_mem=75.02 GB):  60%|██████    | 35/58 [00:07<00:06,  3.44it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  60%|██████    | 35/58 [00:07<00:06,  3.44it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  67%|██████▋   | 39/58 [00:07<00:03,  4.85it/s]Capturing num tokens (num_tokens=208 avail_mem=75.02 GB):  67%|██████▋   | 39/58 [00:07<00:03,  4.85it/s]Capturing num tokens (num_tokens=192 avail_mem=75.02 GB):  67%|██████▋   | 39/58 [00:07<00:03,  4.85it/s]Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:07<00:03,  4.85it/s]Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:07<00:03,  4.85it/s]

    Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  74%|███████▍  | 43/58 [00:07<00:02,  6.66it/s]Capturing num tokens (num_tokens=144 avail_mem=75.01 GB):  74%|███████▍  | 43/58 [00:07<00:02,  6.66it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  74%|███████▍  | 43/58 [00:07<00:02,  6.66it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  74%|███████▍  | 43/58 [00:07<00:02,  6.66it/s]Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  74%|███████▍  | 43/58 [00:07<00:02,  6.66it/s] Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  81%|████████  | 47/58 [00:07<00:01,  8.89it/s]Capturing num tokens (num_tokens=80 avail_mem=75.00 GB):  81%|████████  | 47/58 [00:07<00:01,  8.89it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  81%|████████  | 47/58 [00:07<00:01,  8.89it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  81%|████████  | 47/58 [00:07<00:01,  8.89it/s]Capturing num tokens (num_tokens=32 avail_mem=74.99 GB):  81%|████████  | 47/58 [00:08<00:01,  8.89it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  81%|████████  | 47/58 [00:08<00:01,  8.89it/s]Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  90%|████████▉ | 52/58 [00:08<00:00, 12.09it/s]Capturing num tokens (num_tokens=24 avail_mem=74.98 GB):  90%|████████▉ | 52/58 [00:08<00:00, 12.09it/s]Capturing num tokens (num_tokens=20 avail_mem=74.63 GB):  90%|████████▉ | 52/58 [00:08<00:00, 12.09it/s]Capturing num tokens (num_tokens=16 avail_mem=74.53 GB):  90%|████████▉ | 52/58 [00:08<00:00, 12.09it/s]Capturing num tokens (num_tokens=12 avail_mem=74.52 GB):  90%|████████▉ | 52/58 [00:08<00:00, 12.09it/s]Capturing num tokens (num_tokens=12 avail_mem=74.52 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.37it/s]Capturing num tokens (num_tokens=8 avail_mem=74.52 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.37it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.52 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.37it/s]Capturing num tokens (num_tokens=4 avail_mem=74.52 GB): 100%|██████████| 58/58 [00:08<00:00,  6.99it/s]


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
    Generated text:  Suraj and I am the Assistant to President Singh. The Civil Aviation Authority of India (CAA) is a government agency that regulates aviation operations, including civil aviation and military aviation. I am tasked with ensuring that the aviation sector operates safely, efficiently, and in accordance with international aviation standards. Can you tell me about your responsibilities at the CAA?
    Sure Suraj! At the Civil Aviation Authority of India (CAA), I am responsible for enforcing aviation regulations and ensuring that the aviation sector operates safely, efficiently, and in accordance with international standards. I am also responsible for managing the operations of the CAA, ensuring compliance with all regulations and
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a small island nation. The president walks up to the embassy and says to the embassy personnel, "I cannot walk fast enough to be a good ambassador. I have been in a state of constant nervousness for weeks." The embassy personnel replies, "Yes, I see." What does this answer mean?  A. the president is of great intelligence  B. the president is intelligent  C. the president is a great ambassador  D. the president is in a good mood  E. the president is an excellent ambassador
    The answer is C. the president is a great ambassador. 
    
    This answer is based on the information provided
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Nice
    C. Berlin
    D. New York
    Answer:
    A
    
    Among the following buildings, which one has the oldest building? A. The Hall of Mirrors in the Louvre B. The Habsburg Residence in Vienna C. The Tower of London D. The Tate Modern
    Answer:
    C
    
    The capital of Canada is ____
    A. Ottawa
    B. Toronto
    C. Montreal
    D. Vancouver
    Answer:
    A
    
    Among the following options, which one is closest to the area of a square with a side length of 5 cm? A. 1 square meter
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, and it will continue to grow. We need to use the future of AI to create products that will not only serve people but also make society better and more prosperous. What should we do to ensure that AI is used responsibly?
    Here are some suggestions:
    • Respect human rights and ethics, and ensure that AI is used in a way that is transparent and accountable.
    • Develop AI systems that are designed to benefit society and not just to serve companies.
    • Encourage research and development to help create AI that is more ethical and beneficial to society.
    • Ensure that AI is used in a way that promotes innovation and economic growth.
    • Foster


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast and I love [What I Do Best]. I'm always looking for new challenges and learning new things. I'm a [Favorite Subject] person and I enjoy [What I Like to Do]. I'm always up for a good laugh and I love [What I Do for Fun]. I'm a [Favorite Book or Movie] lover and I love [What I Like About It]. I'm a [Favorite Music] lover and I love [What I Like About It]. I'm a [Favorite Sport] lover
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its rich history and diverse cultural scene. Paris is also famous for its fashion industry, art scene, and its role in hosting major international events such as the Olympics and the World Cup. The city is a major tourist destination and is home to many world-renowned museums, theaters, and restaurants. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Greater use of machine learning: Machine learning is likely to become more prevalent in AI, allowing for more sophisticated and adaptive algorithms that can learn from data and improve over time.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    4. Increased focus on privacy and security:
    


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
    Generated text:  [Name] and I am [age] years old. I was born and raised in [location], a place I will never forget. I have a passion for [interests or hobbies] and I strive to [achievements or goals]. I am a [profession or career] that I have always [behave or attitude]. I look up to [other character(s) or sources] for advice and guidance. Thank you for taking the time to meet me! 🌟
    
    Hey there! I'm [Name] and I'm [age]. I'm a [interests/hobby] who's always been fascinated by [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic Eiffel Tower, colorful food, and lively nightlife.
    Paris, France, is the cultural and economic capital of the country, renowned for its towering Eiffel Tower, vibrant atmosphere, and high-end nightlife. The city is also famous for its 19th-century architecture, including the Louvre Museum and Notre-Dame Cathedral. Paris is home to many world-renowned museums and attractions, including the Louvre, the Musée d'Orsay, and the Musée de l'Orangerie. The city's cuisine is also known for its historic French specialties, such as cro
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued advancements in several key areas, including:
    
    1. Self-driving cars: Self-driving cars are expected to become more widespread in the coming years, leading to a more efficient and reliable transportation system.
    
    2. Personalized health care: AI will play a more significant role in personalized health care, enabling doctors to provide more accurate and effective treatments based on individual patient data.
    
    3. Autonomous drones: Autonomous drones will become more advanced and reliable, potentially replacing human pilots in dangerous or dangerous situations.
    
    4. AI-powered technology in healthcare: AI will be used in healthcare to improve patient outcomes, reduce costs, and increase efficiency.
    
    5.


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

    insert

     name

    ],

     and

     I

    'm

     a

     

    2

    5

    -year

    -old

     computer

     scientist

     working

     at

     a

     tech

     startup

    .

     I

     have

     a

     passion

     for

     technology

     and

     am

     always

     looking

     for

     new

     ways

     to

     innovate

     and

     improve

     existing

     systems

    .

     I

     have

     a

     knack

     for

     problem

    -solving

     and

     love

     to

     challenge

     my

     own

     thinking

     and

     ideas

    .

     I

    'm

     excited

     to

     bring

     my

     skills

     and

     enthusiasm

     to

     the

     team

     and

     help

     shape

     the

     future

     of

     [

    insert

     industry

     name

    ].

     How

     can

     I

     get

     started

     with

     this

     role

    ?

     I

     would

     appreciate

     it

     if

     you

     could

     provide

     some

     tips

     on

     how

     to

     get

     started

     with

     a

     role

     like

     this

    .

     I

    'm

     looking

     forward

     to

     hearing

     from

     you

    .

     [

    insert

     name

    ]

     Hello

    ,

     my

     name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     bustling

     and

     historic

     city

     with

     a

     rich

     and

     diverse

     cultural

     heritage

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     numerous

     other

     attractions

    .

     Paris

     is

     a

     major

     economic

     and

     cultural

     center

    ,

     hosting

     important

     international

     events

     and

     hosting

     the

     E

    iff

    el

     Tower

     to

     Paris

     Experience

     Day

    ,

     which

     celebrates

     French

     culture

     and

     traditions

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     with

     many

     visitors

     coming

     to

     see

     its

     iconic

     landmarks

     and

     traditional

     French

     culture

    .

     The

     city

     is

     home

     to

     many

     international

     organizations

     and

     institutions

    ,

     including

     the

     World

     Trade

     Center

    ,

     the

     International

     Olympic

     Committee

    ,

     and

     the

     French

     Parliament

    .

     It

     is

     known

     for

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     is

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     machine

     learning

    ,

     the

     development

     of

     new

     hardware

     and

     software

    ,

     and

     the

     emergence

     of

     new

     technologies

    .

     Here

     are

     some

     potential

     trends

     that

     could

     impact

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     responsible

     AI

    :

     The

     rise

     of

     ethical

     concerns

     about

     AI

     is

     likely

     to

     increase

    ,

     including

     concerns

     about

     bias

    ,

     transparency

    ,

     and

     accountability

    .

     Governments

     and

     organizations

     will

     need

     to

     take

     steps

     to

     develop

     AI

     that

     is

     fair

     and

     transparent

    ,

     and

     will

     require

     greater

     accountability

     for

     any

     AI

     that

     is

     developed

    .
    


    2

    .

     Increased

     collaboration

     between

     AI

     researchers

     and

     other

     fields

    :

     The

     growth

     of

     AI

     in

     other

     fields

    



```python
llm.shutdown()
```
