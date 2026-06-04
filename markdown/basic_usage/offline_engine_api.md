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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.70it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.56it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 26.31it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 34.10it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.05 GB):   3%|▎         | 2/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.04 GB):   3%|▎         | 2/58 [00:00<00:03, 14.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.04 GB):   3%|▎         | 2/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.04 GB):   7%|▋         | 4/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.04 GB):   7%|▋         | 4/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.03 GB):   7%|▋         | 4/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.02 GB):   7%|▋         | 4/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.02 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.02 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.93it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=59.02 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.02 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.01 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.01 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.00 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.00 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.51it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.99 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.99 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.99 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.99 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.99 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=960 avail_mem=58.98 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.50it/s] Capturing num tokens (num_tokens=896 avail_mem=58.98 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=832 avail_mem=58.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=768 avail_mem=58.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=704 avail_mem=58.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.50it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.97 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=640 avail_mem=58.96 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=576 avail_mem=58.96 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=512 avail_mem=58.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=480 avail_mem=58.96 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=448 avail_mem=58.96 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=448 avail_mem=58.96 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=416 avail_mem=58.96 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=384 avail_mem=58.96 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=352 avail_mem=58.95 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.94it/s]Capturing num tokens (num_tokens=320 avail_mem=58.95 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.94it/s]Capturing num tokens (num_tokens=288 avail_mem=58.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.94it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=256 avail_mem=58.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=240 avail_mem=58.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=224 avail_mem=58.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=208 avail_mem=58.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=192 avail_mem=58.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=192 avail_mem=58.93 GB):  71%|███████   | 41/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=176 avail_mem=58.93 GB):  71%|███████   | 41/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=160 avail_mem=58.92 GB):  71%|███████   | 41/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=144 avail_mem=58.92 GB):  71%|███████   | 41/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=128 avail_mem=58.92 GB):  71%|███████   | 41/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=112 avail_mem=58.92 GB):  71%|███████   | 41/58 [00:01<00:00, 45.42it/s]

    Capturing num tokens (num_tokens=112 avail_mem=58.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.32it/s]Capturing num tokens (num_tokens=96 avail_mem=58.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.32it/s] Capturing num tokens (num_tokens=80 avail_mem=58.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.32it/s]Capturing num tokens (num_tokens=64 avail_mem=58.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.32it/s]Capturing num tokens (num_tokens=48 avail_mem=58.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.32it/s]Capturing num tokens (num_tokens=32 avail_mem=58.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.32it/s]Capturing num tokens (num_tokens=32 avail_mem=58.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=28 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=24 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=20 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=16 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=12 avail_mem=58.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.94it/s]

    Capturing num tokens (num_tokens=12 avail_mem=58.88 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=8 avail_mem=58.88 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.51it/s] Capturing num tokens (num_tokens=4 avail_mem=58.88 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=4 avail_mem=58.88 GB): 100%|██████████| 58/58 [00:01<00:00, 38.05it/s]


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
    Generated text:  Daniel. I'm a high school student and I'm very into computers. I want to write a story about a scientist who becomes the next leader of the world. What should I write about? It is not for the weak.
    Certainly! Given the strong desire for leadership, your story could be centered around a brilliant scientist who is eager to seize the opportunity to revolutionize the world. Here are some ideas to get you started:
    
    ---
    
    **Title: The Innovator's Edge**
    
    Daniel, a brilliant young physicist and software developer, had always been fascinated by computers and technology. When his family moved to a small town, they purchased a computer
    ===============================
    Prompt: The president of the United States is
    Generated text:  two years older than the president of Central America, and three years younger than the president of Asia. If the president of Asia is 25 years old, how old is the president of Central America? Let's denote the age of the president of Asia as \( A \), the president of Central America as \( C \), and the president of the United States as \( U \).
    
    We are given that:
    1. The president of Asia is 25 years old, so \( A = 25 \).
    2. The president of the United States is two years older than the president of Central America, so \( U =
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and is located in the center of which of the following regions?
    A) North-Western Europe
    B) Central Europe
    C) Southern Europe
    D) Eastern Europe
    
    1. **Identify the capital of France:**
       - The capital of France is typically Paris. This is because Paris is the largest and most populous city in France, and it serves as the administrative and cultural center of the country.
    
    2. **Determine the region of Paris:**
       - Paris is located in the northern region of France, specifically in the region of the Alps and the Pyrenees mountains.
    
    3. **Conclusion:**
    
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, but many researchers and businesses are still just beginning to understand and harness its power. AI is being used in a variety of fields, including healthcare, finance, transportation, and more. While AI has the potential to revolutionize many industries, it is also raising important ethical, social, and legal questions. As AI continues to advance, it is important for stakeholders to take steps to ensure that the technology is used responsibly and in ways that benefit all individuals and communities.
    AI is fundamentally different from traditional computing because it involves the use of algorithms and machine learning to process data and make decisions. Traditional computing relies on the use of hardware and


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], and I'm [Hobbies/Interests]. I'm always looking for new experiences and learning new things, and I'm always eager to share my knowledge and skills with others. I'm always looking for opportunities to grow and improve, and I'm always eager to learn and adapt to new challenges. I'm always eager to help others and make a positive impact in the world
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and is a major tourist destination. It is home to many famous French artists, writers, and musicians. The city is also known for its cuisine, including its famous croissants and its traditional French dishes. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that is both ancient and modern,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more efficient and capable of performing tasks that were previously done by humans. This could lead to a more automated workforce, where machines can perform tasks that were previously done by humans.
    
    2. AI ethics and privacy: As AI becomes more advanced, there will be a need to address ethical and privacy concerns. This could lead to new regulations and standards for AI development and use.
    
    3. AI for healthcare: AI is already being
    


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
    Generated text:  [Your Name], and I am a [Title/Title and Subtitle] at [Company Name]. I am an [age], [gender], and [country] [nationality] [Hometown]. I am a [field of study] who have always had an interest in [interest or hobby] and love to [adventure or activity]. I am excited to have the opportunity to work with you at [company name], and I am eager to bring my [strength or skill] to the team. Thank you! 🌟✨ [Your Name] 🌊 [Your Job Title] 🌊 [Your Company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is located in the northwest of the country and is the largest city in France. It is the political, cultural, and economic center of France. Paris is also known as "la Ville Flottante" meaning floating city. The city is surrounded by several hills that give it a picturesque appearance, and it is often referred to as the "City of Lights" due to its sunny climate. It is home to many iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum, and it is an important center for arts and culture. Paris is also known for its cuisine, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be a blend of several different trends and advancements. Here are some of the most likely areas of development:
    
    1. Self-driving cars: Self-driving cars are expected to become more prevalent as AI technology continues to improve and costs decrease. This will likely lead to more widespread adoption of autonomous vehicles on the road.
    
    2. Speech recognition and language translation: AI will continue to advance in areas like speech recognition and language translation, making it easier for machines to understand and interact with humans. This will likely lead to more widespread use of AI-powered virtual assistants and chatbots.
    
    3. Robotics: AI is already making inroads into many areas of


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

     [

    Age

    ].

     I

    'm

     a

     [

    occupation

    ]

     who

     has

     been

     working

     in

     [

    job

     title

    ]

     for

     [

    number

     of

     years

    ].

     I

    'm

     passionate

     about

     [

    what

     interests

     you

    ],

     and

     my

     [

    strength

     or

     skill

    ]

     is

     [

    how

     you

     demonstrate

     this

    ].

     I

    'm

     always

     eager

     to

     learn

     and

     continue

     to

     grow

     as

     an

     individual

    ,

     and

     I

    'm

     always

     ready

     to

     challenge

     myself

     and

     seek

     new

     opportunities

    .

     If

     you

     need

     help

     or

     advice

    ,

     don

    't

     hesitate

     to

     reach

     out

     to

     me

    !

     What

     is

     your

     profession

     and

     how

     has

     it

     been

     shaping

     your

     life

    ?

     [

    Optional

    ]:

     How

     do

     you

     keep

     healthy

     and

     active

    ?

     [

    Optional

    ]:

     What

     are

     your

     goals

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     
    


    (

    5

    0

     words

    )

     
    


    [

    Answer

     in

     French

    ]

     


    Le

     capital

     de

     la

     France

     est

     Paris

    ,

     conn

    u

     pour

     ses

     landmarks

     t

    els

     que

     la

     Cath

    éd

    rale

     Notre

    -D

    ame

    ,

     l

    '

    É

    to

    ile

     de

     la

     Ch

    amps

    -

    É

    lys

    ées

     et

     le

     Mus

    ée

     Lou

    vre

    .

     
    


    (

    5

    0

     words

    )

     
    


    [N

    ou

    velle

     version

     en

     anglais

    ]


    The

     capital

     of

     France

     is

     Paris

    ,

     renowned

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     rapidly

     changing

    ,

     and

     it

     will

     likely

     involve

     several

     key

     trends

     that

     are

     expected

     to

     shape

     the

     industry

     in

     the

     coming

     years

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

     Deep

     Learning

     and

     Machine

     Learning

    :

     These

     are

     the

     two

     main

     branches

     of

     AI

     that

     have

     been

     developing

     in

     recent

     years

    .

     Deep

     learning

     is

     based

     on

     neural

     networks

     with

     many

     layers

    ,

     while

     machine

     learning

     is

     based

     on

     algorithms

     that

     can

     learn

     from

     data

     without

     being

     explicitly

     programmed

    .

     Both

     of

     these

     techniques

     are

     expected

     to

     continue

     to

     advance

     and

     become

     more

     sophisticated

    .
    


    2

    .

     Natural

     Language

     Processing

    :

     AI

     is

     already

     becoming

     increasingly

     sophisticated

     in

     natural

     language

     processing

    ,

     and

     it

     is

     expected

     to

     continue

     to

    



```python
llm.shutdown()
```
