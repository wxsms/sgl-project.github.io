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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.19s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.19s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.19s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.19it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 31.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s] Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.97it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.43it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.43it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.43it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.27it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.42it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.42it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.42it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.42it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.42it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.42it/s] Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 39.74it/s]


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
    Generated text:  Emil and I'm a Research Scientist at the University of California at Berkeley. I'm working on a project called P4, which is about the human brain and how it responds to external stimuli. I'm excited to share my research with you and to help you understand how the brain responds to different stimuli. If you have any questions about what I'm studying or what my research is doing, please feel free to ask. I'm here to help and answer any questions you may have. Let's get started. What would you like to ask me? If you have any questions or concerns about what I'm doing or what my research is about
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government official who is elected for a two-year term. The person who has won the most elections in a given election cycle, and represents the largest majority, is elected to the position of President. When they were elected in the 2020 election cycle, Democrats won 52% of the votes and Republicans won 48%. Republicans won 52% of the seats in the House of Representatives and 60% of the seats in the Senate. What is the difference in the number of votes that Republicans received and Democrats received? The number of votes Republicans received was 52% of 5
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is a historical center, famous for its old city, the Louvre, the Arc de Triomphe, and the Notre-Dame Cathedral. There are many interesting landmarks and museums in Paris, like the Musée d'Orsay, the Musée d'Orsay and the Musée des Arts Décoratifs. It is also a place where the great writers of literature, music, and history are buried in the Abbey of Saint-Denis. It is a perfect location for a great wine tour and a great place to enjoy its beautiful climate. The climate of Paris is very pleasant all year round. It receives
    ===============================
    Prompt: The future of AI is
    Generated text:  fast approaching and the latest advancements in this field are expected to have a significant impact on various fields such as healthcare, finance, transportation, and more. One of the most exciting developments in AI is the ability to create highly sophisticated models that can analyze and interpret complex data sets with unprecedented accuracy and speed. This capability has already been demonstrated in the field of medical imaging, where AI algorithms can accurately diagnose and treat diseases with unprecedented precision.
    
    However, with such powerful AI capabilities, comes the need for ethical considerations and accountability. One of the biggest ethical concerns is the potential for biased AI algorithms to perpetuate discrimination and discrimination. The healthcare industry, for


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I love [book/movie]. I'm always looking for new ways
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament and the French Quarter, where many famous French artists and intellectuals have lived and worked. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is the largest city in France and is home to many of the country's most famous landmarks and attractions. Paris is also known for its cuisine, with many famous French dishes and restaurants serving up to the best in the world. The city is a melting pot of cultures and is a major hub for business
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate tasks that are currently done by humans, such as data analysis, decision-making, and routine maintenance. This will lead to increased efficiency and productivity, but it will also create new job opportunities.
    
    2. Enhanced human-computer interaction: AI will continue to improve its ability to understand and respond to human language, emotions, and intentions. This will lead to more natural and intuitive interactions between humans and machines.
    
    3. AI will become more integrated with other technologies: AI will continue to be integrated with other technologies, such as sensors, cameras, and machine
    


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
    Generated text:  [Name], and I'm a [job title] who specializes in [specialization or profession]. I have [number of years of experience] years of experience in [occupation], and I'm always eager to learn new skills and stay up-to-date on the latest trends in my field. In my free time, I enjoy [hobbies or interests] and try to make the most of my experiences in life. I'm passionate about [motivational quote or message], and I strive to make a positive impact on the world through my work. Thank you for considering me for a job, and I look forward to the opportunity to help
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. The city is located in the south of France and is the third-largest city in the European Union. Paris has a rich cultural and artistic heritage, with landmarks such as Notre-Dame Cathedral and the Louvre Museum. The city is known for its art, fashion, and cuisine, as well as its famous fashion runway. Paris is a major tourist destination, with visitors from all over the world flocking to the city to experience its diverse culture and history. The city is also home to several international landmarks, including the Eiffel Tower, Louvre Museum, and the Palace of Versailles
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by several key trends:
    
    1. Advancements in machine learning and deep learning: The development of more powerful algorithms and the implementation of machine learning and deep learning techniques will continue to improve AI performance and make it more capable of solving increasingly complex problems. This will lead to the creation of more sophisticated and autonomous machines that can perform a wider range of tasks, from language translation to autonomous vehicles.
    
    2. Increased focus on fairness and ethical considerations: As AI becomes more sophisticated, there will be increased focus on fairness and ethical considerations. This will lead to a greater emphasis on developing algorithms and models that are designed to be unbiased and fair


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

    name

    ],

     and

     I

     am

     a

     [

    role

    ]

     with

     [

    number

    ]

     years

     of

     experience

     in

     [

    specific

     field

    ].

     I

     am

     confident

     in

     my

     ability

     to

     [

    describe

     a

     task

     or

     accomplishment

     you

     have

     completed

    ].

     I

     am

     dedicated

     to

     [

    describe

     a

     goal

     you

     have

     set

     or

     a

     project

     you

     are

     working

     on

    ],

     and

     I

     am

     always

     looking

     to

     learn

     and

     grow

     as

     a

     professional

    .

     I

     am

     passionate

     about

     [

    mention

     a

     hobby

     or

     interest

     that

     you

     have

    ].

     I

     am

     driven

     by

     [

    mention

     a

     reason

     why

     you

     are

     pursuing

     this

     path

    ].

     My

     work

     ethic

     is

     [

    describe

     your

     character

     trait

     or

     personality

    ].

     I

     am

     confident

     in

     my

     ability

     to

     [

    describe

     a

     project

     or

     role

     you

     have

     taken

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     provides

     a

     brief

     and

     accurate

     description

     of

     the

     capital

     city

     of

     France

    ,

     which

     is

     a

     large

     metropolitan

     area

     located

     in

     the

     center

     of

     the

     country

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     the

     capital

     of

     France

    .

     The

     city

     is

     known

     for

     its

     rich

     cultural

     history

    ,

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     and

     its

     role

     as

     a

     major

     cultural

     and

     economic

     hub

     for

     France

    .

     
    


    Paris

     is

     also

     home

     to

     many

     notable

     universities

     and

     colleges

    ,

     and

     is

     a

     major

     center

     for

     art

    ,

     music

    ,

     theater

    ,

     and

     fashion

    .

     The

     city

     also

     has

     a

     strong

     economy

    ,

     with

     its

     economy

     being

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     there

     are

     a

     few

     trends

     that

     are

     likely

     to

     shape

     its

     trajectory

    :
    


    1

    .

     Increased

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     be

     essential

     to

     address

     the

     ethical

     concerns

     that

     it

     raises

    .

     This

     includes

     issues

     such

     as

     bias

    ,

     transparency

    ,

     accountability

    ,

     and

     the

     potential

     for

     AI

     to

     replace

     jobs

    .
    


    2

    .

     Adv

    ancements

     in

     hardware

     and

     software

    :

     As

     AI

     technology

     becomes

     more

     powerful

     and

     capable

    ,

     we

     will

     see

     continued

     advancements

     in

     hardware

     and

     software

     that

     will

     make

     it

     possible

     to

     build

     more

     powerful

     and

     complex

     AI

     systems

    .
    


    3

    .

     Integration

     of

     AI

     into

     new

     applications

    :

     As

     AI

     becomes

     more

     integrated

     into

     new

     applications

    ,

     it

     will

     become

     an

     increasingly

     important

     part

     of

    



```python
llm.shutdown()
```
