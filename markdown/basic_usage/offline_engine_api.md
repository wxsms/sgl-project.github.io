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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.24it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.00it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.22 GB):   3%|▎         | 2/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.22 GB):   3%|▎         | 2/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.22 GB):   3%|▎         | 2/58 [00:00<00:03, 14.13it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.22 GB):   7%|▋         | 4/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.22 GB):   7%|▋         | 4/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.21 GB):   7%|▋         | 4/58 [00:00<00:03, 16.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.21 GB):  10%|█         | 6/58 [00:00<00:03, 17.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.20 GB):  10%|█         | 6/58 [00:00<00:03, 17.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.20 GB):  10%|█         | 6/58 [00:00<00:03, 17.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.20 GB):  10%|█         | 6/58 [00:00<00:03, 17.13it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.20 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.19 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.19 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.19 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.19 GB):  21%|██        | 12/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.18 GB):  21%|██        | 12/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.18 GB):  21%|██        | 12/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.18 GB):  21%|██        | 12/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.17 GB):  21%|██        | 12/58 [00:00<00:02, 21.13it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=55.17 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.17 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.14 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.12 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.45it/s]Capturing num tokens (num_tokens=960 avail_mem=55.14 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.45it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=55.13 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.45it/s]Capturing num tokens (num_tokens=896 avail_mem=55.13 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.66it/s]Capturing num tokens (num_tokens=832 avail_mem=55.13 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.66it/s]Capturing num tokens (num_tokens=768 avail_mem=55.13 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.66it/s]Capturing num tokens (num_tokens=704 avail_mem=55.12 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=704 avail_mem=55.12 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.80it/s]Capturing num tokens (num_tokens=640 avail_mem=55.12 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.80it/s]Capturing num tokens (num_tokens=576 avail_mem=55.12 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.80it/s]Capturing num tokens (num_tokens=512 avail_mem=55.10 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.80it/s]

    Capturing num tokens (num_tokens=512 avail_mem=55.10 GB):  50%|█████     | 29/58 [00:01<00:01, 28.11it/s]Capturing num tokens (num_tokens=480 avail_mem=55.12 GB):  50%|█████     | 29/58 [00:01<00:01, 28.11it/s]Capturing num tokens (num_tokens=448 avail_mem=55.12 GB):  50%|█████     | 29/58 [00:01<00:01, 28.11it/s]Capturing num tokens (num_tokens=416 avail_mem=55.12 GB):  50%|█████     | 29/58 [00:01<00:01, 28.11it/s]Capturing num tokens (num_tokens=384 avail_mem=55.11 GB):  50%|█████     | 29/58 [00:01<00:01, 28.11it/s]Capturing num tokens (num_tokens=384 avail_mem=55.11 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=352 avail_mem=55.11 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=320 avail_mem=55.10 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=288 avail_mem=55.10 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=256 avail_mem=55.10 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.09it/s]

    Capturing num tokens (num_tokens=256 avail_mem=55.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.95it/s]Capturing num tokens (num_tokens=240 avail_mem=55.09 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.95it/s]Capturing num tokens (num_tokens=224 avail_mem=55.09 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.95it/s]Capturing num tokens (num_tokens=208 avail_mem=55.09 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.95it/s]Capturing num tokens (num_tokens=192 avail_mem=55.09 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.95it/s]Capturing num tokens (num_tokens=192 avail_mem=55.09 GB):  71%|███████   | 41/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=176 avail_mem=55.08 GB):  71%|███████   | 41/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=160 avail_mem=55.08 GB):  71%|███████   | 41/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=144 avail_mem=55.08 GB):  71%|███████   | 41/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=128 avail_mem=55.08 GB):  71%|███████   | 41/58 [00:01<00:00, 32.93it/s]

    Capturing num tokens (num_tokens=128 avail_mem=55.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=112 avail_mem=55.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=96 avail_mem=55.09 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.70it/s] Capturing num tokens (num_tokens=80 avail_mem=55.09 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=64 avail_mem=55.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=48 avail_mem=55.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=48 avail_mem=55.08 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=32 avail_mem=55.08 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=28 avail_mem=55.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=24 avail_mem=55.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=20 avail_mem=55.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=16 avail_mem=55.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.58it/s]

    Capturing num tokens (num_tokens=16 avail_mem=55.07 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=12 avail_mem=55.06 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=8 avail_mem=55.06 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.64it/s] Capturing num tokens (num_tokens=4 avail_mem=55.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=4 avail_mem=55.05 GB): 100%|██████████| 58/58 [00:01<00:00, 30.63it/s]


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
    Generated text:  Viola. I'm a 13 year old girl from United States. I'm going to the Hope International Middle School. It's a new school, and I don't know what to do. Please help me. Who is the man talking with? [A]. His teacher. [B]. His friend. [C]. His sister. [D]. His brother.
    Answer:
    
    To determine the correct answer, we need to analyze the context provided in the question and consider the roles of the characters mentioned in the dialogue:
    
    1. **Identifying the Context:**
       - The passage is about a 13-year-old girl
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office, and the president of the United States is __________.
    A. the head of government
    B. the head of the executive branch
    C. the head of the legislative branch
    D. the head of the judicial branch
    Answer: B
    
    According to the 'Regulations for the Safe Operation of Ships', the ship's captain shall __________.
    A. Not take over steering duties while engaged in navigation operations
    B. Take over steering duties while engaged in navigation operations
    C. Only take over steering duties when a fault occurs in the ship's operation
    D. Take over steering duties after the ship is fully equipped
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Its best-known landmark is the Eiffel Tower. It is in the Tour de Paris, a 24-kilometer railway that runs through the city. In the 1851, 1854 and 1858, and 1859, the Eiffel Tower was built on three different sites. Now, it has the same height and shape as it was in the 1851.
    
    Since 1925, it has been open daily from 8:00 a.m. to 12:00 midnight, and from 2
    ===============================
    Prompt: The future of AI is
    Generated text:  a web of technology that can create products and services that will change the way we live and work. This is likely to happen through the integration of artificial intelligence with big data, machine learning, and deep learning. These technologies have the potential to revolutionize industries such as healthcare, finance, and transportation.
    However, developing and implementing these technologies can be complex and require significant investment in research, development, and deployment. It is also important to consider the ethical implications of AI and ensure that it is developed and used in a responsible and transparent way.
    In summary, the future of AI is likely to be shaped by the integration of various technologies, with


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for interest in the industry]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [number of years] in the industry, and I'm always eager to learn and improve. I'm a [number of years] in the industry, and I'm always eager to learn and improve. I'm a [number of years] in the industry, and I'm always eager to learn and improve. I'm a [number of years] in the industry, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is the largest city in France and the third-largest city in the world by population. Paris is also the birthplace of many famous French artists, writers, and composers. The city is home to many historical landmarks and museums, including the Notre-Dame Cathedral, the Louvre, and the Musée d'Orsay. Paris is a vibrant and diverse city with a rich cultural heritage that has influenced the development of modern France. It is a popular tourist destination and a major economic center in Europe. The city is also known for its cuisine,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increasing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the potential for AI to be used for malicious purposes.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow for more sophisticated and personalized interactions between humans and machines
    


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
    Generated text:  [Name], and I'm a [job title] at [company name]. I'm always up for a challenge, and I enjoy staying up late researching new ideas. I've always been fascinated by the possibility of learning new languages, and I enjoy sharing my knowledge with others. I'm always looking to improve, and I'm always trying to learn more. I enjoy playing games and exploring new places. What's your name, and what's your job title? It's nice to meet you! [Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. The city is renowned for its rich history, iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum, as well as its vibrant culture and cosmopolitan atmosphere. It is the third most populous city in the world, with around 2. 1 million inhabitants. Paris is known for its artistic, literary, and architectural heritage, as well as its cuisine, music, and fashion. It plays a major role in French politics and culture, and has been the seat of the French government since the French Revolution in 1789. Its status as a UNESCO World Heritage site, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of factors, including advances in machine learning algorithms, the development of new hardware and software technologies, and the increasing complexity of real-world applications. Here are some potential future trends in AI:
    
    1. Increased reliance on AI for critical tasks: As AI systems become more advanced and capable of performing complex tasks, they may become an increasingly important tool for critical decision-making processes, such as healthcare, finance, and transportation.
    
    2. Greater integration of AI into everyday life: AI systems may become more integrated into our everyday lives, from wearable technology to smart homes, as they become more sophisticated and ubiquitous.
    
    3. Increased


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

     am

     a

    /an

     [

    occupation

    ]

     with

     extensive

     experience

     in

     [

    industry

     or

     area

     of

     expertise

    ].

     I

     am

     passionate

     about

     [

    career

     objective

    ]

     and

     my

     journey

     is

     marked

     by

     [

    number

     of

     years

     of

     experience

    ]

     of

     continuous

     learning

     and

     development

    .

     My

     [

    professional

     title

    ]

     combines

     my

     technical

     and

     creative

     skills

    ,

     making

     me

     a

     valuable

     asset

     to

     any

     organization

    .

     I

     am

     always

     eager

     to

     learn

     and

     share

     what

     I

     have

     learned

     with

     others

    ,

     and

     I

     am

     an

     advocate

     for

     [

    cause

     or

     cause

     I

     care

     about

    ].

     How

     would

     you

     describe

     yourself

    ?


    [

    Your

     Name

    ]:

     A

     seasoned

     professional

     with

     a

     diverse

     range

     of

     experiences

    ,

     skills

    ,

     and

     a

     strong

     passion

     for

     innovation

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     and

     the

     capital

     of

     France

    ,

     located

     on

     the

     Se

    ine

     river

     and

     facing

     the

     Mediterranean

     Sea

    ,

     and

     is

     the

     second

     most

     populous

     city

     in

     the

     world

    .

     The

     city

     is

     famous

     for

     its

     historical

     and

     cultural

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     Lou

    vre

     Museum

    ,

     and

     for

     its

     fashion

    ,

     art

    ,

     and

     food

     scene

    .

     It

     is

     also

     an

     important

     center

     of

     science

    ,

     education

    ,

     and

     business

    .

     Paris

     is

     considered

     one

     of

     the

     most

     beautiful

     and

     vibrant

     cities

     in

     the

     world

    .

     It

     was

     named

     UNESCO

     World

     Heritage

     site

     in

     

    1

    9

    9

    7

    .

     The

     city

     has

     a

     rich

     and

     diverse

     cultural

     scene

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     and

     develop

     as

     new

     technologies

     emerge

     and

     breakthrough

    s

     are

     made

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

     Integration

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     into

     everyday

     life

    ,

     from

     self

    -driving

     cars

     to

     personalized

     healthcare

    .
    


    2

    .

     Self

    -L

    earning

    :

     AI

     will

     become

     more

     adept

     at

     learning

     and

     improving

     on

     its

     own

    ,

     without

     human

     intervention

    .
    


    3

    .

     Big

     Data

    :

     AI

     will

     continue

     to

     rely

     on

     large

     amounts

     of

     data

     to

     make

     informed

     decisions

    .
    


    4

    .

     Explain

    ability

    :

     AI

     will

     become

     more

     explain

    able

    ,

     making

     it

     easier

     for

     humans

     to

     understand

     and

     trust

     its

     decisions

    .
    


    5

    .

     Personal

    ization

    :

     AI

     will

     be

     able

     to

     learn

     from

    



```python
llm.shutdown()
```
