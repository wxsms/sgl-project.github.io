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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.25it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.54it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 21.64it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 21.64it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 21.64it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 21.64it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 21.64it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 21.64it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.64it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.64it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.64it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.64it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.71it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 39.79it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 39.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.35it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.35it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.35it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.35it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.80it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.80it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.80it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.80it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.80it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.80it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.65it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.65it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.65it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.75it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.29it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.15it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.15it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.15it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.15it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.15it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.15it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.55it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.55it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.55it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.55it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.55it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.55it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 38.47it/s]


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
    Generated text:  Lil and I'm a smart girl, I can speak English, Mandarin, Spanish and Korean. I'm a busy girl. I have a lot of work. I have a lot of exams. I want to be a teacher. But when I'm going to school, I feel very sad. I can't speak English well and I can't understand what my teacher is saying. I'm very sad.
    
    Can you please give me advice on how to improve my speaking English and listening skills so I can be a better teacher? What should I do?
    
    The teacher should be able to understand me and communicate with me. I also want to have
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person who has been in office for a minimum of ____ years and the holder of the highest executive office in the United States.
    A. 20
    B. 15
    C. 10
    D. 5
    Answer:
    D
    
    For treating biliary obstruction, which of the following is the most appropriate choice?
    A. Compound Danshen Injection
    B. Compound Danshen Capsule
    C. Compound Danshen Tablets
    D. Chinese Yam Pill
    E. Compound Danshen Oral Liquid
    Answer:
    C
    
    Based on the patient's blood routine test results, what is the most likely diagnosis?
    A.
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. Rome
    C. London
    D. Moscow
    Answer:
    A
    
    To prevent the safety valve from opening due to the pressure at the top of the tower increasing or the liquid level in the tower increasing, which of the following measures can be taken?
    A. Increase the feed rate
    B. Add a safety valve
    C. Increase the steam pressure
    D. Increase the reflux rate
    Answer:
    B
    
    Which of the following is NOT a correct way to deal with the explosive atmosphere of gas liquefied petroleum gas (LPG)?
    A. Use explosion-proof electrical equipment
    B. Install explosion
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting. We have seen examples of AI that can outperform humans in complex tasks such as chess or robo-assembly. However, what we haven’t seen yet is the ability of AI to interact with and influence human beings. This is something that has fascinated scientists and philosophers alike for years, and what we have now seen is a step forward. This is the first time that AI has been able to understand and respond to the emotions of humans.
    
    The breakthrough was made by researchers at the University of Oxford in the UK. The researchers used a new type of artificial neural network called a generative adversarial network (GAN). GAN


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art scene, and its role in the French Revolution. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is the largest city in France and one of the most visited cities in the world. The city is also home to many international organizations and institutions, including the European Parliament
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in various industries, including manufacturing, healthcare, and transportation. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. There will likely be a push for greater transparency and accountability in AI development and deployment, as well as greater consideration
    


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
    Generated text:  [Name], and I'm a self-made entrepreneur from [Location]. I'm currently living in [City]. I've always been passionate about [Why you started something] and have worked hard to build a strong foundation to achieve success. I'm organized, persistent, and always looking for ways to grow my business. I'm a problem-solver and enjoy problem-solving challenges. My goal is to create a successful and fulfilling life through my entrepreneurial journey. I'm excited to share my experiences and lessons learned with anyone who wants to learn from me. How can I get started with my entrepreneurship? Do you have any tips on how to make the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. It is also a major center for fashion, art, and music, and has been a major trading center for centuries. The city is home to a diverse population, with a population of over 7 million people. Paris is a bustling metropolis with a rich history and culture, and is a popular tourist destination for visitors from around the world. As of 2021, Paris is the world's 10th-largest economy and home to many of the world's top universities. The French capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and involves many variables. However, some possible trends that are commonly discussed include:
    
    1. Increasing use of AI in healthcare: AI has the potential to revolutionize the healthcare industry by improving diagnostic accuracy, personalized treatment plans, and predicting patient outcomes. However, ethical concerns about data privacy and bias must be addressed.
    
    2. Enhanced autonomous vehicles: AI is being developed to improve autonomous vehicle safety and efficiency. However, the integration of AI into driverless cars raises concerns about the safety of self-driving technology and the need for human drivers.
    
    3. More widespread use of AI in finance: AI has the potential to revolutionize the finance industry


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

    ]

     and

     I

    'm

     a

     [

    insert

     age

    ,

     role

    ,

     or

     title

    ]

     who

     enjoys

     [

    insert

     hobby

     or

     activity

    ].

     I

     love

     [

    insert

     something

     that

     brings

     me

     joy

    ],

     and

     I

    'm

     always

     looking

     for

     opportunities

     to

     grow

     and

     improve

     myself

    .

     I

     believe

     that

     being

     a

     good

     listener

     and

     being

     respectful

     of

     others

     is

     crucial

     in

     a

     healthy

     relationship

    .

     I

    'm

     excited

     to

     learn

     more

     about

     you

    ,

     and

     I

     look

     forward

     to

     seeing

     you

    .

     How

     can

     I

     help

     you

     today

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     the

     largest

     and

     most

     famous

     city

     in

     France

    ,

     is

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

    -D

    ame

     Cathedral

    ,

     and

     the

     Sac

    ré

    -C

    œur

     Basil

    ica

    .

     It

     is

     also

     a

     cultural

     and

     economic

     center

    ,

     known

     for

     its

     rich

     history

    ,

     art

    ,

     music

    ,

     and

     fashion

     scene

    .

     Paris

     is

     a

     major

     hub

     for

     international

     politics

     and

     business

    ,

     hosting

     major

     European

     capitals

     and

     being

     a

     popular

     tourist

     destination

    .

     With

     a

     population

     of

     over

     

    2

    .

    5

     million

     people

    ,

     it

    's

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    .

     The

     city

     is

     home

     to

     many

     of

     the

     world

    's

     cultural

     and

     political

     centers

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     challenges

    ,

     and

     there

     are

     many

     different

     areas

     where

     it

     could

     develop

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

     integration

     with

     other

     technologies

    :

     As

     AI

     continues

     to

     advance

    ,

     it

     could

     become

     more

     integrated

     with

     other

     technologies

     such

     as

     blockchain

    ,

     artificial

     intelligence

    ,

     and

     the

     internet

     of

     things

    .

     This

     integration

     could

     lead

     to

     new

     opportunities

     for

     AI

     applications

    ,

     such

     as

     smart

     homes

    ,

     smart

     cities

    ,

     and

     self

    -driving

     cars

    .
    


    2

    .

     Improved

     accuracy

     and

     efficiency

    :

     AI

     can

     be

     used

     to

     improve

     accuracy

     and

     efficiency

     in

     a

     wide

     range

     of

     applications

    ,

     from

     finance

     and

     healthcare

     to

     manufacturing

     and

     transportation

    .

     As

     AI

     continues

     to

     improve

    ,

     it

     could

     be

     used

     to

    



```python
llm.shutdown()
```
