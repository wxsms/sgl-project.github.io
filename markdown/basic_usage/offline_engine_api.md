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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.11it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.24it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.21it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.06it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.18 GB):   3%|▎         | 2/58 [00:00<00:03, 17.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.17 GB):   3%|▎         | 2/58 [00:00<00:03, 17.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.17 GB):   3%|▎         | 2/58 [00:00<00:03, 17.45it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.17 GB):   3%|▎         | 2/58 [00:00<00:03, 17.45it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.17 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.16 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.15 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.15 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.15 GB):   9%|▊         | 5/58 [00:00<00:02, 20.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.48it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=55.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.12 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.12 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.12 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.10 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]

    Capturing num tokens (num_tokens=960 avail_mem=55.11 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s] Capturing num tokens (num_tokens=896 avail_mem=55.11 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]Capturing num tokens (num_tokens=896 avail_mem=55.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=832 avail_mem=55.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=768 avail_mem=55.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=704 avail_mem=55.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=640 avail_mem=55.09 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=576 avail_mem=55.09 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=576 avail_mem=55.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=512 avail_mem=55.08 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=480 avail_mem=55.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=448 avail_mem=55.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.73it/s]

    Capturing num tokens (num_tokens=416 avail_mem=55.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=384 avail_mem=55.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=384 avail_mem=55.09 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=352 avail_mem=55.08 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=320 avail_mem=55.07 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=288 avail_mem=55.07 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=256 avail_mem=55.07 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=240 avail_mem=55.07 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=240 avail_mem=55.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=224 avail_mem=55.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=208 avail_mem=55.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.54it/s]

    Capturing num tokens (num_tokens=192 avail_mem=55.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=176 avail_mem=55.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=160 avail_mem=55.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=160 avail_mem=55.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=144 avail_mem=55.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=128 avail_mem=55.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=112 avail_mem=58.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.92it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.92it/s] Capturing num tokens (num_tokens=80 avail_mem=58.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=80 avail_mem=58.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=64 avail_mem=58.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=48 avail_mem=58.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=32 avail_mem=58.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=28 avail_mem=58.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=24 avail_mem=58.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=24 avail_mem=58.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=20 avail_mem=58.89 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=16 avail_mem=58.89 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=12 avail_mem=58.89 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.49it/s]

    Capturing num tokens (num_tokens=8 avail_mem=58.89 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.49it/s] Capturing num tokens (num_tokens=4 avail_mem=58.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.49it/s]Capturing num tokens (num_tokens=4 avail_mem=58.88 GB): 100%|██████████| 58/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=4 avail_mem=58.88 GB): 100%|██████████| 58/58 [00:01<00:00, 37.83it/s]


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
    Generated text:  Tessa and I am from the United Kingdom. I am an English teacher at Haverfordwest High School. My daughter is currently at the same school as me. She is a 10th grade student. My daughter loves to write poetry and she is the lead singer of a band called Harvest. I am currently looking for creative ways to get the message of our school into the minds of our students. I have found that many students are struggling with their grades and I want to help them improve. I have been looking at a number of different topics and one of them is how to improve reading skills. I have read lots of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she has to deal with a lot of different jobs. Each of those jobs have different rules and responsibilities. This article will list down the different rules and responsibilities of the President of the United States. These rules and responsibilities are also called duties.
    The duties of the President of the United States are very important. He or she is the head of the government. He or she is responsible for creating the policy of the country. This means that he or she has to make the decisions. This article will list down the duties of the President of the United States.
    The president of the United States is responsible for preparing
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Brussels
    C. Brussels
    D. Lyon
    Answer:
    
    A
    
    Which of the following statements about the structure and function of the human musculoskeletal system is correct? A. The bones in the human body are composed of organic matter. B. The head of the spinal cord is located at the junction of the first and second cervical vertebrae. C. The limbs are primarily composed of muscles. D. The heart is the organ that controls the blood flow.
    Answer:
    
    C
    
    If the first card is red, the second card is blue, and the third card is black, then the
    ===============================
    Prompt: The future of AI is
    Generated text:  always evolving, with the potential to transform the way we work, communicate, and interact with technology. As AI technology continues to advance, so too do the challenges it presents. One of the biggest challenges facing AI is the question of fairness and bias. AI systems are often trained on biased data, which can lead to unintended consequences such as discrimination or unfair treatment of certain groups of people.
    To address this challenge, researchers are developing new techniques and methods to ensure that AI systems are fair and unbiased. One approach is to use a technique called adversarial training, which involves adding adversarial perturbations to the training data to make the system more


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


    Generated text:  Paris. It is the largest city in Europe and the third largest city in the world. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also a major center for art, music, and literature, and is a popular tourist destination for millions of visitors each year. The city is known for its annual Eiffel Tower Festival and its annual fashion week. Paris is a city of contrasts, with its modern skyscrapers and historical architecture blending seamlessly into the surrounding landscape.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in various industries. This could lead to increased efficiency, productivity, and cost savings for businesses and individuals.
    
    2. Improved privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This could lead to increased regulations and standards for AI systems, as well as more robust measures to protect user data and prevent cyber
    


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
    Generated text:  Jane, and I'm a beginner in the art of making jello.
    
    Jane, the self-introduction goes.
    
    Jane, the introverted, introverted artist, (reading like a book title) has a passion for making jello, a tradition of the Philippines. Her unique skill lies in crafting exquisite jellies, luscious candies, and charming presents, often creating limited edition collections that sell like hotcakes. Jane's sense of humor and artistic vision shine through in her creations. She's been practicing jello for years, and enjoys sharing her craft with the world. What kind of role is Jane in? Jane is a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its cuisine is popular throughout the world and its cultural institutions include the Louvre and Notre-Dame Cathedral. The city is known for its romantic and picturesque landscapes, especially during the summer months. The city is also home to many museums and public spaces. Paris is the world's third-largest city by population and is renowned for its historical architecture, art, and music. In 2010, Paris had a population of approximately 2.3 million people. Despite its size, Paris is often described as the most desirable city in the world for residents and visitors alike.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and highly promising, and there are many different directions that it may take. Here are a few possible trends that could be expected in the near future:
    
    1. Increased integration with human decision-making: One of the most promising trends is the increasing integration of AI with human decision-making. This could lead to more intelligent and nuanced AI that can make better-informed decisions based on multiple sources of information and human intuition.
    
    2. More ethical and responsible AI: As more and more AI systems become integrated into our daily lives, there will be a growing demand for systems that are designed to be more ethical and responsible. This could include designing AI


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

     __

    ________

    ,

     and

     I

     am

     an

     AI

     language

     model

    .

     My

     purpose

     is

     to

     assist

     and

     provide

     information

     on

     a

     wide

     range

     of

     topics

    .

     I

     was

     created

     by

     Alibaba

     Cloud

    ,

     and

     I

     am

     currently

     in

     the

     process

     of

     being

     trained

     on

     a

     massive

     dataset

     of

     texts

    .

     If

     you

     have

     any

     questions

     or

     need

     assistance

     with

     anything

    ,

     don

    't

     hesitate

     to

     reach

     out

    .

     Thank

     you

     for

     your

     time

     and

     I

     look

     forward

     to

     helping

     you

    !

     [

    Your

     name

    ]

     (

    Optional

    ):

     Hello

    ,

     my

     name

     is

     [

    Your

     name

    ],

     and

     I

     am

     an

     AI

     language

     model

    .

     My

     purpose

     is

     to

     assist

     and

     provide

     information

     on

     a

     wide

     range

     of

     topics

    .

     I

     was

     created

     by

     Alibaba

     Cloud

    ,

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     on

     the

     French

     Riv

    iera

     and

     is

     one

     of

     the

     world

    's

     most

     important

     cultural

     and

     economic

     centers

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

    ,

     and

     is

     famous

     for

     its

     vibrant

     arts

    ,

     gastr

    onomy

    ,

     and

     fashion

     scene

    .

     The

     city

     also

     has

     a

     large

    ,

     diverse

     population

    ,

     with

     many

     different

     ethnic groups

     and

     cultures

    .

     Paris

     is

     also

     home

     to

     some

     of

     the

     world

    's

     most

     renowned

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Centre

     Pom

    pid

    ou

    .

     Overall

    ,

     Paris

     is

     a

     truly

     unique

     and

     fascinating

     city

     with

     a

     rich

     history

     and

     a

     vibrant

     culture

    .

     
    


    In

     summary

    ,

     Paris

     is

     a

     major

     French

     city

     located

     on

     the

     French

     Riv

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     can

     be

     influenced

     by

     a

     variety

     of

     factors

    .

     However

    ,

     here

     are

     some

     potential

     trends

     that

     are

     currently

     being

     explored

     or

     predicted

     by

     researchers

     and

     industry

     experts

    :
    


    1

    .

     Adv

    ancements

     in

     computer

     hardware

    :

     With

     advancements

     in

     computing

     power

     and

     storage

     capacity

    ,

     we

     can

     expect

     to

     see

     improvements

     in

     AI

     systems

     that

     are

     more

     efficient

    ,

     faster

    ,

     and

     more

     powerful

    .

     This

     could

     lead

     to

     significant

     improvements

     in

     AI

     capabilities

     and

     applications

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     As

     more

     data

     is

     collected

     and

     analyzed

    ,

     AI

     is

     expected

     to

     play

     an

     increasingly

     important

     role

     in

     healthcare

    .

     This

     could

     lead

     to

     improvements

     in

     diagnosis

    ,

     treatment

    ,

     and

     patient

     care

    .
    


    3

    .

     Integration

     of

    



```python
llm.shutdown()
```
