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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 23.58it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  31%|███       | 18/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 35.58it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=640 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=480 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.20it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.20it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.20it/s]Capturing num tokens (num_tokens=288 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.20it/s]Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.20it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.20it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.30it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=96 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.16it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=48 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.70it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=24 avail_mem=71.62 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=24 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=12 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.10it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.24it/s]


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
    Generated text:  Mike. I am 13 years old and I am a student in Junior High School. I have a new friend named Tom. We are both in the same class. Tom and I both like playing cards. I like playing cards because it is fun and it is easy to learn. But Tom likes it because he can win. He thinks it is exciting to win. One day, Tom and I lost the card game we were playing. I couldn't win because I was not good at playing cards. But Tom won. He said he had won because he thought he was lucky and he thought he was good at playing cards. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. In order to become president, one must be a senator for six years and pass an election, usually in the same year that he or she becomes president. No candidate has won the general election in at least 200 years. One of the most important offices to be president is that of the Supreme Court. The Supreme Court is a very important part of the federal government. It is the highest court in the United States. Since its creation in 1789, it has ruled on important cases. It has also tried to prevent some of the things that happened in the American Civil War. The Supreme
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    Which of the following options best explains the capital of France?
    A. It is located in the North of Europe.
    B. It is the most populous city in the world.
    C. It has a rich history.
    D. It is the largest city in Europe.
    Answer:
    
    C
    
    The following is a multiple-choice question from a Chinese professional examination. Please select the correct answer.
    According to the passage, why is the juice of a banana not considered medicinal?
    A) Because it is often associated with diabetes.
    B) Because it is an alkaloid.
    C) Because it has a high concentration of potassium.
    D) Because it
    ===============================
    Prompt: The future of AI is
    Generated text:  close at hand, and with the advancement of technology, the need for data security has only grown stronger. This makes it necessary for businesses to not only protect their data but also ensure its privacy. In this post, we will discuss the importance of data privacy and security and how businesses can protect their data.
    The Importance of Data Privacy and Security
    Data privacy and security are crucial aspects of any business. They are essential for maintaining the trust of customers and stakeholders, as well as protecting their data from unauthorized access, breaches, and cyberattacks. Data privacy and security can lead to the loss of customer trust, reputational damage, and financial losses


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


    Generated text:  Paris. 
    
    The statement is concise and accurately describes the capital city of France. It provides the name of the city, its location, and its significance in French culture and politics. The statement is brief and easy to understand, making it suitable for a general audience. It also includes the capital city's name, which is a common and widely recognized identifier for French cities. The statement is factual and accurate, providing a clear and unambiguous description of the capital city. The statement is concise and informative, making it suitable for a wide range of readers. The statement is clear and easy to understand, making it suitable for a general audience. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to improve, we can expect to see even more innovative applications emerge, such as more advanced chatbots, more accurate medical diagnoses, and more efficient ways to manage and analyze large amounts of data. Additionally, as AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption of these technologies, leading to a more connected and interconnected world. However, it is important
    


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
    Generated text:  [Name] and I am a [Career] with [Years in the Industry] years of experience. I have always had a passion for [Field of Interest] and have been a true believer in the importance of [Reason]. Despite my experience, I am always looking for ways to [Action]. I am always eager to learn and grow, and I am always seeking out opportunities to contribute to [Project or Cause]. Thank you for having me. How can I assist you today? [Name] [Phone Number] [Email Address] [LinkedIn Profile] [Gmail Profile] [Social Media Links] [Portfolio URL] [About
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is also the largest city in the European Union and the seat of government and culture. The city is known for its rich history, beautiful architecture, and cultural attractions. It is also the most populous city in Europe, with around 6.6 million people living in the metropolitan area. The city is also famous for its music, fashion, and cuisine. Paris is home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. 
    
    The city has a strong tradition of literature and art, with many museums and galleries dedicated to France's art history and culture. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undeniably promising, with potential applications ranging from advanced healthcare and education to autonomous vehicles and even future forms of artificial intelligence that could transform entire industries.
    
    One of the primary trends in AI is the increasing integration of AI into various industries. The internet of things (IoT) and cloud computing have made it possible for AI to be integrated into everyday devices such as smartphones, home appliances, and smart cities. This has opened up new opportunities for AI applications, such as optimizing energy use, improving healthcare outcomes, and enhancing customer experiences.
    
    Another trend is the development of AI that can learn and adapt to new situations without being programmed, known as machine


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

    ’m

     a

     [

    Age

    ]

     year

     old

     who

     live

     in

     [

    Location

    ].

     I

     enjoy

     [

    Inter

    ests

     or

     hobbies

    ].

     I

     am

     a

     [

    Job

     or

     hobby

    ]

     who

     is

     passionate

     about

     [

    Reason

     for

     passion

    ].

     I

     have

     been

     [

    Reason

     for

     starting

    /

    eng

    aging

    /

    finding

     this

     job

     or

     hobby

    ].

     I

     hope

     to

     one

     day

     be

     able

     to

     [

    Goal

     or

     dream

    ].

     I

     hope

     to

     [

    Future

     goal

     or

     dream

    ].

     I

     would

     like

     to

     be

     known

     for

     [

    What

     I

     would

     like

     to

     be

     known

     for

    ].

     How

     can

     I

     help

     someone

    ?

     To

     help

     someone

    ,

     I

     could

     provide

     [

    Service

     or

     Help

    ].

     Can

     you

     tell

     me

     more

     about

     yourself

    ?

     What

     are

     some

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     third

    -largest

     city

     in

     the

     European

     Union

     by

     population

    .

     It

     is

     a

     cosm

    opolitan

     city

     with

     a

     rich

     history

    ,

     art

     scene

    ,

     and

     vibrant

     nightlife

    .

     The

     city

     is

     also

     famous

     for

     its

     iconic

     landmarks

    ,

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     global

     hub

     for

     culture

    ,

     fashion

    ,

     and

     food

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     Its

     status

     as

     the

     capital

     is

     a

     reflection

     of

     France

    ’s

     importance

     in

     the

     world

     and

     its

     influence

     on

     the

     global

     economy

    .

     Despite

     its

     size

    ,

     Paris

     is

     also

     a

     city

     of

     tranqu

    ility

     and

     a

     place

     of

     reflection

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     very

     different

     from

     what

     it

     is

     today

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     AI

     will

     become

     more

     complex

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     an

     increase

     in

     the

     complexity

     of

     AI

     systems

    .

     This

     will

     require

     more

     sophisticated

     algorithms

     and

     models

     to

     handle

     the

     increasing

     data

     and

     interpret

     the

     increasingly

     complex

     interactions

     between

     AI

     systems

     and

     humans

    .
    


    2

    .

     AI

     will

     become

     more

     human

    -like

    :

     AI

     is

     already

     becoming

     more

     human

    -like

     in

     many

     applications

    ,

     such

     as

     through

     natural

     language

     processing

     and

     machine

     learning

    .

     However

    ,

     there

     is

     potential

     for

     AI

     to

     become

     more

     human

    -like

     in

     other

     areas

    ,

     such

     as

     through

     enhanced

    



```python
llm.shutdown()
```
