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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]

    Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]

    Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.09it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 29.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.06 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.81it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.01 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.94 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.93 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.88 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.12it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=42.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.12it/s]Capturing num tokens (num_tokens=960 avail_mem=42.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.12it/s] Capturing num tokens (num_tokens=896 avail_mem=42.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.12it/s]Capturing num tokens (num_tokens=832 avail_mem=42.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.12it/s]Capturing num tokens (num_tokens=832 avail_mem=42.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=768 avail_mem=42.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=704 avail_mem=42.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=640 avail_mem=42.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=576 avail_mem=42.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=512 avail_mem=42.66 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=512 avail_mem=42.66 GB):  50%|█████     | 29/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=480 avail_mem=42.68 GB):  50%|█████     | 29/58 [00:00<00:00, 41.27it/s]

    Capturing num tokens (num_tokens=448 avail_mem=42.67 GB):  50%|█████     | 29/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=416 avail_mem=42.67 GB):  50%|█████     | 29/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=384 avail_mem=42.67 GB):  50%|█████     | 29/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=352 avail_mem=42.66 GB):  50%|█████     | 29/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=352 avail_mem=42.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=320 avail_mem=42.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=288 avail_mem=42.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=256 avail_mem=42.65 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=240 avail_mem=42.65 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.87it/s]Capturing num tokens (num_tokens=224 avail_mem=42.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.87it/s]Capturing num tokens (num_tokens=224 avail_mem=42.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=208 avail_mem=42.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.59it/s]

    Capturing num tokens (num_tokens=192 avail_mem=42.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=176 avail_mem=42.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=160 avail_mem=42.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=144 avail_mem=42.63 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=144 avail_mem=42.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=128 avail_mem=42.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=112 avail_mem=42.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=96 avail_mem=42.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.23it/s] Capturing num tokens (num_tokens=80 avail_mem=42.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=64 avail_mem=42.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.23it/s]

    Capturing num tokens (num_tokens=64 avail_mem=42.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=48 avail_mem=42.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=32 avail_mem=42.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=28 avail_mem=42.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=24 avail_mem=42.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=20 avail_mem=42.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=20 avail_mem=42.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.80it/s]Capturing num tokens (num_tokens=16 avail_mem=42.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.80it/s]Capturing num tokens (num_tokens=12 avail_mem=42.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.80it/s]Capturing num tokens (num_tokens=8 avail_mem=42.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.80it/s] Capturing num tokens (num_tokens=4 avail_mem=42.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.80it/s]

    Capturing num tokens (num_tokens=4 avail_mem=42.59 GB): 100%|██████████| 58/58 [00:01<00:00, 38.75it/s]


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
    Generated text:  Tom. I am a teacher of English. I teach many kinds of English, including English for people with disabilities, English for the deaf and hard of hearing, and English for English Language Learners. I teach with the belief that all students can learn English. I am always eager to learn new things and I love helping students learn by asking questions and providing their answers in a way that is understandable to them. I love to travel and make new friends when I have time to do so. What are some ways in which Tom enjoys traveling and making new friends? Tom enjoys traveling and making new friends by taking trips to different places around the world
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing a new policy that will affect millions of people's lives. Despite the complexities of the policy, the president is determined to make it work. The president believes that his policy can be implemented without causing chaos and that it will create jobs and improve the lives of the people who will be affected. However, the implementation of the policy will require a significant amount of money and time, which the president believes is necessary to ensure that the policy will be effective. 
    
    Based on the president's reasoning, what is the likelihood of the policy being effective? To determine the likelihood of the policy being effective, we need to analyze the president's reasoning and
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) Geneva
    C) London
    D) Rome
    E) Berlin
    
    To determine the capital of France, we can look at a common list of capital cities of European countries. The capital cities of France are:
    
    A) Paris
    B) Geneva
    C) London
    D) Rome
    E) Berlin
    
    Based on this list, the correct answer is:
    
    A) Paris
    
    Therefore, the answer is A) Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting! But what are the real risks? Let’s have a look at some of the most significant ones.
    
    ### 1. Global Security Risks
    
    ### 2. AI Bias Risks
    
    ### 3. Privacy Risks
    
    ### 4. Data Privacy Risks
    
    ### 5. Employment Risks
    
    ### 6. AI Safety Risks
    
    ### 7. Cybersecurity Risks
    
    ### 8. Security Risks
    
    ### 9. Cyber Threats
    
    ### 10. Hardware and System Risks
    
    ### 11. Software and Applications Risks
    
    ### 12. Regulatory


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Expertise] who has been [Number of Years] years in the field of [Field of Interest]. I'm passionate about [Why I'm Passionate About This Field]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Favorite Activity] that I enjoy doing, and I'm always looking for ways to incorporate it into my daily routine. I'm a [Favorite Book or Movie] that I read or watch, and I'm always looking for new reads or movies to add to my collection
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower, Notre-Dame Cathedral, and the annual Eiffel Tower Parc de la Villette festival. 
    
    The city is also famous for its rich history, including the Roman Empire, the French Revolution, and the French Revolution itself. Paris is a cultural and artistic hub, with many famous museums, theaters, and art galleries. It is also a major transportation hub, with the Eiffel Tower serving as a symbol of the city's importance in world history. 
    
    Paris is a city of contrasts, with its historic center and modern districts, and its rich history and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater reliance on machine learning: Machine learning is expected to become more prevalent in AI, allowing machines to learn from data and improve their performance over time. This could lead to more efficient and effective AI systems that can adapt
    


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
    Generated text:  ____. My name is ____. My name is ____. I am a ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I am a ____. My name is ____. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the "City of Light" due to its iconic skyline and lively atmosphere. It serves as the political, economic, cultural, and scientific capital of the nation. France's largest city, with a population of around 2.2 million people, is located in the southeast of the country. The city is home to many historical landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its cuisine, music, fashion, and art scene. Its cultural significance and infrastructure of the city continue to evolve and change over time. With its diverse population and rich history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with numerous possible trends shaping the industry. Some of the most promising and imminent trends include:
    
    1. Increased automation and optimization: As machines become more sophisticated, their ability to perform tasks more efficiently and accurately will continue to grow. We'll see more automation in manufacturing, transportation, and other industries, and more automation in AI itself.
    
    2. Enhanced human-machine collaboration: As AI continues to improve, we may see more human-machine collaboration, where machines are involved in decision-making and problem-solving, potentially leading to more efficient and effective human-machine interactions.
    
    3. Ethical AI: With the potential for AI to be used for harmful purposes


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

     am

     [

    Age

    ].

     I

    'm

     a

     [

    Skill

    /

    特长

    ]

     programmer

    ,

     and

     I

    've

     worked

     in

     [

    Current

     job

     or

     company

    ].

     I

    'm

     an

     active

    ,

     collaborative

    ,

     and

     self

    -m

    ot

    ivated

     individual

     who

     thr

    ives

     in

     a

     fast

    -paced

    ,

     high

    -st

    akes

     environment

    .

     I

    'm

     comfortable

     with

     both

     technical

     and

     non

    -

    technical

     communication

    ,

     and

     am

     skilled

     in

     debugging

     and

     troubleshooting

    .

     I

     am

     always

     eager

     to

     learn

     new

     technologies

    ,

     and

     I

    'm

     constantly

     seeking

     out

     new

     challenges

     and

     opportunities

     to

     grow

     as

     a

     programmer

    .

     I

     value

     teamwork

     and

     the

     ability

     to

     collaborate

     with

     other

     developers

     to

     achieve

     common

     goals

    .

     I

     am

     a

     strong

     problem

    -s

    olver

     and

     enjoy

     finding

     creative

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     as

     the

     City

     of

     Light

     and

     is

     home

     to

     numerous

     UNESCO

     World

     Heritage

     sites

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     the

     birth

    place

     of

     the

     French

     Revolution

    ,

     and

     is

     the

     site

     of

     the

     E

    iff

    el

     Tower

    ,

     a

     

    3

    3

    3

    -meter

    -t

    all

     industrial

     steel

     tower

     that

     serves

     as

     the

     centerpiece

     of

     Paris

    .

     In

     addition

    ,

     Paris

     is

     a

     major

     transportation

     hub

     and

     the

     seat

     of

     the

     French

     government

    ,

     and

     is

     the

     largest

     city

     in

     France

     by

     population

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     rich

     history

     and

     culture

    .

     Paris

     has

     a

     strong

     and

     diverse

     cuisine

    ,

     with

     many

     unique

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     speculative

    ,

     but

     here

     are

     some

     possible

     trends

     that

     could

     shape

     its

     development

    :
    


    1

    .

     Increased

     Automation

    :

     As

     AI

     technology

     advances

    ,

     it

     will

     likely

     become

     more

     automated

    ,

     with

     AI

     systems

     taking

     on

     more

     tasks

     that

     typically

     require

     human

     decision

    -making

    .

     This

     could

     lead

     to

     the

     widespread

     adoption

     of

     AI

     in

     various

     industries

    ,

     such

     as

     manufacturing

    ,

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     AI

     Integration

    :

     AI

     will

     likely

     become

     even

     more

     integrated

     into

     our

     daily

     lives

    ,

     with

     more

     of

     its

     applications

     becoming

     available

     to

     us

    .

     This

     could

     include

     the

     use

     of

     AI

     in

     consumer

     goods

    ,

     transportation

    ,

     and

     entertainment

    .
    


    3

    .

     AI

     Ethics

     and

     Regulation

    :

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     there

    



```python
llm.shutdown()
```
