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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.70it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.70it/s]


    2026-05-03 16:31:58,311 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 16:31:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.92it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:02, 13.87it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.07 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.02 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.02 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=960 avail_mem=58.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=58.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=832 avail_mem=58.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=832 avail_mem=58.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=768 avail_mem=58.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=704 avail_mem=58.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=640 avail_mem=58.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=576 avail_mem=58.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=512 avail_mem=57.98 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=512 avail_mem=57.98 GB):  50%|█████     | 29/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=480 avail_mem=58.00 GB):  50%|█████     | 29/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=448 avail_mem=58.00 GB):  50%|█████     | 29/58 [00:00<00:00, 41.55it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.00 GB):  50%|█████     | 29/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=384 avail_mem=57.99 GB):  50%|█████     | 29/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=352 avail_mem=57.99 GB):  50%|█████     | 29/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=352 avail_mem=57.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=320 avail_mem=57.98 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=288 avail_mem=57.98 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=256 avail_mem=57.98 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=240 avail_mem=57.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.59it/s]Capturing num tokens (num_tokens=224 avail_mem=57.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.59it/s]Capturing num tokens (num_tokens=224 avail_mem=57.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=208 avail_mem=57.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.74it/s]

    Capturing num tokens (num_tokens=192 avail_mem=57.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=176 avail_mem=57.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=160 avail_mem=57.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=144 avail_mem=57.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=144 avail_mem=57.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=128 avail_mem=57.95 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=112 avail_mem=57.95 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=96 avail_mem=57.95 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.47it/s] Capturing num tokens (num_tokens=80 avail_mem=57.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=64 avail_mem=57.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=64 avail_mem=57.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=48 avail_mem=57.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.29it/s]

    Capturing num tokens (num_tokens=32 avail_mem=57.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=28 avail_mem=57.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=24 avail_mem=57.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=20 avail_mem=57.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=20 avail_mem=57.92 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=16 avail_mem=57.92 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=12 avail_mem=57.92 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=8 avail_mem=57.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.01it/s] Capturing num tokens (num_tokens=4 avail_mem=57.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=4 avail_mem=57.91 GB): 100%|██████████| 58/58 [00:01<00:00, 39.36it/s]


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
    Generated text:  Agnes and I will be the new student at the English club. I have been a member of the club for three years and have had lots of fun and I really enjoy English Literature. I have done many English Literature activities such as reading aloud, listening to my own and others’ readings, writing my own essays, giving presentations on various topics and even going to the university library. I do not think that I can keep up with all the readings that my classmates do because there are so many books available, so I think it is important to have some time to read and enjoy myself. The English club has also taught me some other things
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is like a king. The president has two main jobs. He makes decisions for the country. And he helps the country by going to war. The president is very important because he makes sure that all of the country's needs are met. There is also a vice president. He is like a helper to the president. The vice president works with the president and they work together to make decisions. The vice president also helps the president go to war. The vice president is like a helper. He makes sure that the country is safe. Some other important people help the president. They are the top of the government
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    [Answer: A]
    
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    
    The capital of France is Paris. Therefore, the correct answer is A. Paris. However, since the question asks for the capital of France and typically capital cities have a more complex political and historical context, Paris would also be the correct answer for a more general answer about French cities. Given the choices provided, Paris is the most accurate option in this context.
    
    To summarize:
    - A. Paris
    - B. London
    - C. Berlin
    - D. Moscow
    
    Given the context, the correct answer is:
    A.
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and with it comes the rise of cyber threats. In fact, cyber attacks have been increasing by 72% in 2020 alone, according to a report from the Cybersecurity and Infrastructure Security Agency (CISA). Cybersecurity experts say the number of attacks on the internet is growing because cybercriminals are getting more sophisticated at creating new vulnerabilities.
    What’s more, these cyberattacks are becoming more targeted, meaning the cybercriminals are looking for specific information that can be used to launch a new attack. In fact, a recent report from the National Institute of Standards and Technology (NIST) found that cyber


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic center, with a rich history dating back to ancient times and a modern city that is home to many world-renowned museums, art galleries, and restaurants. Paris is a popular tourist destination, with millions of visitors each year, and is known for its fashion, food, and wine. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. The city is a major hub for business and commerce, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more sophisticated and accurate AI systems being used in healthcare, such as in diagnosing diseases, predicting patient outcomes, and managing complex medical cases.
    
    
    


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
    Generated text:  __________ and I am a/an _____________. I am a/an _____________ whose ___________ is ___________.
    
    I want to start with a little bit of background information to help you get to know me a bit better. What is the first thing you would like to tell us about yourself? In your mind's eye, a picture comes to mind of a certain person. They walk into a room full of people who have very different jobs and professions, and yet they are all able to share something common about themselves. What is it? A) The average person who enjoys drinking coffee and listening to music. B)
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement provided provides a brief summary of Paris's capital city, indicating that it is the capital of France. The statement can be formatted as follows:
    
    "Paris is the capital of France." 
    
    This concise statement encapsulates the core information about Paris's location as the capital city of France, leaving no ambiguity about its status as the capital of the country. 
    
    For a more detailed version or different information, please provide additional context or clarification. Here's an example using a slightly different wording for variety:
    
    "Paris is the capital of France, the capital city of which is the seat of the French government." 
    
    In this case
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  diverse and complex, with many potential trends that could shape the field in the years to come. Here are some possible trends that experts believe will drive future progress in AI:
    
    1. Increased focus on ethical AI: As more and more data is generated, ethical considerations are becoming more important. AI systems that are designed with ethical implications in mind will be in high demand in the future. Governments and organizations will likely be more cautious about how they use AI, and there will be more emphasis on creating systems that are transparent, accountable, and responsible.
    
    2. Enhanced machine learning capabilities: Machine learning is one of the key areas of focus for AI research


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

     ____

     and

     I

     am a

    /an

     ____

    _.

     May

     I

     ask

     what

     kind

     of

     job

     or

     hobby

     you

     have

    ?


    Hello

    ,

     my

     name

     is

     __

     and

     I

     am

     a

    /an

     __

    _.

     May

     I

     ask

     what

     kind

     of

     job

     or

     hobby

     you

     have

    ?

     Hi

    ,

     my

     name

     is

     __

     and

     I

    'm

     a

    /an

     ____

    _.

     May

     I

     ask

     what

     kind

     of

     job

     or

     hobby

     you

     have

    ?

     Hello

    ,

     my

     name

     is

     __

     and

     I

     am

     a

    /an

     __

    _.

     May

     I

     ask

     what

     kind

     of

     job

     or

     hobby

     you

     have

    ?

     Hi

    ,

     my

     name

     is

     __

     and

     I

    'm

     a

    /an

     __

    _.

     May

     I

     ask

     what

     kind

     of

     job

     or

     hobby

     you

     have

    ?

     Hello

    ,

     my

     name

     is

     __

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     museums

    ,

     and

     annual

     world

     music

     festival

    .
    


    Paris

     is

     a

     bustling

     city

     in

     central

     France

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     museums

    ,

     and

     annual

     world

     music

     festival

    .

     Its

     charming

     streets

    ,

     rich

     history

    ,

     and

     vibrant

     culture

     make

     it

     a

     popular

     tourist

     destination

    .

     The

     city

     is

     also

     home

     to

     numerous

     museums

    ,

     including

     the

     Lou

    vre

    ,

     and

     has

     a

     thriving

     music

     scene

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     a

     lively

     nightlife

     and

     cultural

     activities

    ,

     making

     it

     a

     popular

     tourist

     destination

     for

     its

     entertainment

    ,

     cuisine

    ,

     and

     heritage

    .

     Its

     rich

     history

     and

     elegant

     architecture

     make

     it

     a

     UNESCO

     World

     Heritage

     site

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advances

     in

     areas

     such

     as

     machine

     learning

    ,

     deep

     learning

    ,

     natural

     language

     processing

    ,

     and

     robotics

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     privacy

     concerns

    :

     As

     AI

     becomes

     more

     integrated

     into

     everyday

     life

    ,

     there

     will

     be

     an

     increasing

     need

     for

     robust

     privacy

     protections

    .

     Developers

     will

     need

     to

     create

     ethical

     guidelines

     for

     AI

     systems

    ,

     and

     consumers

     will

     be

     more

     concerned

     about

     the

     data

     they

     share

     with

     AI

    -powered

     systems

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     integrate

     with

     other

     technologies

    ,

     such

     as

     IoT

    ,

     blockchain

    ,

     and

     cloud

     computing

    .

     This

     integration

     will

     create

     new

     business

     models

     and

     challenges

     for

     businesses

     that

     rely

     on

     AI

    .
    


    3

    



```python
llm.shutdown()
```
