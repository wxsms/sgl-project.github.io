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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.46it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.57it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.78it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.84it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.84it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.84it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.84it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.84it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.84it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.45it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.34it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 39.15it/s]


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
    Generated text:  Remy and I'm a Software Developer at a global technology company. I'm looking for a way to communicate more effectively with my team members and clients. 
    
    As a software developer, I understand the importance of clear communication, and I would like to use my language skills to convey my ideas in a way that is easily understood by my team members and clients.
    
    I am considering using emojis in my messages and emails to add visual elements to my text, but I'm not sure how to use them effectively.
    
    Is it possible to use emojis in a way that allows for both a visual and written communication? If so, how would I go about
    ===============================
    Prompt: The president of the United States is
    Generated text:  a type of ____
    A. Teacher
    B. Government official
    C. Leader
    D. Leader
    Answer: B
    
    As a middle school student, which of the following statements about the students' safety is incorrect?
    A. Pay attention to physical safety and personal safety
    B. Pay attention to financial safety and property safety
    C. Pay attention to personal safety and psychological safety
    D. Pay attention to safety and security
    Answer: D
    
    Which of the following statements about the red team conducting vulnerability scans on the target system is incorrect?
    A. The scanning results should be kept for at least a month.
    B. The scanning
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the second most populous city in the world and the 16th most populous city in the European Union. Paris is the capital of the Île de la Cité, a walled city dating from the 7th century.
    
    The city of Paris has a history dating back to the times of the Phoenicians and Babylonians. The ancient city of Carthage, a city-state of the ancient Western Mediterranean, stood on the site of the present-day city. The medieval city of Paris was founded by the Frankish Kingdom in the 8th century. Paris was the capital of France from 13
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of researchers
    
    AI is evolving at an incredible pace, with the ability to analyze vast amounts of data to create new insights into human behavior and decision-making. While it has the potential to revolutionize industries such as healthcare, finance, and transportation, it is also creating new problems that researchers must address. In this article, we will explore the future of AI and its impact on society.
    
    Firstly, AI has the potential to improve healthcare in the future. Researchers are working on developing AI-powered tools to help doctors diagnose diseases and predict patient outcomes. For example, AI algorithms can analyze medical images to identify signs of disease, such


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


    Generated text:  Paris, also known as "La Ville Blanche" or "The White City". It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. The city is located on the Seine River and is home to many famous landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower. Paris is known for its rich history, art, and culture, and is a major tourist destination for millions of visitors each year. It is also home to many important institutions of higher education, including the University of Paris and the Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Enhanced privacy and security: As AI becomes more prevalent in our daily lives, there will be increased concerns about privacy and security. AI systems will need to be designed with privacy and security in mind, and there will be a need for robust security measures to protect against cyber threats.
    
    3. Greater use of AI in
    


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
    Generated text:  [Your Name]. I’m a [Type of Character] who enjoys [a personal interest or hobby] and [a personal trait or characteristic]. What can you tell me about yourself? As a [type of character] who enjoys [a personal interest or hobby] and [a personal trait or characteristic], I love spending time with friends and family, exploring new places, and trying new foods. I'm always up for a good laugh and love to learn new things. What do you think makes you unique? As a unique individual, I'm a creative problem-solver who thrives in different environments and has a natural curiosity for new things
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historical landmarks, diverse culture, and famous landmarks such as Notre-Dame Cathedral, the Louvre Museum, and Eiffel Tower.
    The statement is that Paris is the capital city of France and known for its historical landmarks, diverse culture, and famous landmarks such as Notre-Dame Cathedral, the Louvre Museum, and Eiffel Tower. Let me know if you need any further assistance! 
    Paris
    The statement is accurate. Paris is the capital of France and is known for its historical landmarks, diverse culture, and famous landmarks such as Notre-Dame Cathedral, the Louvre Museum, and Eiffel
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be filled with exciting new possibilities and trends. Here are some potential areas of development that are likely to shape the next decade:
    
    1. AI for healthcare: AI is being used in healthcare to improve patient care and outcomes. For example, AI can be used to analyze medical images, detect diseases, and predict patient outcomes. It can also be used to develop personalized treatment plans based on a patient's unique genetic and medical profile.
    
    2. AI for education: AI is being used to personalize learning experiences for students, with applications in education ranging from personalized learning platforms to predictive teaching algorithms.
    
    3. AI for transportation: AI is being used


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

     Alex

     and

     I

     am

     a

     data

     scientist

    .

     I

     specialize

     in

     predictive

     analytics

     and

     have

     been

     working

     in

     this

     field

     for

     the

     past

     five

     years

    .

     I

     have

     a

     passion

     for

     exploring

     complex

     datasets

     and

     using

     advanced

     algorithms

     to

     solve

     complex

     problems

    .

     I

     also

     enjoy

     collaborating

     with

     my

     team

     and

     working

     on

     innovative

     projects

     that

     push

     the

     boundaries

     of

     what

     is

     possible

    .

     What

     other

     role

     do

     you

     work

     in

     or

     do

     you

     have

     any

     other

     interests

    ?

     As

     a

     data

     scientist

    ,

     I

     work

     with

     a

     team

     of

     researchers

     and

     engineers

     to

     analyze

     and

     interpret

     large

     amounts

     of

     data

     to

     uncover

     insights

     that

     can

     inform

     business

     decisions

     and

     drive

     innovation

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     and

     grow

     in

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    France

    's

     capital

     city

    ,

     Paris

    ,

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     It

    's

     home

     to

     the

     Lou

    vre

     Museum

    ,

     one

     of

     the

     largest

     and

     most

     famous

     art

     museums

     in

     the

     world

    ,

     and

     the

     E

    iff

    el

     Tower

    ,

     a

     marvel

     of

     engineering

    .

     The

     city

     is

     also

     famous

     for

     its

     fashion

    ,

     with

     Paris

     Fashion

     Week

     being

     one

     of

     the

     world

    's

     largest

     fashion

     shows

    .

     It

    's

     a

     bustling

     met

    ropolis

     that

    's

     full

     of

     life

     and

     culture

    .

     
    


    Is

     there

     anything

     else

     you

     would

     like

     to

     know

     about

     Paris

    ?

     If

     you

     have

     any

     other

     questions

     about

     Paris

     or

     want

     to

     know

     more

     about

     its

     history

    ,

     culture

    ,

     architecture

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     more

     complex

     and

     interconnected

     than

     it

     is

     today

    .

     Some

     possible

     trends

     include

    :
    


    1

    .

     Increased

     development

     of

     AI

     to

     recognize

     and

     understand

     human

     emotions

     and

     motivations

    ,

     leading

     to

     more

     effective

     emotional

     intelligence

     and

     empathy

    -based

     AI

    .
    


    2

    .

     The

     ability

     to

     create

     and

     control

     AI

     systems

     at

     a

     larger

     scale

    ,

     from

     autonomous

     vehicles

     to

     smart

     homes

     and

     healthcare

     systems

    .
    


    3

    .

     The

     development

     of

     AI

     systems

     that

     can

     learn

     and

     adapt

     to

     new

     situations

     and

     situations

     that

     may

     arise

    .
    


    4

    .

     The

     ability

     to

     create

     more

     ethical

     and

     responsible

     AI

     systems

     that

     follow

     certain

     ethical

     guidelines

     and

     principles

    .
    


    5

    .

     The

     ability

     to

     create

     AI

     systems

     that

     can

     learn

     from

     and

     adapt

     to

     new

     information

     and

     data

    



```python
llm.shutdown()
```
