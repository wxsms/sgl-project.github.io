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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.56it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.53it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.61it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:04, 11.89it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.40 GB):   3%|▎         | 2/58 [00:00<00:04, 11.89it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.40 GB):   3%|▎         | 2/58 [00:00<00:04, 11.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.40 GB):   7%|▋         | 4/58 [00:00<00:04, 12.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.40 GB):   7%|▋         | 4/58 [00:00<00:04, 12.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.30 GB):   7%|▋         | 4/58 [00:00<00:04, 12.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.29 GB):   7%|▋         | 4/58 [00:00<00:04, 12.97it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.29 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.88 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.79 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.72 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.71 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.71 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.26it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.71 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.71 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.26it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.71 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.70 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.26it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=73.70 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.69 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.69 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.69 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.88it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.67 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.88it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.67 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=960 avail_mem=73.68 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s] Capturing num tokens (num_tokens=896 avail_mem=73.68 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=832 avail_mem=73.68 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=768 avail_mem=73.67 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]

    Capturing num tokens (num_tokens=704 avail_mem=73.67 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=704 avail_mem=73.67 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=640 avail_mem=73.67 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=576 avail_mem=73.67 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=512 avail_mem=73.65 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=480 avail_mem=73.67 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=448 avail_mem=73.66 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=448 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.74it/s]Capturing num tokens (num_tokens=416 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.74it/s]Capturing num tokens (num_tokens=384 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=352 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=320 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.74it/s]

    Capturing num tokens (num_tokens=288 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=288 avail_mem=73.65 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=240 avail_mem=73.64 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=224 avail_mem=73.64 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=208 avail_mem=73.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=192 avail_mem=73.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=192 avail_mem=73.63 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=176 avail_mem=73.63 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=128 avail_mem=73.62 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=96 avail_mem=73.62 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.85it/s] Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.03it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.03it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.03it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.03it/s]

    Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.03it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.03it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.21it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.21it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.21it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:01<00:00, 31.62it/s]


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
    Generated text:  Tristan. I'm a health and wellness professional, and I enjoy helping people understand and control their health.
    
    ### What are my health goals?
    
    I am always asking patients what they want to improve in their life. Common goals include:
    
    • Losing weight
    
    • Building strength or endurance
    
    • Improving balance or flexibility
    
    • Getting into better physical shape
    
    • Getting better sleep
    
    • Getting more energy
    
    • Improving eating habits
    
    • Quitting smoking
    
    • Lowering blood pressure
    
    • Removing toxins from the body
    
    • Getting more vitamins and minerals
    
    ### What are your tips for me?
    
    When it comes to improving my own health,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of which of the following ____.
    A. State Council
    B. Supreme Court
    C. National Security Council
    D. House of Representatives
    Answer:
    C
    
    In China, the highest organ of state power is the ____.
    A. National People's Congress
    B. Central Military Commission
    C. Supreme Court
    D. State Council
    Answer:
    A
    
    To ensure the safety of the uplink links of 5G networks, which technology is used to prevent unauthorized access?
    A. PN
    B. QoS
    C. Encryption
    D. DDoS
    Answer:
    D
    
    The core of the prosperity and
    ===============================
    Prompt: The capital of France is
    Generated text:  located at the ____
    A. Atlantic Ocean
    B. Mediterranean Sea
    C. Atlantic Ocean, Mediterranean Sea
    D. Pacific Ocean
    Answer: C
    
    Which of the following is NOT a non-renewable resource?
    A. Oil
    B. Coal
    C. Wind power
    D. Natural gas
    Answer: C
    
    In the "Terror of 2004", which country was the British Prime Minister not elected for?
    A. United Kingdom
    B. United States
    C. France
    D. China
    Answer: D
    
    After the end of World War II, which organization was established to maintain international peace
    ===============================
    Prompt: The future of AI is
    Generated text:  a digital future. The future of AI is a digital future. The future of AI is a digital future. A. 错误 B. 正确
    正确
    
    1602、云服务的生命周期可以分为使用阶段、运行阶段、部署阶段、运维阶段和停运阶段。
    正确
    
    以下关于影响搜索引擎排名的因素的描述不正确的是（）。 A：网站质量越高，排名越高 B：网站发布的时间越早，排名越高 C：网站发布的时间越晚，排名越低 D：网站的更新速度越快，排名越高
    C 解析：网站发布的时间越早，


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your profession or experience here]. I enjoy [insert a short description of your hobbies or interests here]. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity here]. What's your favorite book or movie? I love [insert a short description of your favorite book or movie here]. What's your favorite color? I love [insert a short description of your favorite color
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Library. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. The city is known for its fashion, art, and cuisine, and is a major hub for business and commerce in Europe. It is also home to many international organizations and institutions, including the European Parliament and the European Central Bank. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the urban landscape. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be a greater emphasis on ensuring that they are used ethically and responsibly. This may involve developing guidelines and standards for how AI systems should be used, and ensuring that they are not used for harmful purposes.
    
    2. Greater integration with human decision
    


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
    Generated text:  [Name] and I’m a [Job Title]. I’m excited to have the opportunity to work with [Company Name]. If you have any questions or need help, please don’t hesitate to reach out. I’m always looking to learn and grow and look forward to our future together. [Name] (Type in your name if you prefer not to use your real name)
    
    ---
    
    **Personal Information**
    
    - **Name:** [Name]  
    - **Job Title:** [Job Title]  
    - **Company Name:** [Company Name]  
    - **LinkedIn:** [LinkedIn Profile] (Optional, for LinkedIn connections)  
    - **Email
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city and the most populous metropolitan area in France, located on the Île de France, on the River Seine. It is the cultural, economic and political centre of France. It is also known as "la Ville de Paris" in French and "État d'Or" in English.
    
    The city is the seat of the French government and the chief administrative centre of France, and is the largest city in Europe. It is the economic and cultural centre of France and also the cultural capital of the world. The French capital has a metropolitan area of around 1 million inhabitants and is ranked 4th
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends, including:
    
    1. Improved accuracy and efficiency: AI is increasingly capable of processing large amounts of data faster and more accurately than human beings, making it more efficient and reliable in various applications.
    
    2. Increased human involvement: As AI continues to develop, it may become more human-like, with more interaction between humans and AI systems, leading to a more intuitive and responsive user experience.
    
    3. Integration of AI with other technologies: AI is likely to become more integrated with other technologies, such as robotics and autonomous vehicles, leading to more complex and sophisticated applications.
    
    4. Development of AI ethics and privacy concerns


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

    ]

     and

     I

    'm

     a

     [

    occupation

    ]

     who

     has

     been

     [

    number

     of

     years

     in

     the

     industry

    ]

     years

     in

     the

     industry

    .

     I

    'm

     passionate

     about

     [

    describe

     your

     passion

     or

     interest

    ].

     In

     your

     view

    ,

     what

     are

     the

     biggest

     challenges

     in

     your

     industry

    ,

     and

     how

     do

     you

     approach

     them

    ?

     I

    'm

     always

     eager

     to

     learn

     and

     expand

     my

     knowledge

    ,

     so

     please

     feel

     free

     to

     ask

     any

     questions

     you

     might

     have

    !

     What

    's

     your

     favorite

     movie

     or

     book

     of

     all

     time

    ,

     and

     why

    ?

     Your

     answer

     should

     be

     concise

     and

     relevant

    .

     As

     a

     [

    Type

     of

     avatar

    ]

     I

     will

     be

     able

     to

     control

     my

     thoughts

     and

     emotions

    ,

     allowing

     me

     to

     make

     choices

     that

     align

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    The

     statement

     can

     be

     summarized

     as

    :

     Paris

     is

     the

     capital

     city

     of

     France

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     evolve

     in

     many

     different

     directions

    ,

     but

     some

     key

     trends

     are

     likely

     to

     be

     key

     drivers

     of

     change

     in

     the

     coming

     years

    .
    


    1

    .

     Increased

     AI

     ethics

     and

     transparency

    :

     As

     AI

     continues

     to

     become

     more

     integrated

     into

     our

     lives

    ,

     there

     will

     be

     an

     increased

     emphasis

     on

     developing

     ethical

     guidelines

     and

     transparency

    .

     This

     will

     require

     that

     AI

     is

     transparent

     and

     able

     to

     communicate

     its

     decisions

     in

     clear

     and

     understandable

     ways

    ,

     and

     that

     it

     is

     designed

     with

     the

     needs

     of

     users

     and

     society

     in

     mind

    .
    


    2

    .

     Adv

    ancements

     in

     AI

     technology

    :

     There

     are

     currently

     many

     exciting

     advances

     in

     AI

     technology

    ,

     including

     the

     development

     of

     models

     that

     can

     perform

     tasks

     that

     were

     previously

     thought

     impossible

    .

     These

     include

    



```python
llm.shutdown()
```
