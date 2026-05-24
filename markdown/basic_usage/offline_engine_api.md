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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.37it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.74it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 21.14it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 29.28it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 29.28it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 35.01it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 35.01it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 35.01it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 35.01it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 35.01it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   9%|▊         | 5/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.88it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.88it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s] Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.34it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.73it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.73it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.73it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.73it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.73it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.73it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  50%|█████     | 29/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  50%|█████     | 29/58 [00:00<00:00, 39.92it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  50%|█████     | 29/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  50%|█████     | 29/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  50%|█████     | 29/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  50%|█████     | 29/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.28it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.28it/s]

    Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.28it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.28it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.28it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.69it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.69it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.69it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.69it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.69it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.69it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=128 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.09it/s] Capturing num tokens (num_tokens=80 avail_mem=72.20 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=80 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=48 avail_mem=71.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=32 avail_mem=71.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.85it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=28 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=24 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=20 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=16 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=12 avail_mem=71.89 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=8 avail_mem=71.89 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.90it/s] Capturing num tokens (num_tokens=8 avail_mem=71.89 GB):  98%|█████████▊| 57/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB):  98%|█████████▊| 57/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB): 100%|██████████| 58/58 [00:01<00:00, 34.68it/s]


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
    Generated text:  Jane and I have just begun my medical studies. Can you please tell me about the process of obtaining a medical degree?
    Certainly, obtaining a medical degree involves several steps. Here's a general overview of the process:
    
    1. **Application for Admission**: You apply for admission to a medical school. This is typically done through your university or a regional medical school.
    
    2. **Pre-admission Preparation**: You usually need to complete certain pre-admission preparation courses. These may include pre-clinical courses, which are focused on learning the basic science of medicine, as well as clinical rotations in various medical specialties.
    
    3. **Application for Admission
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the United States Congress. Is this statement correct? A. Correct B. Incorrect C. Unclear D. Unable to judge
    Answer: A
    
    For individuals who violate laws and regulations, which of the following actions is considered a restraining measure?
    A. Taking them to a designated location for investigation
    B. Detaining them by police force
    C. Removing them from the scene of the violation
    D. Allowing them to remain at home without any restrictions
    Answer: B
    
    Which of the following statements about the 'Four Comprehensives' strategic layout is true?
    A. The 'Four Comprehensives
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lyon
    C. Nice
    D. London
    Answer:
    
    A
    
    In the event of a train or ship collision, the first priority should be:
    A. Saving lives
    B. Protecting property
    C. Ensuring safety of life
    D. Maintaining vessel stability
    Answer:
    
    C
    
    Which of the following statements is true regarding the treatment of hepatocellular jaundice?
    A. It is caused by acute necrosis of liver cells.
    B. It can be treated with oral administration of cholecalciferol.
    C. It is a common complication of viral hepatitis.
    D.
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it's already being felt in your workplace. How can you harness AI's potential to boost productivity and efficiency? Here are some tips to consider.
    
    ### 1. **Enhanced Data Analysis:**
       - **Data Integration:** Ensure that all data sources are integrated to provide a comprehensive view of the workplace. This will help in analyzing trends, patterns, and anomalies in real-time.
       - **Advanced Analytics Tools:** Use advanced analytics tools to extract insights from the data. These tools can help in detecting patterns, predicting outcomes, and automating tasks based on the data.
    
    ### 2. **Improved Decision-Making:


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] with [Number of Wheels] wheels. I'm [Favorite Color] and I love [Favorite Activity/Job]. I'm [Favorite Book/TV Show/Video Game] and I enjoy [Favorite Food/Drink/Activity]. I'm [Favorite Animal/Insect/Plant] and I love [Favorite Music/Artist]. I'm [Favorite Movie/Book/TV Show/Video Game] and I'm [Favorite Place]. I'm [Favorite Sport/Activity/Job]. I'm [Favorite Movie/Book
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major cultural and economic center, with a diverse population and a vibrant nightlife. The city is home to many famous museums, including the Louvre and the Musée d'Orsay, and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its status as the capital of France is a testament to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision, which will enable more sophisticated and complex AI systems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations, such as privacy, bias, and transparency.
    
    3. Increased focus on AI ethics and AI governance: As AI becomes more integrated into our daily lives, there will be a greater focus on AI ethics and AI governance, which will help to
    


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
    Generated text:  [Name], and I'm a [role] [Character]. I'm a [age], [gender], [occupation]. I love [excuse me], and I'm passionate about [a hobby or activity]. If you had the chance to meet me in person, what would you want me to know about me?
    Hello, my name is [Name] and I'm a [role] [Character]. I'm a [age], [gender], [occupation]. I love [excuse me], and I'm passionate about [a hobby or activity]. If you had the chance to meet me in person, what would you want me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Analyze the impact of climate change on Paris's environment. As of 2020, Paris had a relatively low-carbon footprint due to its high efficiency in using renewable energy sources and a wide range of green spaces. However, there have been concerns about the long-term effects of climate change on the city's environment. As the climate warms, the city may experience more extreme weather events such as heatwaves and droughts, which can have significant impacts on public health, infrastructure, and property values. Additionally, rising sea levels and more frequent hurricanes may also threaten the city's coastal areas, leading to loss of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but it is likely to continue to evolve and transform our world in significant ways. Here are some possible future trends in AI:
    
    1. Increased focus on ethical and social implications: As AI becomes more integrated into our daily lives, there will be increased scrutiny of its impact on society. This could lead to new ethical guidelines and regulations to govern its use and development.
    
    2. Rise of autonomous robots: The development of more advanced AI systems that can operate independently without human intervention is expected to revolutionize industries such as manufacturing, healthcare, and transportation.
    
    3. Integration of AI with other technologies: AI will likely become more integrated with other


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    ],

     [

    Job

     Title

    ]

     or

     [

    Name

    ]

     with

     [

    Character

    istics

    ].

     I

     have

     a

     very

     strong

     work

     ethic

     and

     are

     always

     [

    positive

    /

    amb

    itious

    /

    positive

    ]

     about

     my

     career

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     I

    'm

     passionate

     about

     [

    Job

     Title

    ],

     [

    Name

    ]

     or

     whatever

     my

     current

     occupation

     is

    .

     I

     have

     a

     positive

     attitude

     and

     try

     my

     best

     to

     achieve

     my

     goals

    .

     I

    'm

     always

     eager

     to

     learn

     and

     take

     on

     new

     things

    .

     I

    'm

     always

     willing

     to

     go

     above

     and

     beyond

     to

     do

     what

    's

     necessary

     to

     get

     my

     job

     done

    .

     I

     have

     a

     strong

     work

     ethic

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     True

    


    B

    .

     False

    


    A

    .

     True

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

     and

     there

     are

     many

     potential

     areas

     where

     it

     is

     expected

     to

     advance

    .

     Here

     are

     some

     potential

     trends

     to

     keep

     in

     mind

    :
    


    1

    .

     Autonomous

     vehicles

    :

     As

     the

     technology

     advances

    ,

     autonomous

     vehicles

     are

     becoming

     more

     feasible

    .

     This

     could

     lead

     to

     a

     decrease

     in

     traffic

     accidents

     and

     an

     increase

     in

     efficiency

    ,

     which

     could

     have

     a

     ripple

     effect

     on

     society

    .
    


    2

    .

     Artificial

     general

     intelligence

    :

     AI

     that

     can

     think

     and

     learn

     on

     its

     own

     could

     revolution

    ize

     many

     industries

    ,

     from

     healthcare

     to

     finance

    .

     This

     would

     require

     a

     significant

     investment

     in

     AI

     research

     and

     development

    .
    


    3

    .

     Virtual

     and

     augmented

     reality

    :

     As

     these

     technologies

     become

     more

     advanced

    ,

     we

     can

     expect

     to

     see

     more

     immersive

     experiences

     in

     our

    



```python
llm.shutdown()
```
