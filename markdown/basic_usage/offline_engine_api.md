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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:54,  4.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  9.42it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 32.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.63 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.40it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.60 GB):  31%|███       | 18/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=960 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 29.72it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.41it/s]Capturing num tokens (num_tokens=896 avail_mem=74.61 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.41it/s]Capturing num tokens (num_tokens=832 avail_mem=74.52 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.41it/s]Capturing num tokens (num_tokens=768 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.41it/s]Capturing num tokens (num_tokens=704 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.41it/s]Capturing num tokens (num_tokens=704 avail_mem=74.11 GB):  45%|████▍     | 26/58 [00:00<00:01, 27.92it/s]Capturing num tokens (num_tokens=640 avail_mem=74.02 GB):  45%|████▍     | 26/58 [00:00<00:01, 27.92it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:01, 27.92it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.92it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.92it/s]

    Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.92it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  60%|██████    | 35/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  60%|██████    | 35/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  60%|██████    | 35/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  60%|██████    | 35/58 [00:01<00:00, 33.17it/s]

    Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  60%|██████    | 35/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.25it/s]Capturing num tokens (num_tokens=208 avail_mem=73.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.25it/s]Capturing num tokens (num_tokens=192 avail_mem=73.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.25it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.25it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.25it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.25it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s] Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=32 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.47it/s]

    Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=8 avail_mem=73.86 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.47it/s] Capturing num tokens (num_tokens=8 avail_mem=73.86 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.32it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.32it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 32.53it/s]


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
    Generated text:  Nana, and I am a student at the University of Minnesota. I have a Masters in Public Policy from the University of Minnesota and a bachelor's degree in Political Science from the University of Minnesota. I am an active participant in local and national organizations such as the North American Council on Social Development, the National Women's Law Center, the National Association of Teachers of English, the National Association of Black Journalists, and the American Federation of Teachers. I have published articles in various English language journals and teach English Literature at the University of Minnesota. My passion is to make a difference in the world through my words, and to be a public
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. Every year, he or she is chosen by the people to be the leader of the country. He or she has many important jobs, including being the leader of the country and making sure that the people's lives are safe and the government is running properly. At the end of each year, the president gives a speech called a White House address. He or she asks the people to vote for him or her to be the next president. He or she will then hand out the papers of election to the people, who will select one person to replace him or her. The president is very busy every year. He or
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is also the seat of the government, and it is located on the left bank of the Seine River. This river is the longest in the world and it flows through the city. 
    
    Given the above information, answer the following question: In which direction does the Seine River flow?
    
    The answer is:
    
    North-South. The Seine River flows from the left bank of Paris (in the center of the city) to the right bank of Paris. This means it flows north-south through the city. The river meanders through Paris, connecting various areas, and ultimately empties into the ocean at the Seine
    ===============================
    Prompt: The future of AI is
    Generated text:  in the future.
    AI is the technology that enables machines to perform tasks that require human intelligence and expertise. It is a rapidly advancing field, with new applications emerging regularly and existing applications improving and evolving. It has the potential to transform many industries, industries, and fields of study.
    There are many aspects to AI, including machine learning, natural language processing, computer vision, computer graphics, robotics, and so on. It is a large and complex field, with many different tools and techniques that are used to develop and improve AI systems.
    One of the key benefits of AI is its ability to make decisions and actions based on data and patterns.


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


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is also the oldest continuously inhabited city in Europe, having been inhabited since the Neolithic period. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for art, culture, and commerce in Europe. Paris is home to many famous museums, including the Louvre and the Musée d'Orsay. It is also a major center for the French language and culture, with many French-language schools and cultural institutions. Paris is a popular tourist
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more sophisticated applications in healthcare, such as personalized
    


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
    Generated text:  [Name], and I'm a [Age] year old human. I have a warm smile and a friendly personality, and I love to share my knowledge and insights. My background is in [Field of Study], and I have worked in various industries, but I'm particularly interested in [Specific Area of Interest]. I'm always eager to learn new things and make the world a better place through my work. How can I be a better person for you, my dear friend? I'm always looking for new challenges and opportunities to grow and improve. What do you think? [Name] [Name] Your name is [Name], and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France, located on the Seine River in the northeast of the country. It is the cultural, economic, and political center of France and one of the world's most popular tourist destinations. It is home to numerous world-renowned attractions, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris also has a rich history and is known for its vibrant culture, art, and cuisine. The city is known for its mix of Gothic and modern architecture and is considered a global cultural hub. Paris is a major political and economic hub, with a strong influence on
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and constantly evolving, with new developments and applications emerging regularly. Here are some potential trends that could be observed in the AI field in the coming years:
    
    1. Increased focus on ethical and ethical AI: As AI becomes more widespread, there will be a growing need for ethical considerations. This includes questions around privacy, data security, bias, and the potential misuse of AI. As a result, there may be increased focus on ethical and moral standards in AI development and deployment.
    
    2. AI integration with other technologies: AI is being increasingly integrated with other technologies, such as machine learning, computer vision, and natural language processing. This integration could


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

     Jane

    .

     I

    'm

     a

     dedicated

    ,

     driven

    ,

     and

     talented

     professional

     with

     a

     passion

     for

     innovation

     and

     excellence

    .

     I

     have

     a

     knack

     for

     problem

    -solving

    ,

     and

     I

    'm

     always

     looking

     for

     new

     and

     exciting

     ways

     to

     solve

     challenges

    .

     I

     enjoy

     working

     on

     complex

     projects

     and

     pushing

     the

     boundaries

     of

     what

    's

     possible

    .

     My

     team

     is

     an

     all

    -star

    ,

     and

     we

    're

     constantly

     growing

     our

     skills

     and

     knowledge

    .

     I

    'm

     a

     driven

     leader

     who

     is

     passionate

     about

     making

     a

     positive

     impact

     on

     the

     world

    .

     I

     believe

     in

     creating

     lasting

     change

     and

     I

    'm

     committed

     to

     always

     doing

     my

     best

     work

    .

     I

    'm

     excited

     to

     bring

     my

     unique

     blend

     of

     skills

     and

     passion

     to

     any

     project

    ,

     and

     I

    'm

     confident

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     in

     the

     region

     of

     the

     same

     name

     and

     is

     the

     largest

     city

     in

     France

    .

     It

     is

     also

     one

     of

     the

     most

     popular

     tourist

     destinations

     in

     the

     world

    .

     Paris

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

     lively

     cultural

     scene

    .

     It

     is

     home

     to

     many

     museums

    ,

     art

     galleries

    ,

     and

     theaters

    ,

     as

     well

     as

     many

     famous

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

    .

     Paris

     is

     also

     a

     cultural

     hub

     for

     France

     and

     plays

     an

     important

     role

     in

     the

     country

    's

     economy

    ,

     hosting

     important

     cultural

     events

     such

     as

     the

     Paris

     Review

     and

     the

     European

     Film

     Festival

    .

     Overall

    ,

     Paris

     is

     a

     cultural

     and

     urban

     center

     that

     is

     a

     must

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     progress

     in

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

    .

     These

     technologies

     are

     already

     being

     used

     in

     a

     wide

     range

     of

     applications

    ,

     from

     smart

     home

     devices

     to

     autonomous

     vehicles

    ,

     and

     there

     is

     great

     potential

     for

     continued

     innovation

     in

     the

     coming

     years

    .

     
    


    One

     possible

     trend

     is

     the

     increasing

     reliance

     on

     AI

     for

     decision

    -making

     in

     critical

     areas

     such

     as

     healthcare

    ,

     law

     enforcement

    ,

     and

     transportation

    .

     As

     AI

     becomes

     more

     capable

     of

     understanding

     and

     analyzing

     large

     amounts

     of

     data

    ,

     it

     may

     be

     able

     to

     provide

     more

     accurate

     and

     efficient

     decision

    -making

     processes

    ,

     reducing

     errors

     and

     improving

     outcomes

    .

     Additionally

    ,

     as

     AI

     is

     integrated

     into

     everyday

     products

     and

     services

    ,

    



```python
llm.shutdown()
```
