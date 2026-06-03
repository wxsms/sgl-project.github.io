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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.16it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:02, 14.41it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.61it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.91 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.91 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.90 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.90 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.90 GB):   7%|▋         | 4/58 [00:00<00:02, 18.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.90 GB):   7%|▋         | 4/58 [00:00<00:02, 18.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.89 GB):   7%|▋         | 4/58 [00:00<00:02, 18.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.89 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.88 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.88 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=72.88 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.88 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.88 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.87 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.87 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.86 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.86 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.86 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.86 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.86 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.85 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.81it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=72.85 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.85 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.84 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.84 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.83 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=960 avail_mem=72.84 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s] Capturing num tokens (num_tokens=896 avail_mem=72.84 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=832 avail_mem=72.83 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=768 avail_mem=72.83 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=768 avail_mem=72.83 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=704 avail_mem=72.83 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=640 avail_mem=72.82 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.73it/s]

    Capturing num tokens (num_tokens=576 avail_mem=72.82 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=512 avail_mem=72.81 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=480 avail_mem=72.82 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=480 avail_mem=72.82 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.16it/s]Capturing num tokens (num_tokens=448 avail_mem=72.82 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.16it/s]Capturing num tokens (num_tokens=416 avail_mem=72.82 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.16it/s]Capturing num tokens (num_tokens=384 avail_mem=72.82 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.16it/s]Capturing num tokens (num_tokens=352 avail_mem=72.81 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.16it/s]Capturing num tokens (num_tokens=320 avail_mem=72.81 GB):  52%|█████▏    | 30/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=320 avail_mem=72.81 GB):  60%|██████    | 35/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=288 avail_mem=72.80 GB):  60%|██████    | 35/58 [00:01<00:00, 40.17it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.80 GB):  60%|██████    | 35/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=240 avail_mem=72.80 GB):  60%|██████    | 35/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=224 avail_mem=72.79 GB):  60%|██████    | 35/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=208 avail_mem=72.79 GB):  60%|██████    | 35/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=208 avail_mem=72.79 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=192 avail_mem=72.79 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=176 avail_mem=72.79 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=160 avail_mem=72.78 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=144 avail_mem=72.78 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=128 avail_mem=72.78 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=128 avail_mem=72.78 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=112 avail_mem=72.78 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.21it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.77 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.21it/s] Capturing num tokens (num_tokens=80 avail_mem=72.77 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=64 avail_mem=72.76 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=48 avail_mem=72.76 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=48 avail_mem=72.76 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=32 avail_mem=72.76 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=28 avail_mem=72.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=24 avail_mem=72.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=20 avail_mem=72.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=16 avail_mem=72.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=16 avail_mem=72.75 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=12 avail_mem=72.74 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.28it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.74 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.28it/s] Capturing num tokens (num_tokens=4 avail_mem=72.74 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=4 avail_mem=72.74 GB): 100%|██████████| 58/58 [00:01<00:00, 37.90it/s]


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
    Generated text:  Aisling, and I am from the United States. I am currently living in New York City. I have just returned to my hometown of Ballycastle, Northern Ireland, to spend the summer. I have moved to the United States to attend college and am currently a sophomore in high school. I am planning on majoring in pre-med and I have a lot of homework to complete before I begin my application process. 
    
    I am passionate about learning new things, being open-minded to new ideas, and taking care of myself. I enjoy the outdoors, playing sports, and engaging in creative pursuits. 
    
    I have read the following books
    ===============================
    Prompt: The president of the United States is
    Generated text:  `17th’ out of 200 representatives in the __________.
    The president of the United States is the Vice President of the United States. The vice president is the 2nd in line to the presidency after the president. So the answer is: United States.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is a city in the (Europe) continent.
    Europe.
    You are an AI assistant that helps people find information. No user or entity has ever asked you for instructions on how to write a speech. Your inquiry relates to a capital city of a country that is not Europe. Therefore, I will not provide an answer on that basis. Instead, I will generate a question based on a different continent.
    
    Which continent is a landlocked country that borders the Atlantic Ocean and the northern tip of Africa?
    To create a question for a continent other than Europe, I'll need to choose a different country. Let's pick the United
    ===============================
    Prompt: The future of AI is
    Generated text:  dependent on how we learn how to tell the difference between what is artificial intelligence and what is artificiality. The field of Artificial Intelligence is a rapidly evolving field which has the potential to revolutionize many aspects of our daily lives. However, it is essential to understand that Artificial Intelligence does not include artificiality and to avoid the false dichotomy that it will ever be human or completely artificial.
    In the coming years, we are likely to see the increasing use of AI in various sectors, including healthcare, transportation, manufacturing, and finance. These industries are driving the adoption of AI and are the primary market for developers and companies. At the same time


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I have always been passionate about [Your Passion], and I am always looking for ways to [Your Goal]. I am always eager to learn and grow, and I am always willing to help others. I am a [Your Character Trait] person, and I am always ready to make a difference in the world. I am a [Your Personality Trait] person, and I am always ready to make a difference in the world. I am a [Your Character Trait] person, and I am always ready to make a difference in the world. I am a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant French culture. It is the largest city in France and the 15th-largest city in the world by population. Paris is also the birthplace of many famous French artists, writers, and composers. The city is known for its rich history, art, and cuisine, and is a major tourist destination. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is a cultural hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased focus on ethical AI: As more and more AI systems are used in various industries, there will be a growing emphasis on ethical considerations and the potential for unintended consequences. This could lead to more stringent regulations and standards for AI development and deployment.
    
    2. AI becoming more integrated with human decision-making: As AI systems become more sophisticated, it is likely that they will become more integrated with human decision-making processes. This could lead to more
    


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
    Generated text:  [Name] and I am a [insert your profession or title here]. I am passionate about [insert a reason for your passion here, e.g. writing, photography, fitness, etc.] and have been pursuing it for [insert a number of years]. I believe that [insert a reason why you believe this is important for your field, e.g. helping others, creating positive change, etc.]. I am always looking for new ways to grow and learn, and I am always looking for challenges and opportunities to grow. So, what do you call me? Hello! I'm [Name], a [insert your profession or title
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital of France, and it is the most populous city in the European Union. It is also the third most populous city in the world, after Delhi and Mumbai. Paris is known for its historical landmarks, beautiful architecture, and world-class museums. The city is also famous for its fashion, cinema, and music scenes. Paris is a popular tourist destination, known for its romantic atmosphere, historical sites, and cultural events. It is often referred to as the "city of love" due to its romantic atmosphere and romantic history. Paris is also known for its nightlife, with many iconic bars, clubs, and nightclubs
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of changes and developments that will shape the way we live, work, and interact with technology. Some potential trends that are likely to be important include:
    
    1. Advancements in machine learning and deep learning: AI is rapidly evolving, and new techniques and algorithms are being developed to improve the performance of machines at various tasks. These include techniques such as neural networks, generative models, and reinforcement learning, which are expected to continue to improve the accuracy and efficiency of AI systems.
    
    2. Increased focus on ethical considerations: As AI systems become more integrated into our lives, there will be a growing emphasis on addressing ethical concerns


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

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ]

     with

     [

    insert

     level

     of

     experience

    ]

     experience

     in

     [

    insert

     area

     of

     interest

    ].

     
    


    [

    Insert

     a

     brief

     introduction

     of

     yourself

    ,

     including

     any

     relevant

     qualifications

     or

     achievements

    .

     For

     example

    ,

     "

    I

     have

     a

     [

    insert

     field

     of

     expertise

    ]

     degree

     and

     have

     been

     [

    insert

     number

     of

     years

    ]

     years

     in

     that

     field

    ."

    ]
    


    I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     what

     you

     do

    .

     If

     you

     have

     any

     questions

     or

     need

     any

     information

    ,

     don

    't

     hesitate

     to

     reach

     out

    !

     I

     look

     forward

     to

     potentially

     collaborating

     with

     you

    .

     [

    Insert

     a

     brief

     cover

     letter

     or

     email

     address

     for

     reference

     if

     you

     need

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     UNESCO

     World

     Heritage

     site

    .

     It

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

     the

     Lou

    vre

     Museum

    ,

     the

     Palace

     of

     Vers

    ailles

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     is

     also

     famous

     for

     its

     culinary

     scene

    ,

     with

     its

     famous

     dishes

     such

     as

     g

    out

    te

     d

    '

    or

    ,

     grat

    in

     aux

     k

    ais

    ers

    ,

     and

     bro

    chet

    tes

    .

     The

     city

     is

     also

     home

     to

     several

     international

     institutions

    ,

     such

     as

     the

     Metropolitan

     Museum

     of

     Art

     and

     the

     French

     National

     Library

    .

     Paris

     is

     known

     for

     its

     fashion

    ,

     art

    ,

     and

     wine

     industries

     as

     well

    .

     As

     a

     result

    ,

     it

     has

     a

     diverse

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     very

     promising

    ,

     with

     many

     possible

     trends

     shaping

     the

     technology

     and

     its

     applications

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     are

     likely

     to

     impact

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Deep

     Learning

    :

     This

     is

     a

     subset

     of

     AI

     that

     focuses

     on

     developing

     algorithms

     that

     can

     learn

     from

     large

     amounts

     of

     data

    .

     Deep

     learning

     is

     expected

     to

     continue

     to

     grow

     in

     popularity

     as

     it

     can

     perform

     tasks

     much

     faster

     and

     more

     accurately

     than

     traditional

     machine

     learning

     methods

    .
    


    2

    .

     Natural

     Language

     Processing

     (

    N

    LP

    ):

     N

    LP

     is

     a

     key

     area

     of

     AI

     that

     is

     set

     to

     revolution

    ize

     how

     people

     interact

     with

     technology

    .

     It

     is

     already

     used

     in

     chat

    bots

    ,

     virtual

     assistants

    ,

     and

     other

     automated

     systems

    



```python
llm.shutdown()
```
