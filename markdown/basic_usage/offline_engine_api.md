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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.10it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.22it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 22.70it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.48it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.33it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.62it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.62it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.62it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.62it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.98it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 46.17it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 46.17it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 46.17it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 46.17it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 46.17it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 46.17it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 46.17it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  71%|███████   | 41/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.42it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.82it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.60it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.60it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.60it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.60it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.60it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.60it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.92it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 42.11it/s]


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
    Generated text:  Abigail and I am 37. I am a lifelong member of the Church of Jesus Christ of Latter-day Saints. I have always loved to serve the Lord and to do what I could to serve others. I love to learn about the Bible, ancient history, and the natural world and I enjoy traveling to other places. I also enjoy reading books and listening to music. I am a non-denominational Methodist and I have been in the Church for 10 years. I enjoy the church experience and I am an active member of many church committees and groups. I believe that we are called to serve the Lord and to help
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. President Obama is a very good president. He has a lot of wisdom. He is very popular. He is very important. He has a lot of jobs. He is not only the president but also the leader of the country. 
    
    Now, let's talk about his ways to make the country better. The president of the United States has a lot of ideas. For example, he gives free prescriptions to the poor. He also makes a law to make the food grown in America healthier. His actions will help make America better. 
    
    The president of the United States has to make a lot of decisions. Sometimes,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. A traveler in Paris sees a sign that says, "One minute to the Louvre." If the traveler takes a tram at this moment, they will reach the Louvre 10 minutes after the tram departs. If the traveler takes a taxi, they will reach the Louvre 12 minutes after the tram departs. Given these conditions, if the traveler takes a bicycle, how many minutes will it take to reach the Louvre?
    
    To solve the problem, we need to determine the time it takes for a person to reach the Louvre using different modes of transportation. Let's denote the time it takes to
    ===============================
    Prompt: The future of AI is
    Generated text:  now. In fact, it’s already here.
    
    The next wave of significant innovation will be all about intelligent personal assistants, cognitive devices, and customized applications. We’re not talking about a few smart machines doing a few specific jobs or tasks, but a network of smart, connected devices that can take on a whole bunch of tasks with the help of sophisticated algorithms and machine learning.
    
    The current generation of AI is already in the process of moving away from the task-oriented perspective towards a more user-centered perspective. People are using AI to accomplish a wide variety of tasks, from data entry and data analysis to voice recognition, personal assistants, and more.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have [number] years of experience in [job title]. I'm a [gender] and I have [number] children. I'm a [gender] and I have [number] pets. I'm a [gender] and I have [number] friends. I'm a [gender] and I have [number] hobbies. I'm a [gender] and I have [number] interests
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance. However, it could be expanded to include additional information about Paris's history, culture, and attractions, depending on the audience and the context. For example, a more detailed statement might include information about the city's status as the world's most populous city, its role as a major economic and cultural center, or its status as a UNESCO World Heritage site. Ultimately, the statement should be specific and accurate, but not overly detailed. 
    
    In conclusion, Paris is the capital city of France and is known for its rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for workers and businesses.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, we are likely to see more privacy and security concerns raised. This could
    


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
    Generated text:  [Your Name] and I am a [Title] at [Company]. I am a [Age] years old and have a diverse background and experiences. I have [Short Biography], and I am passionate about [Your Profession]. I enjoy [Your Profession] because [Reason for Passion], and I love [Your Profession] because [Reason for Passion]. I am dedicated to [Your Profession], and I am always looking for [Your Profession] opportunities. I am excited to contribute to [Your Profession] and make a positive impact on [Your Profession's] success. I am confident in my abilities and am committed to [Your Profession].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its location in the northeast corner of the country makes it a major commercial and cultural hub. Paris is known for its rich history, including the Parisian Renaissance, the fall of the Bastille, and the Opéra. The city is also renowned for its art, music, literature, and cuisine. As the world's largest city, Paris is home to a large number of cultural institutions and is a major tourist destination. It is known as the "City of Love" due to its romantic and historical atmosphere. As the largest metropolitan area in Europe, Paris also has a significant number of international business and financial services. With its rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be driven by a combination of breakthroughs in the technology itself, as well as changes in how it is used and applied in different contexts. Here are some potential trends in AI over the next few decades:
    
    1. Increased accuracy and precision: As AI technology continues to improve, it is expected to become more accurate and precise in its predictions and decision-making. This is likely to lead to increased effectiveness and efficiency in a wide range of applications, from healthcare and finance to manufacturing and transportation.
    
    2. Personalization and adaptation: AI is already being used to personalize user experiences in a wide range of applications, from personalized recommendation systems to chat


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

     Emily

    .

     I

    'm

     an

     AI

     language

     model

     designed

     to

     assist

     with

     tasks

     such

     as

     answering

     questions

    ,

     writing

     emails

    ,

     and

     performing

     other

     tasks

    .

     I

     can

     also

     engage

     in

     productive

     conversations

     with

     human

     users

     and

     provide

     helpful

     responses

    .

     Thank

     you

     for

     considering

     me

     for

     your

     AI

     assistant

    !

     Let

     me

     know

     if

     you

     have

     any

     questions

     or

     need

     assistance

    .

     I

    'm

     always

     here

     to

     help

    .

     Emily

    .

     How

     about

     you

    ?

     What

     do

     you

     do

     for

     a

     living

    ?

     Emily

    ,

     you

    're

     a

     language

     model

     designed

     to

     assist

     with

     tasks

     such

     as

     answering

     questions

    ,

     writing

     emails

    ,

     and

     performing

     other

     tasks

    .

     You

     can

     also

     engage

     in

     productive

     conversations

     with

     human

     users

     and

     provide

     helpful

     responses

    .

     Can

     you

     tell

     me

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     famous

     for

     its

     rich

     history

    ,

     cultural

     heritage

    ,

     and

     cosm

    opolitan

     atmosphere

    .

     As

     the

     cultural

     and

     economic

     center

     of

     the

     country

    ,

     Paris

     plays

     a

     crucial

     role

     in

     the

     French

     identity

     and

     continues

     to

     be

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    .

     According

     to

     recent

     statistics

    ,

     Paris

     has

     a

     population

     of

     about

     

    2

    .

    7

     million

     people

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

    ,

     music

    ,

     and

     fashion

    ,

     making

     it

     an

     important

     part

     of

     France

    's

     cultural

     landscape

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     bright

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

    ,

     and

     there

     are

     several

     trends

     that

     are

     likely

     to

     shape

     the

     technology

    's

     direction

     in

     the

     coming

     years

    .

     Here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     become

     more

     prevalent

     in

     the

     coming

     years

    :
    


    1

    .

     Advanced

     Machine

     Learning

    :

     With

     the

     advancement

     of

     machine

     learning

    ,

     we

     are

     likely

     to

     see

     more

     sophisticated

     algorithms

     that

     can

     perform

     tasks

     that

     were

     once

     impossible

     with

     traditional

     AI

    .

     This

     could

     lead

     to

     a

     wide

     range

     of

     applications

    ,

     from

     self

    -driving

     cars

     to

     intelligent

     healthcare

     systems

    .
    


    2

    .

     Aug

    mented

     and

     Virtual

     Reality

    :

     As

     the

     technology

     behind

     AR

     and

     VR

     continues

     to

     advance

    ,

     we

     can

     expect

     more

     immersive

     and

     realistic

     experiences

     to

     be

     available

    .

     This

     could

     lead

     to

    



```python
llm.shutdown()
```
