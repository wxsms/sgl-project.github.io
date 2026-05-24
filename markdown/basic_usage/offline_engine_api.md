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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.14it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=1024):  19%|█▉        | 11/58 [00:04<00:11,  4.03it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.66it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.90it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.89it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.89it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.89it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.89it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.89it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.76it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.76it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.76it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.76it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 44.00it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 44.00it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 44.00it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 44.00it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 44.00it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 44.00it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 44.00it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:00<00:00, 46.22it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:00<00:00, 46.22it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:00<00:00, 46.22it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:00<00:00, 46.22it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:00<00:00, 46.22it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:00<00:00, 46.22it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 46.22it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  71%|███████   | 41/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 47.70it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  71%|███████   | 41/58 [00:01<00:00, 47.70it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.85it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.85it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.85it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.85it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.85it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.85it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.89it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.89it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.89it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.89it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.89it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.89it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.32it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.32it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.32it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 41.24it/s]


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
    Generated text:  Camelia and I'm an aspiring writer. I currently hold a Master's degree in Journalism from Rutgers University, New Jersey. I specialize in writing about women's issues. I have a passion for uncovering the dark side of the human condition and how to live a healthier and happier life. As an aspiring writer, what are some specific ways I can improve my writing skills and learn more about the complexities of the human psyche? Also, I am interested in learning more about how to write about women's issues in a way that is respectful and inclusive. Can you provide some guidance on how to do so while also ensuring that the writing is accurate
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide between two candidates for his running mate. He has 1000 votes to his name and the two candidates have the following combined votes:
    
    Candidate A has 600 votes and Candidate B has 400 votes.
    
    What is the total votes that Candidate A needs in order to win the election?
    To determine how many more votes Candidate A needs to win the election, we need to compare the number of votes Candidate A has with the number of votes Candidate B has. The total number of votes for the election is the sum of the votes of both candidates.
    
    First, we calculate the total number of votes:
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is known for its historical landmarks such as the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Arc de Triomphe.
    
    How many stars would you expect the population of Paris to increase to within 5 years, assuming that the population grows at a constant rate of 1.3% per year? (Round your answer to two decimal places.)
    
    To estimate the population of Paris in 2036, we can use the exponential growth formula:
    
    P = P0 * (1 + r)^t
    
    Where:
    P = projected population (in thousands)
    P0 = initial
    ===============================
    Prompt: The future of AI is
    Generated text:  about to change – what to expect? After a long time of uncertainty and debate, the European Commission announced its new AI Roadmap. The plan addresses the rapid and global scale of AI use in different sectors. It aims to ensure that AI is used responsibly and for the benefit of all, regardless of the industry sector or the user.
    The roadmap, which starts in 2024 and lasts three years, aims to produce and coordinate international standards and regulatory frameworks, such as the General Data Protection Regulation (GDPR), to safeguard the right of individuals to control their personal data and protect sensitive data.
    The roadmap identifies the main challenges and


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. Paris is also famous for its fashion industry, art scene, and its role as a cultural hub for Europe. The city is home to many world-renowned museums, including the Louvre, the Musée d'Orsay, and the Musée d'Art Moderne. Paris is a popular tourist destination and is known for its cuisine, including French cuisine, and its nightlife. The city is also home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to play an increasingly important role in areas such as healthcare, finance, and energy, as these industries increasingly rely on data and algorithms to make decisions and optimize operations. Finally, AI is likely to continue to be a key driver of innovation and economic growth, as it enables new forms of technology and business models that were previously unimaginable. Overall, the future of AI is likely to
    


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
    Generated text:  [Name], and I'm a [character type]. I'm here to share my personal experiences, skills, and passions with everyone who asks. I'm passionate about [reason for interest] and [reason for interest in other areas]. I enjoy [activities I enjoy doing]. I also love [reason for being loved/liked by others]. What do you think makes you successful in your field? What challenges have you overcome? What advice would you give to someone aspiring to be [or a similar role]? What challenges would you face in a [other role] and how would you overcome them?
    [Name] is a [character type]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital city of France and one of the world's most famous and historically significant cities. The city is home to the country's government, cultural institutions, and many of its iconic landmarks, including the Eiffel Tower and the Louvre Museum. Paris is also known for its vibrant culture and cuisine, as well as its love of wine and fashion. Paris is a major cultural and economic hub, and the city has been an important center of politics and diplomacy for centuries. As the seat of the French government, it continues to play a significant role in French society and politics. The city has also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of significant trends, including:
    
    1. Increased accuracy and efficiency: As AI continues to improve, it is likely to become more accurate in its predictions and recommendations. This will lead to more efficient and streamlined processes, freeing up resources and allowing for greater productivity.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning and natural language processing. This will allow for more sophisticated and personalized experiences, as well as the development of new applications and services.
    
    3. Personalization and context-awareness: AI is likely to become more personal, with machines being able to understand and


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

    Type

     of

     Person

    ],

     [

    Job

     Title

    ],

     [

    Achie

    vements

    ].

     I

    'm

     passionate

     about

     [

    Describe

     Something

     Interesting

     About

     Yourself

    ].

     I

     love

     [

    What

     I

     Do

     Profession

    ally

    ],

     and

     I

    'm

     [

    How

     Much

     Time

     Do

     You

     Spend

     on

     It

    ?

    ].

     I

    'm

     [

    Your

     Age

    ],

     [

    Your

     Gender

    ],

     [

    Your

     National

    ity

    ],

     and

     I

    'm

     [

    Your

     Country

    ].

     I

    'm

     known

     for

     my

     [

    Your

     Best

     Quality

     or

     Quality

     of

     Work

    ].

     I

    'm

     [

    Your

     Education

     Level

    ]

     and

     I

    'm

     [

    Your

     Relationship

     Status

    ].

     I

    'm

     [

    Your

     Favorite

     Food

    ],

     [

    Your

     Favorite

     Drink

    ],

     and

     [

    Your

     Pets

    '.

    ].

     I

    'm

     [

    Your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     is

     home

     to

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

     and

     many

     other

     landmarks

    .

     Its

     history

     dates

     back

     to

     the

     ancient

     city

     of

     Paris

    ,

     founded

     in

     the

     

    6

    th

     century

     BCE

    .

     The

     city

     was

     conquered

     by

     the

     Romans

     in

     the

     

    1

    st

     century

     CE

    ,

     and

     was

     later

     ruled

     by

     the

     French

     kings

     and

     queens

    .

     Paris

     is

     known

     for

     its

     fashion

    ,

     gastr

    onomy

    ,

     and

     art

    ,

     and

     continues

     to

     be

     a

     popular

     tourist

     destination

     in

     the

     world

    .

     The

     city

     is

     also

     home

     to

     many

     cultural

     institutions

    ,

     including

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

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Overall

    ,

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    :
    


    1

    .

     Increased

     automation

     and

     precision

    :

     AI

     is

     expected

     to

     become

     more

     sophisticated

     and

     capable

     of

     performing

     a

     wider

     range

     of

     tasks

     with

     greater

     accuracy

     and

     efficiency

    .

     This

     will

     lead

     to

     more

     precise

     and

     efficient

     applications

     of

     AI

     in

     various

     industries

    .
    


    2

    .

     Development

     of

     autonomous

     systems

    :

     AI

     will

     continue

     to

     develop

     autonomous

     systems

     that

     can

     operate

     in

     a

     wide

     range

     of

     environments

    ,

     including

     physical

    ,

     chemical

    ,

     and

     biological

     environments

    .

     These

     systems

     will

     be

     able

     to

     perform

     tasks

     with

     minimal

     human

     oversight

    ,

     leading

     to

     increased

     safety

     and

     reliability

    .
    


    3

    .

     Enhanced

     cognitive

     capabilities

    :

     AI

     will

     continue

     to

     develop

     and

     expand

     its

     capabilities

    ,

     including

     more

     advanced

     algorithms

    ,

     natural

    



```python
llm.shutdown()
```
