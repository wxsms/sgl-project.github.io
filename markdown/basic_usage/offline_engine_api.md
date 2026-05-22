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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:43,  1.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:43,  1.23it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:43,  1.23it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:43,  1.23it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:43,  1.23it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:43,  1.23it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:43,  1.23it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:43,  1.23it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.69it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 20.40it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:06<00:00, 28.79it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:06<00:00, 28.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.82 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.82 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.82 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.09 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=75.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.45it/s]Capturing num tokens (num_tokens=960 avail_mem=75.06 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.45it/s] Capturing num tokens (num_tokens=896 avail_mem=75.06 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.45it/s]

    Capturing num tokens (num_tokens=832 avail_mem=75.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.45it/s]Capturing num tokens (num_tokens=832 avail_mem=75.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=768 avail_mem=75.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=704 avail_mem=75.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=640 avail_mem=75.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=576 avail_mem=75.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=512 avail_mem=75.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=480 avail_mem=75.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=480 avail_mem=75.04 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=448 avail_mem=75.04 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=416 avail_mem=75.04 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=384 avail_mem=75.04 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=352 avail_mem=75.03 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.38it/s]

    Capturing num tokens (num_tokens=320 avail_mem=75.03 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=320 avail_mem=75.03 GB):  60%|██████    | 35/58 [00:00<00:00, 46.02it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:00<00:00, 46.02it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:00<00:00, 46.02it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:00<00:00, 46.02it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:00<00:00, 46.02it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  60%|██████    | 35/58 [00:00<00:00, 46.02it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  60%|██████    | 35/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=144 avail_mem=74.57 GB):  71%|███████   | 41/58 [00:01<00:00, 47.43it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  71%|███████   | 41/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  71%|███████   | 41/58 [00:01<00:00, 47.43it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.80it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.80it/s] Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.80it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.80it/s]Capturing num tokens (num_tokens=48 avail_mem=74.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.80it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.80it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.84it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.84it/s]Capturing num tokens (num_tokens=24 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.84it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.84it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.84it/s]Capturing num tokens (num_tokens=12 avail_mem=74.53 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.84it/s]Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.84it/s] Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.62it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.62it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 42.32it/s]


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
    Generated text:  Yu Chengyu, and my three friends, Yang, Ling, and Chen, are from the same school. The four of us are going to have a picnic at the park. How should we ask to invite our three friends to the picnic?
    
    A) Can I invite three friends to the picnic?
    
    B) Can I invite some friends to the picnic?
    
    C) Can I invite some friends to the picnic?
    
    D) Can I invite some friends to the picnic?
    
    Answer: $\boxed{\text{C}}$ (fill in the blank)
    
    Let's determine the correct answer to the question "How should we ask to invite our three friends
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to hold another election after the 2020 presidential election. The most recent polls show that 55% of voters support him in a hypothetical vote. If the president is unsure about his campaign strategy and wants to predict the results of the upcoming election with 95% confidence, how many additional polls should he conduct, and what would be the confidence interval for his predictions?
    
    To determine the number of additional polls the president should conduct to achieve a 95% confidence interval for his predictions, we can use the formula for the required sample size \( n \) for a proportion:
    
    \[ n = \
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In 1615, the Treaty of Amiens ended the Hundred Years' War between England and France. To commemorate the treaty, the people of Paris built the Parisian Pantheon, an impressive array of stone columns. The Pantheon was built to honor the peace and friendship between France and England. In the following paragraph, which one includes the concept of 'to honor'? A. The people of Paris built the Parisian Pantheon, an impressive array of stone columns. B. To commemorate the treaty between England and France, the people of Paris built the Parisian Pantheon. C. To honor the peace and
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but the practical implications and benefits will be significant for businesses and individuals alike.
    The future of AI is uncertain, but the practical implications and benefits will be significant for businesses and individuals alike. AI can automate routine tasks, leading to increased productivity and efficiency. It can also help identify potential issues before they become critical. AI can also be used to provide personalized recommendations to customers based on their preferences and behavior patterns. The benefits of AI include increased customer satisfaction, better decision-making, and improved efficiency. However, there are also risks associated with AI, such as job displacement and privacy concerns. As technology continues to advance, it is likely that


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or goal]. I'm a [character trait or quality] who is always [description of something]. I'm [character's age] years old, and I'm currently [occupation] in [industry]. I'm [character's profession] and I'm [character's profession]. I'm [character's profession] and I'm [character's profession]. I'm [character's profession] and I'm [character's profession]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The French capital is a vibrant and dynamic city with a diverse population and a rich history. It is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased automation: As AI becomes more advanced, it is likely to automate many tasks that are currently done by humans, such as data analysis, decision-making, and routine maintenance. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. Enhanced privacy and security: As AI becomes more sophisticated, it is likely to require more data to function effectively. This could lead to increased
    


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
    Generated text:  [insert first name and last name]. I'm a [insert age] year old [insert profession or hobby]. I'm a/an [insert personality trait or hobby] person. I love [insert something you enjoy doing]. I'm [insert the three- or four-letter word that best describes your personality or hobby]. I'm [insert your favorite color, something you have a pet, or something else that makes you unique]. I'm [insert a personal saying or quote]. I'm [insert your favorite quote]. I'm [insert a tagline or catchphrase that represents me]. I'm [insert a question mark or exclamation
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a bustling metropolis with a rich history and culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The city also hosts world-renowned art galleries, museums, and restaurants, making it a popular tourist destination. With its rich cultural heritage and modern amenities, Paris is considered one of the most beautiful and exciting cities in the world. According to the World Heritage site database, Paris is recognized as a UNESCO World Heritage Site. Additionally, it is home to the city-state of Monaco, which is recognized by the United Nations as a non-permanent member of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a number of possible trends, including:
    
    1. Increased availability and affordability: As AI becomes more widely adopted, the costs of developing and deploying AI systems will decrease, making them more accessible to businesses and organizations.
    
    2. Improved models and algorithms: AI models and algorithms will continue to improve, as researchers and developers work to develop new techniques and techniques to improve model accuracy, efficiency, and interpretability.
    
    3. Increased ethical and privacy concerns: As AI systems become more involved in decision-making processes, there will likely be increased ethical and privacy concerns. Governments and organizations will need to work to develop ethical guidelines and regulations to


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

    ],

     and

     I

    'm

     a

     [

    general

     description

     of

     your

     profession

    ,

     skills

    ,

     or

     attributes

    ].

     I

     thrive

     on

     [

    reason

     for

     this

     profession

     or

     attribute

    ],

     and

     I

     enjoy

     [

    why

     you

    're

     passionate

     about

     this

     topic

    ].

     If

     you

     have

     any

     questions

     or

     interests

     that

     I

     can

     answer

    ,

     please

     let

     me

     know

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     the

     information

     you

    're

     looking

     for

    .

     [

    Your

     name

    ,

     if

     applicable

    ].


    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    general

     description

     of

     your

     profession

    ,

     skills

    ,

     or

     attributes

    ].

     I

     thrive

     on

     [

    reason

     for

     this

     profession

     or

     attribute

    ],

     and

     I

     enjoy

     [

    why

     you

    're

     passionate

     about

     this

     topic

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     the

     

    6

    th

    -largest

     country

     in

     the

     world

     by

     area

     and

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     and

     cultural

     attractions

    .

     The

     city

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     and

     many

     other

     famous

     landmarks

    .

     It

    's

     a

     bustling

     hub

     of

     activity

     with

     a

     rich

     history

    ,

     culture

    ,

     and

     cuisine

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    Paris

     experience

    "

     due

     to

     its

     diverse

     and

     extensive

     offerings

    .

     It

    's

     a

     must

    -

    visit

     destination

     for

     anyone

     looking

     for

     an

     unforgettable

     French

     experience

    .

     Paris

     is

     a

     vibrant

    ,

     cosm

    opolitan

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

    ,

     with

     many

     possibilities

     and

     possibilities

     for

     the

     future

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     AI

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    .

     This

     includes

     the

     integration

     of

     AI

     into

     our

     homes

    ,

     work

    ,

     and

     entertainment

    .

     As

     AI

     improves

     in

     accuracy

     and

     efficiency

    ,

     we

     will

     see

     more

     seamless

     integration

     of

     these

     technologies

    .
    


    2

    .

     AI

     will

     become

     more

     general

     and

     less

     specialized

    .

     With

     the

     increasing

     amount

     of

     data

     and

     knowledge

     available

    ,

     AI

     will

     become

     more

     general

    ,

     meaning

     it

     will

     be

     able

     to

     understand

     and

     make

     decisions

     on

     a

     wide

     range

     of

     topics

    .

     As

     a

     result

    ,

     we

     will

     see

     more

     complex

     AI

     systems

     that

     are

     capable

     of

     understanding

     and

     making

     decisions

     on

    



```python
llm.shutdown()
```
