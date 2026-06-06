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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:07,  5.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:07,  5.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:07,  5.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:07,  5.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:07,  5.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:12,  3.64it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:02, 13.44it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 20.62it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 20.62it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 20.62it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 20.62it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 20.62it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 20.62it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 20.62it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:06<00:00, 20.62it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:06<00:00, 20.62it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:06<00:00, 26.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.18it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.18it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.18it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.18it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.18it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.84it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.84it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.84it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.84it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.84it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.84it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=448 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=416 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=256 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.86it/s]Capturing num tokens (num_tokens=208 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.86it/s]Capturing num tokens (num_tokens=192 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.86it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=144 avail_mem=70.90 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.86it/s]

    Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.96it/s] Capturing num tokens (num_tokens=80 avail_mem=70.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=48 avail_mem=70.88 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=48 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=32 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=24 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.39it/s]

    Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=12 avail_mem=70.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.78it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 42.19it/s]


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
    Generated text:  Simon. I have a close friend named Mike. We live together in the same building. Mike has a dog named Spot. I haven't seen Spot since it was a big dog. I have been away from our home. I miss him so much. Can you tell me Spot's age? What do you do for work? What do you like to do in your free time? What do you do when you don't feel well? Can you help me answer these questions? Simon
    A:
    Sure, I'd be happy to help answer your questions! 
    
    Spot's age is not mentioned in the information provided. 
    
    As for work
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. The president of the United States is an elected official. The president of the United States has the power to dissolve Congress. The president of the United States has the power to declare war. The president of the United States has the power to make treaties. The president of the United States is not a political party representative. The president of the United States has the power to make executive order. Which of the following is the most likely statement?
    A: The president of the United States is the owner of the White House.
    B: The president of the United States is the owner of the United States.
    C: The president of the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The most famous landmark in the center of the city is the Eiffel Tower, a wrought iron tower with twelve main floors. The Eiffel Tower is famous for its tall tower and the famous red and white lute-like sculpture, the Cupola. The Eiffel Tower was built in 1889 as a gift to the French people by the French ironworkers who were in Paris during the Second French Revolution. In 1889, 200 workers from the city of Saint-Cloud, France, began construction of the Eiffel Tower. The workers were paid 200
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of individuals and businesses. For AI to be truly useful and beneficial, it needs to be embedded in the core values of businesses, while also influencing individual attitudes and behaviors.
    As the AI technology keeps evolving, the field of AI ethics is also constantly evolving. This means that it is essential to understand the ethical implications of AI and how it should be used.
    In recent years, the world has seen an increase in the use of AI in various industries, including healthcare, finance, and manufacturing. AI has the potential to revolutionize these industries, making them more efficient and effective. However, it is also important to be aware of


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. The city is known for its vibrant nightlife, fashion, and food scene, and is a popular tourist destination. Paris is a city of contrasts, with its historical architecture and modernity blending together in a unique and fascinating way. It is a city that has played a significant role in shaping French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. AI in finance: AI is already being used in finance to improve fraud detection, risk management, and trading algorithms
    


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
    Generated text:  [Name]. I'm a [job title] with over [number] years of experience in the field. I've won numerous awards and accolades for my [job title], and I'm constantly pushing the boundaries of what's possible in my profession. I'm here to share my knowledge, passion, and excitement for what's out there, so you can be part of the adventure. [Name] is always eager to learn, curious about new ideas and fresh perspectives. So if you have any questions, or if you're interested in joining my journey, I'd love to hear from you. 🌟✨ #selfintroduction
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest and most populous city in France, and it serves as the capital of the country, home to the French government, the French Parliament, and the official residence of the President of the Republic. Paris is an important center for culture, arts, and fashion, and it is known as the "city of love" due to its rich historical and romantic history. The city is also known for its numerous museums, cultural events, and festivals throughout the year, such as the Eiffel Tower Festival, the Gare Montmartre Eiffel Tower Festival, and the Le Caire Festival. Paris is also a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and unpredictable. However, some possible trends that are likely to occur in the coming years include:
    
    1. Increased integration with human decision-making: As AI becomes more sophisticated, it is likely to become more integrated with human decision-making, allowing for more nuanced and informed decision-making. This could lead to more ethical and responsible AI systems that take into account the broader context of human values and desires.
    
    2. Improved privacy and security: As AI systems become more advanced, there is a risk of increased privacy and security breaches. This could lead to significant changes in how we collect, store, and use personal data. There may also be emerging


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

     talented

     individual

    .

     I

    've

     always

     been

     fascinated

     by

     the

     world

     around

     me

     and

     always

     sought

     to

     learn

     new

     things

    .

     Whether

     it

    's

     through

     reading

     books

    ,

     attending

     seminars

    ,

     or

     simply

     walking

     around

     my

     city

    ,

     I

    'm

     always

     on

     the

     lookout

     for

     new

     experiences

    .

     I

    'm

     a

     quick

     learner

     and

     always

     try

     to

     incorporate

     new

     information

     into

     my

     daily

     life

    .

     My

     goal

     is

     to

     be

     a

     valuable

     resource

     and

     share

     my

     knowledge

     with

     others

    .

     I

     believe

     that

     with

     dedication

     and

     hard

     work

    ,

     I

     can

     accomplish

     anything

     I

     set

     my

     mind

     to

    .

     Thank

     you

     for

     considering

     me

     as

     a

     potential

     colleague

     or

     friend

    .

     What

     is

     your

     name

     and

     what

     are

     your

     professional

     interests

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    City

     of

     light

    ,

     art

    ,

     and

     culture

    ,

     Paris

     is

     a

     global

     city

     of

     more

     than

     

    1

    0

     million

     people

    ,

     ranking

     as

     one

     of

     the

     world

    's

     most

     important

     cities

     in

     fashion

    ,

     cuisine

    ,

     and

     the

     arts

    .

     
    


    The

     city

     was

     founded

     by

     the

     Romans

     and

     was

     the

     political

     and

     cultural

     capital

     of

     France

     from

     

    1

    2

    0

    0

     to

     

    1

    8

    7

    0

    .

     It

     has

     been

     a

     major

     economic

     center

     since

     the

     

    1

    6

    th

     century

    ,

     with

     the

     financial

     center

     of

     Europe

    .

     The

     city

     is

     home

     to

     the

     Lou

    vre

    ,

     the

     E

    iff

    el

     Tower

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     famous

     landmarks

    .

     Paris

     is

     also

     known

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    .

     Here

     are

     some

     possible

     trends

     that

     AI

     is

     likely

     to

     experience

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

    :

     AI

     will

     become

     more

     integrated

     into

     the

     decision

    -making

     process

    ,

     especially

     in

     areas

     where

     human

     judgement

     is

     not

     sufficient

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     areas

     like

     home

     automation

    :

     AI

     is

     already

     being

     used

     in

     a

     variety

     of

     home

     automation

     applications

    ,

     such

     as

     smart

     home

     devices

    ,

     voice

     assistants

    ,

     and

     automated

     appliances

    .

     We

     can

     expect

     to

     see

     even

     more

     integration

     of

     AI

     into

     everyday

     life

     in

     the

     future

    ,

     such

     as

     in

     building

     automation

    ,

     smart

     cities

    ,

     and

     consumer

     electronics

    



```python
llm.shutdown()
```
