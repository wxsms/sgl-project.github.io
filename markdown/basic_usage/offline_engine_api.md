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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 17.53it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 33.46it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.59 GB):   3%|▎         | 2/58 [00:00<00:03, 15.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.59 GB):   3%|▎         | 2/58 [00:00<00:03, 15.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.58 GB):   3%|▎         | 2/58 [00:00<00:03, 15.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=70.58 GB):   3%|▎         | 2/58 [00:00<00:03, 15.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.58 GB):   9%|▊         | 5/58 [00:00<00:02, 20.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.58 GB):   9%|▊         | 5/58 [00:00<00:02, 20.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.57 GB):   9%|▊         | 5/58 [00:00<00:02, 20.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.57 GB):   9%|▊         | 5/58 [00:00<00:02, 20.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.57 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.57 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.56 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.56 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=70.55 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.55 GB):  21%|██        | 12/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.55 GB):  21%|██        | 12/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.55 GB):  21%|██        | 12/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.54 GB):  21%|██        | 12/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.54 GB):  21%|██        | 12/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.54 GB):  21%|██        | 12/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.46it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.46it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=70.51 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.51 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=960 avail_mem=70.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.13it/s] Capturing num tokens (num_tokens=896 avail_mem=70.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=832 avail_mem=70.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=768 avail_mem=70.51 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=704 avail_mem=70.51 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=704 avail_mem=70.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.19it/s]Capturing num tokens (num_tokens=640 avail_mem=70.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.19it/s]Capturing num tokens (num_tokens=576 avail_mem=70.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.19it/s]Capturing num tokens (num_tokens=512 avail_mem=70.49 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.19it/s]

    Capturing num tokens (num_tokens=480 avail_mem=70.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.19it/s]Capturing num tokens (num_tokens=448 avail_mem=70.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.19it/s]Capturing num tokens (num_tokens=448 avail_mem=70.51 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=416 avail_mem=70.50 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=384 avail_mem=70.50 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=352 avail_mem=70.50 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.53it/s]Capturing num tokens (num_tokens=320 avail_mem=70.49 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.53it/s]Capturing num tokens (num_tokens=288 avail_mem=70.49 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.53it/s]Capturing num tokens (num_tokens=288 avail_mem=70.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=256 avail_mem=70.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=240 avail_mem=70.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=224 avail_mem=70.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.56it/s]

    Capturing num tokens (num_tokens=208 avail_mem=70.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=192 avail_mem=70.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=192 avail_mem=70.48 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=176 avail_mem=70.47 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=160 avail_mem=70.47 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=144 avail_mem=70.47 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=128 avail_mem=70.46 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=112 avail_mem=70.46 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=112 avail_mem=70.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=96 avail_mem=70.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.12it/s] Capturing num tokens (num_tokens=80 avail_mem=70.45 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.12it/s]

    Capturing num tokens (num_tokens=64 avail_mem=70.45 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=48 avail_mem=70.45 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=32 avail_mem=70.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=32 avail_mem=70.44 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=28 avail_mem=70.44 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=24 avail_mem=70.44 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=20 avail_mem=70.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=16 avail_mem=70.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=12 avail_mem=70.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=12 avail_mem=70.43 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=8 avail_mem=70.42 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.28it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=70.42 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=4 avail_mem=70.42 GB): 100%|██████████| 58/58 [00:01<00:00, 34.76it/s]


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
    Generated text:  Katherine and I am a professional content writer for the Social Media Examiner. I help people find the best ways to utilize social media to reach their goals, and to engage with the right audience at the right time. As a content writer, my expertise includes an extensive knowledge of SEO, social media marketing, and content creation.
    I am passionate about helping businesses grow and reaching their goals through social media. I am also an avid fan of podcasts and have over 150k listening hours to my name. If you have any questions or need assistance with content creation, please don't hesitate to reach out. I look forward to helping you achieve
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office, holding the office for a term of ( ) years.
    A. 3
    B. 5
    C. 7
    D. 9
    Answer:
    
    B
    
    Which of the following is NOT a basic requirement for the term of office of the President of the People's Republic of China?
    A. The term of office is 5 years
    B. The term of office is 3 years
    C. Term extension is allowed
    D. Term extension is not allowed
    Answer:
    
    B
    
    The department responsible for safety production supervision and management is the ____.
    A. State Administration of Work Safety
    B.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is 2.24 million. If the population grows by 20% in the first year, by what percent does the population grow each year if the growth rate increases to 21% over the next two years? Express the percentage growth rate for the second year as a decimal rounded to three decimal places. The initial population of Paris is 2.24 million. If the population grows by 20% in the first year, the population after the first year can be calculated as follows:
    
    \[
    \text{Population after the first year} = 2.24 \
    ===============================
    Prompt: The future of AI is
    Generated text:  dark. AI systems are not only translating many different languages, but they are also inherently biased. On the basis of these facts, a new study from the University of Sussex has come up with a solution for this problem. The study has been published in the journal Artificial Intelligence. The researchers have come up with a new algorithm that can help engineers to ensure that AI systems are not biased.
    
    The team from the University of Sussex has developed a new algorithm called AI Bias Classifier. The AI Bias Classifier analyzes a system's behavior to find any biases that can be detected. The system itself is not biased; the AI system is just translating and processing the


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I have always been passionate about [Your Passion], and I am always looking for ways to [Your Goal]. I am always eager to learn and grow, and I am always willing to take on new challenges. I am a [Your Character Trait] and I am always ready to help others. I am a [Your Personality] and I am always ready to make a difference in the world. I am a [Your Goal] and I am always looking for ways to [Your Goal]. I am a [Your Character Trait] and I am always ready to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its rich history and culture. The city is also home to many famous museums, including the Musée d'Orsay, the Musée Rodin, and the Musée d'Orsay. Paris is a vibrant and cosmopolitan city with a diverse population and a rich cultural heritage. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of other technologies, such as smart homes, self-driving cars, and virtual assistants. As more and more technologies become integrated with AI, we can expect to see even more integration in the future.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our lives, there will be a greater emphasis on ethical considerations. This will include issues
    


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
    Generated text:  Sarah, and I'm a dedicated book lover. I love reading and exploring new genres, but my true passion is for writing. I spend my days typing away, pouring over countless pages of stories, and dreaming up new worlds for my characters. I'm always on the lookout for new ideas and techniques for writing, and I'm always looking for opportunities to share my work with others. I'm excited to be a part of a team and work on projects with you! How about you? What's your genre and what kind of work are you currently working on? [Your Name] I'm a [genre] writer with some upcoming projects
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and there are many potential trends we could see in the coming years. Here are a few:
    
    1. Advancements in machine learning: Machine learning is one of the fastest growing areas of AI. We expect to see more sophisticated models of decision-making, self-driving cars, and more. We also expect to see more applications of AI in industries like healthcare and finance.
    
    2. AI in medicine: AI is already being used to analyze medical images, predict disease, and assist with treatment decisions. We expect to see even more advanced applications, including virtual assistants for medical patients, personalized treatment plans, and better diagnostics.
    
    3. AI in finance


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

    ]

     and

     I

    'm

     a

     [

    job

     title

    ],

     [

    job

     title

    ]

     alumni

    .

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     about

     your

     background

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     I

    'm

     currently

     working

     on

     a

     [

    project

    ]

     that

     requires

     a

     lot

     of

     research

     and

     analysis

    ,

     and

     I

    'm

     looking

     to

     network

     with

     like

    -minded

     individuals

    .

     What

     exc

    ites

     you

     the

     most

     about

     the

     project

    ?

     And

     what

     do

     you

     like

     to

     do

     for

     fun

    ?

     I

     enjoy

     [

    activity

     or

     hobby

    ]

     and

     I

     have

     a

     bit

     of

     a

     passion

     for

     [

    specific

     interest

    ].

     What

    ’s

     your

     favorite

     color

     and

     why

    ?

     What

    ’s

     your

     favorite

     book

    ?

     I

     love

     [

    book

     title

    ]

     by

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

    ,

     vibrant

     culture

    ,

     and

     beautiful

     architecture

    .

     Paris

     is

     located

     in

     the

     eastern

     part

     of

     the

     country

    ,

     surrounded

     by

     the

     Se

    ine

     River

     and

     the

     surrounding

     countryside

    .

     The

     city

     is

     home

     to

     some

     of

     Europe

    's

     most

     famous

     landmarks

    ,

     including

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

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     a

     bustling

     met

    ropolis

     with

     a

     diverse

     population

    ,

     and

     it

     has

     been

     a

     major

     center

     of

     culture

    ,

     politics

    ,

     and

     trade

     for

     over

     a

     thousand

     years

    .

     Its

     cultural

     and

     historical

     legacy

     continues

     to

     be

     a

     major

     draw

     for

     visitors

     to

     the

     city

    ,

     and

     it

     remains

     one

     of

     the

     most

     important

     cities

     in

     Europe

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     rapidly

    ,

     with

     a

     wide

     range

     of

     potential

     trends

     that

     could

     shape

     the

     direction

     of

     the

     technology

    .

     Here

     are

     some

     potential

     trends

     that

     are

     currently

     being

     explored

     and

     that

     could

     potentially

     influence

     AI

     in

     the

     future

    :
    


    1

    .

     Increased

     Integration

     with

     Other

     Technologies

    :

     One

     of

     the

     most

     common

     trends

     that

     is

     expected

     to

     impact

     AI

     in

     the

     future

     is

     the

     integration

     of

     AI

     with

     other

     technologies

    .

     This

     could

     include

     the

     integration

     of

     AI

     with

     other

     technologies

     like

     blockchain

    ,

     quantum

     computing

    ,

     and

     neu

    rom

    orphic

     computing

    ,

     which

     could

     further

     increase

     the

     speed

    ,

     accuracy

    ,

     and

     scalability

     of

     AI

     models

    .
    


    2

    .

     Enhanced

     Privacy

     and

     Security

    :

     With

     the

     increased

     use

     of

     AI

    ,

     it

     is

    



```python
llm.shutdown()
```
