# Reasoning Parser

SGLang supports parsing reasoning content out from "normal" content for reasoning models such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).

## Supported Models & Parsers

| Model  |  Reasoning tags      | Parser | Notes |
|---------|-----------------------------|------------------|-------|
| [DeepSeek‑R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `<think>` … `</think>` | `deepseek-r1` | Supports all variants (R1, R1-0528, R1-Distill) |
| [DeepSeek‑V3 series](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | `<think>` … `</think>` | `deepseek-v3` | Including [DeepSeek‑V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp). Supports `thinking` parameter |
| [Standard Qwen3 models](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `<think>` … `</think>` | `qwen3` | Supports `enable_thinking` parameter |
| [Qwen3-Thinking models](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) | `<think>` … `</think>` | `qwen3` or `qwen3-thinking` | Always generates thinking content |
| [Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | `◁think▷` … `◁/think▷` | `kimi_k2` | Uses special thinking delimiters. Also requires `--tool-call-parser kimi_k2` for tool use. |
| [GPT OSS](https://huggingface.co/openai/gpt-oss-120b) | `<\|channel\|>analysis<\|message\|>` … `<\|end\|>` | `gpt-oss` | N/A |
### Model-Specific Behaviors

**DeepSeek-R1 Family:**
- DeepSeek-R1: No `<think>` start tag, jumps directly to thinking content
- DeepSeek-R1-0528: Generates both `<think>` start and `</think>` end tags
- Both are handled by the same `deepseek-r1` parser

**DeepSeek-V3 Family:**
- DeepSeek-V3.1/V3.2: Hybrid model supporting both thinking and non-thinking modes, use the `deepseek-v3` parser and `thinking` parameter (NOTE: not `enable_thinking`)

**Qwen3 Family:**
- Standard Qwen3 (e.g., Qwen3-2507): Use `qwen3` parser, supports `enable_thinking` in chat templates
- Qwen3-Thinking (e.g., Qwen3-235B-A22B-Thinking-2507): Use `qwen3` or `qwen3-thinking` parser, always thinks

**Kimi K2:**
- Kimi K2 Thinking: Uses special `◁think▷` and `◁/think▷` tags. For agentic tool use, also specify `--tool-call-parser kimi_k2`.

**GPT OSS:**
- GPT OSS: Uses special `<|channel|>analysis<|message|>` and `<|end|>` tags

## Usage

### Launching the Server

Specify the `--reasoning-parser` option.


```python
import requests
from openai import OpenAI
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --reasoning-parser deepseek-r1 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.18s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.11s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.13s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:37,  5.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:37,  5.92s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:39,  2.85s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:39,  2.85s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:41,  1.85s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:41,  1.85s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:13,  1.35s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:13,  1.35s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:08<00:55,  1.05s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:08<00:55,  1.05s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:44,  1.16it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:44,  1.16it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:09<00:36,  1.39it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:09<00:36,  1.39it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:09<00:30,  1.61it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:09<00:30,  1.61it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:10<00:26,  1.87it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:10<00:26,  1.87it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:10<00:22,  2.14it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:10<00:22,  2.14it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:10<00:19,  2.39it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:10<00:19,  2.39it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:10<00:17,  2.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:10<00:17,  2.67it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:11<00:14,  3.02it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:11<00:14,  3.02it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:11<00:13,  3.33it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:11<00:13,  3.33it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:11<00:11,  3.70it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:11<00:11,  3.70it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:11<00:10,  4.12it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:11<00:10,  4.12it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:11<00:08,  4.59it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:11<00:08,  4.59it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:12<00:07,  5.19it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:12<00:07,  5.19it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:12<00:06,  5.80it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:12<00:06,  5.80it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:12<00:05,  6.57it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:12<00:05,  6.57it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:12<00:05,  6.57it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:12<00:04,  8.28it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:12<00:04,  8.28it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:12<00:04,  8.28it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:12<00:03,  9.92it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:12<00:03,  9.92it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:12<00:03,  9.92it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:12<00:02, 11.71it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:12<00:02, 11.71it/s]

    Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:12<00:02, 11.71it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:12<00:02, 13.07it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:12<00:02, 13.07it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:12<00:02, 13.07it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:12<00:02, 13.07it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:12<00:01, 16.15it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:12<00:01, 16.15it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:13<00:01, 16.15it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:13<00:01, 16.94it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:13<00:01, 16.94it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:13<00:01, 16.94it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:13<00:01, 16.94it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:13<00:01, 19.47it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:13<00:01, 19.47it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:13<00:01, 19.47it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:13<00:01, 19.47it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:13<00:01, 19.47it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:13<00:00, 23.33it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:13<00:00, 23.33it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:13<00:00, 23.33it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:13<00:00, 23.33it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:13<00:00, 24.54it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:13<00:00, 24.54it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:13<00:00, 24.54it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:13<00:00, 24.54it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:13<00:00, 24.54it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:13<00:00, 26.71it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:13<00:00, 26.71it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:13<00:00, 26.71it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:13<00:00, 26.71it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:13<00:00, 26.71it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:13<00:00, 28.45it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:13<00:00, 28.45it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:13<00:00, 28.45it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:13<00:00, 28.45it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:13<00:00, 28.45it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:13<00:00, 28.45it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:13<00:00, 33.88it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:13<00:00, 33.88it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:13<00:00, 33.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:13<00:00,  4.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.52 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.71 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.71 GB):   3%|▎         | 2/58 [00:01<00:50,  1.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.84 GB):   3%|▎         | 2/58 [00:01<00:50,  1.12it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.84 GB):   5%|▌         | 3/58 [00:02<00:46,  1.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.97 GB):   5%|▌         | 3/58 [00:02<00:46,  1.19it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.97 GB):   7%|▋         | 4/58 [00:03<00:42,  1.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.11 GB):   7%|▋         | 4/58 [00:03<00:42,  1.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.11 GB):   9%|▊         | 5/58 [00:03<00:38,  1.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.23 GB):   9%|▊         | 5/58 [00:03<00:38,  1.37it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.23 GB):  10%|█         | 6/58 [00:04<00:34,  1.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.36 GB):  10%|█         | 6/58 [00:04<00:34,  1.51it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.36 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.09 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.63it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.09 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.16 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.81it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.16 GB):  16%|█▌        | 9/58 [00:05<00:24,  1.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.18 GB):  16%|█▌        | 9/58 [00:05<00:24,  1.99it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=26.18 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.21 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.18it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.21 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.67 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.39it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.67 GB):  21%|██        | 12/58 [00:06<00:17,  2.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.66 GB):  21%|██        | 12/58 [00:06<00:17,  2.62it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.66 GB):  22%|██▏       | 13/58 [00:07<00:15,  2.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.65 GB):  22%|██▏       | 13/58 [00:07<00:15,  2.87it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.65 GB):  24%|██▍       | 14/58 [00:07<00:13,  3.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.65 GB):  24%|██▍       | 14/58 [00:07<00:13,  3.18it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.65 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.64 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.64 GB):  28%|██▊       | 16/58 [00:07<00:10,  3.89it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.64 GB):  28%|██▊       | 16/58 [00:07<00:10,  3.89it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=26.64 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.63 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.63 GB):  31%|███       | 18/58 [00:07<00:08,  4.86it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.44 GB):  31%|███       | 18/58 [00:07<00:08,  4.86it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=26.44 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.47 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.47 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.57 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.00it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=26.57 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.80it/s]Capturing num tokens (num_tokens=960 avail_mem=26.56 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.80it/s] Capturing num tokens (num_tokens=896 avail_mem=26.53 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.80it/s]Capturing num tokens (num_tokens=896 avail_mem=26.53 GB):  40%|███▉      | 23/58 [00:08<00:04,  8.20it/s]Capturing num tokens (num_tokens=832 avail_mem=26.44 GB):  40%|███▉      | 23/58 [00:08<00:04,  8.20it/s]

    Capturing num tokens (num_tokens=768 avail_mem=26.43 GB):  40%|███▉      | 23/58 [00:08<00:04,  8.20it/s]Capturing num tokens (num_tokens=768 avail_mem=26.43 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.34it/s]Capturing num tokens (num_tokens=704 avail_mem=26.45 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.34it/s]Capturing num tokens (num_tokens=640 avail_mem=26.44 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.34it/s]

    Capturing num tokens (num_tokens=640 avail_mem=26.44 GB):  47%|████▋     | 27/58 [00:08<00:02, 10.44it/s]Capturing num tokens (num_tokens=576 avail_mem=26.44 GB):  47%|████▋     | 27/58 [00:08<00:02, 10.44it/s]Capturing num tokens (num_tokens=512 avail_mem=26.43 GB):  47%|████▋     | 27/58 [00:08<00:02, 10.44it/s]Capturing num tokens (num_tokens=512 avail_mem=26.43 GB):  50%|█████     | 29/58 [00:08<00:02, 11.64it/s]Capturing num tokens (num_tokens=480 avail_mem=26.42 GB):  50%|█████     | 29/58 [00:08<00:02, 11.64it/s]

    Capturing num tokens (num_tokens=448 avail_mem=26.42 GB):  50%|█████     | 29/58 [00:09<00:02, 11.64it/s]Capturing num tokens (num_tokens=448 avail_mem=26.42 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.40it/s]Capturing num tokens (num_tokens=416 avail_mem=26.46 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.40it/s]Capturing num tokens (num_tokens=384 avail_mem=26.44 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.40it/s]

    Capturing num tokens (num_tokens=384 avail_mem=26.44 GB):  57%|█████▋    | 33/58 [00:09<00:02, 12.04it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  57%|█████▋    | 33/58 [00:09<00:02, 12.04it/s]Capturing num tokens (num_tokens=320 avail_mem=26.40 GB):  57%|█████▋    | 33/58 [00:09<00:02, 12.04it/s]Capturing num tokens (num_tokens=320 avail_mem=26.40 GB):  60%|██████    | 35/58 [00:09<00:01, 12.35it/s]Capturing num tokens (num_tokens=288 avail_mem=26.43 GB):  60%|██████    | 35/58 [00:09<00:01, 12.35it/s]

    Capturing num tokens (num_tokens=256 avail_mem=26.38 GB):  60%|██████    | 35/58 [00:09<00:01, 12.35it/s]Capturing num tokens (num_tokens=256 avail_mem=26.38 GB):  64%|██████▍   | 37/58 [00:09<00:01, 12.74it/s]Capturing num tokens (num_tokens=240 avail_mem=26.37 GB):  64%|██████▍   | 37/58 [00:09<00:01, 12.74it/s]Capturing num tokens (num_tokens=224 avail_mem=26.40 GB):  64%|██████▍   | 37/58 [00:09<00:01, 12.74it/s]Capturing num tokens (num_tokens=224 avail_mem=26.40 GB):  67%|██████▋   | 39/58 [00:09<00:01, 13.93it/s]Capturing num tokens (num_tokens=208 avail_mem=26.35 GB):  67%|██████▋   | 39/58 [00:09<00:01, 13.93it/s]

    Capturing num tokens (num_tokens=192 avail_mem=26.37 GB):  67%|██████▋   | 39/58 [00:09<00:01, 13.93it/s]Capturing num tokens (num_tokens=192 avail_mem=26.37 GB):  71%|███████   | 41/58 [00:09<00:01, 13.71it/s]Capturing num tokens (num_tokens=176 avail_mem=26.36 GB):  71%|███████   | 41/58 [00:09<00:01, 13.71it/s]Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  71%|███████   | 41/58 [00:09<00:01, 13.71it/s]Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:09<00:01, 14.82it/s]Capturing num tokens (num_tokens=144 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:01, 14.82it/s]

    Capturing num tokens (num_tokens=128 avail_mem=26.34 GB):  74%|███████▍  | 43/58 [00:10<00:01, 14.82it/s]Capturing num tokens (num_tokens=128 avail_mem=26.34 GB):  78%|███████▊  | 45/58 [00:10<00:01, 11.33it/s]Capturing num tokens (num_tokens=112 avail_mem=26.33 GB):  78%|███████▊  | 45/58 [00:10<00:01, 11.33it/s]Capturing num tokens (num_tokens=96 avail_mem=26.32 GB):  78%|███████▊  | 45/58 [00:10<00:01, 11.33it/s] Capturing num tokens (num_tokens=96 avail_mem=26.32 GB):  81%|████████  | 47/58 [00:10<00:00, 12.91it/s]Capturing num tokens (num_tokens=80 avail_mem=26.29 GB):  81%|████████  | 47/58 [00:10<00:00, 12.91it/s]

    Capturing num tokens (num_tokens=64 avail_mem=26.30 GB):  81%|████████  | 47/58 [00:10<00:00, 12.91it/s]Capturing num tokens (num_tokens=64 avail_mem=26.30 GB):  84%|████████▍ | 49/58 [00:10<00:00, 13.91it/s]Capturing num tokens (num_tokens=48 avail_mem=26.29 GB):  84%|████████▍ | 49/58 [00:10<00:00, 13.91it/s]Capturing num tokens (num_tokens=32 avail_mem=26.26 GB):  84%|████████▍ | 49/58 [00:10<00:00, 13.91it/s]Capturing num tokens (num_tokens=32 avail_mem=26.26 GB):  88%|████████▊ | 51/58 [00:10<00:00, 15.15it/s]Capturing num tokens (num_tokens=28 avail_mem=26.25 GB):  88%|████████▊ | 51/58 [00:10<00:00, 15.15it/s]

    Capturing num tokens (num_tokens=24 avail_mem=26.24 GB):  88%|████████▊ | 51/58 [00:10<00:00, 15.15it/s]Capturing num tokens (num_tokens=24 avail_mem=26.24 GB):  91%|█████████▏| 53/58 [00:10<00:00, 16.07it/s]Capturing num tokens (num_tokens=20 avail_mem=26.25 GB):  91%|█████████▏| 53/58 [00:10<00:00, 16.07it/s]Capturing num tokens (num_tokens=16 avail_mem=26.23 GB):  91%|█████████▏| 53/58 [00:10<00:00, 16.07it/s]Capturing num tokens (num_tokens=16 avail_mem=26.23 GB):  95%|█████████▍| 55/58 [00:10<00:00, 16.58it/s]Capturing num tokens (num_tokens=12 avail_mem=26.23 GB):  95%|█████████▍| 55/58 [00:10<00:00, 16.58it/s]

    Capturing num tokens (num_tokens=8 avail_mem=26.21 GB):  95%|█████████▍| 55/58 [00:10<00:00, 16.58it/s] Capturing num tokens (num_tokens=4 avail_mem=26.20 GB):  95%|█████████▍| 55/58 [00:10<00:00, 16.58it/s]Capturing num tokens (num_tokens=4 avail_mem=26.20 GB): 100%|██████████| 58/58 [00:10<00:00, 18.52it/s]Capturing num tokens (num_tokens=4 avail_mem=26.20 GB): 100%|██████████| 58/58 [00:10<00:00,  5.31it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


Note that `--reasoning-parser` defines the parser used to interpret responses.

### OpenAI Compatible API

Using the OpenAI compatible API, the contract follows the [DeepSeek API design](https://api-docs.deepseek.com/guides/reasoning_model) established with the release of DeepSeek-R1:

- `reasoning_content`: The content of the CoT.
- `content`: The content of the final answer.


```python
# Initialize OpenAI-like client
client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
model_name = client.models.list().data[0].id

messages = [
    {
        "role": "user",
        "content": "What is 1+3?",
    }
]
```

#### Non-Streaming Request


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": True},
)
print_highlight("==== Reasoning ====")
print_highlight(response_non_stream.choices[0].message.reasoning_content)

print_highlight("==== Text ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I will add the two numbers together to find the total.<br><br>After performing the addition, I will present the final answer clearly.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


#### Streaming Request


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>


Optionally, you can buffer the reasoning content to the last reasoning chunk (or the first chunk after the reasoning content).


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True, "stream_reasoning": False},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>I see the problem is to calculate 1 plus 3.<br><br>I'll start by identifying the two numbers involved: 1 and 3.<br><br>Next, I'll add these two numbers together to find the sum.<br><br>Finally, the result of adding 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = \boxed{4}<br>\]</strong>


The reasoning separation is enable by default when specify . 
**To disable it, set the `separate_reasoning` option to `False` in request.**


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": False},
)

print_highlight("==== Original Output ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that I need to calculate the sum of 1 and 3.<br><br>I start with the number 1.<br><br>Then, I add 3 to it.<br><br>After performing the addition, I find that the total is 4.<br></think><br><br>**Solution:**<br><br>To find the sum of \(1 + 3\), follow these simple steps:<br><br>1. **Start with the number 1.**<br>2. **Add 3 to it.**<br>3. **Calculate the total.**<br><br>\[<br>1 + 3 = \boxed{4}<br>\]</strong>


### SGLang Native API 


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)

gen_url = f"http://localhost:{port}/generate"
gen_data = {
    "text": input,
    "sampling_params": {
        "skip_special_tokens": False,
        "max_new_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    },
}
gen_response = requests.post(gen_url, json=gen_data).json()["text"]

print_highlight("==== Original Output ====")
print_highlight(gen_response)

parse_url = f"http://localhost:{port}/separate_reasoning"
separate_reasoning_data = {
    "text": gen_response,
    "reasoning_parser": "deepseek-r1",
}
separate_reasoning_response_json = requests.post(
    parse_url, json=separate_reasoning_data
).json()
print_highlight("==== Reasoning ====")
print_highlight(separate_reasoning_response_json["reasoning_text"])
print_highlight("==== Text ====")
print_highlight(separate_reasoning_response_json["text"])
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I'll add the two numbers together: 1 plus 3.<br><br>Finally, the result of this addition is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I'll add the two numbers together: 1 plus 3.<br><br>Finally, the result of this addition is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



```python
terminate_process(server_process)
```

### Offline Engine API


```python
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.utils import print_highlight

llm = sgl.Engine(model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
sampling_params = {
    "max_new_tokens": 1024,
    "skip_special_tokens": False,
    "temperature": 0.6,
    "top_p": 0.95,
}
result = llm.generate(prompt=input, sampling_params=sampling_params)

generated_text = result["text"]  # Assume there is only one prompt

print_highlight("==== Original Output ====")
print_highlight(generated_text)

parser = ReasoningParser("deepseek-r1")
reasoning_text, text = parser.parse_non_stream(generated_text)
print_highlight("==== Reasoning ====")
print_highlight(reasoning_text)
print_highlight("==== Text ====")
print_highlight(text)
```

    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.57s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.30s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:34,  5.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:34,  5.86s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:30,  2.69s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:30,  2.69s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:31,  1.66s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:31,  1.66s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:02,  1.16s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:02,  1.16s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:45,  1.15it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:45,  1.15it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:35,  1.46it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:35,  1.46it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:28,  1.80it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:28,  1.80it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:23,  2.15it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:23,  2.15it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:19,  2.55it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:19,  2.55it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:16,  2.97it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:16,  2.97it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:13,  3.38it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:13,  3.38it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:12,  3.82it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:12,  3.82it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:10,  4.26it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:10,  4.26it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.70it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.70it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:08,  5.15it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:08,  5.15it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:07,  5.68it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:07,  5.68it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.30it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.30it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:10<00:05,  7.01it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:10<00:05,  7.01it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:10<00:05,  7.01it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:10<00:04,  8.31it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:10<00:04,  8.31it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:10<00:04,  8.31it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:10<00:03,  9.95it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:10<00:03,  9.95it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:10<00:03,  9.95it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:10<00:02, 11.52it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:10<00:02, 11.52it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:10<00:02, 11.52it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:10<00:02, 13.17it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:10<00:02, 13.17it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:10<00:02, 13.17it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:10<00:02, 14.74it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:10<00:02, 14.74it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:10<00:02, 14.74it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:10<00:02, 14.74it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:10<00:01, 17.43it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:10<00:01, 17.43it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:10<00:01, 17.43it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:10<00:01, 17.43it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:11<00:01, 18.94it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:11<00:01, 18.94it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:11<00:01, 18.94it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:11<00:01, 18.94it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:11<00:01, 20.64it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:11<00:01, 20.64it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:11<00:01, 20.64it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:11<00:01, 20.64it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:11<00:01, 20.64it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:11<00:00, 23.41it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:11<00:00, 23.41it/s]

    Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:11<00:00, 23.41it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:11<00:00, 23.41it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:11<00:00, 24.23it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:11<00:00, 24.23it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:11<00:00, 24.23it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:11<00:00, 24.23it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:11<00:00, 25.41it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:11<00:00, 25.41it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:11<00:00, 25.41it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:11<00:00, 25.41it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:11<00:00, 25.54it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:11<00:00, 25.54it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:11<00:00, 25.54it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:11<00:00, 25.54it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:11<00:00, 25.54it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:11<00:00, 28.07it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:11<00:00, 28.07it/s]

    Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:11<00:00, 28.07it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:11<00:00, 28.07it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:11<00:00, 28.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00, 30.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  4.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.88 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.88 GB):   2%|▏         | 1/58 [00:00<00:37,  1.50it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.84 GB):   2%|▏         | 1/58 [00:00<00:37,  1.50it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.84 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.84 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.84 GB):   5%|▌         | 3/58 [00:01<00:31,  1.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.84 GB):   5%|▌         | 3/58 [00:01<00:31,  1.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.84 GB):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.21 GB):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.21 GB):   9%|▊         | 5/58 [00:02<00:27,  1.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=24.21 GB):   9%|▊         | 5/58 [00:02<00:27,  1.90it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=24.21 GB):  10%|█         | 6/58 [00:03<00:24,  2.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=24.17 GB):  10%|█         | 6/58 [00:03<00:24,  2.09it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=24.17 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.17 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.25it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=24.17 GB):  14%|█▍        | 8/58 [00:03<00:20,  2.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=24.17 GB):  14%|█▍        | 8/58 [00:03<00:20,  2.46it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=24.17 GB):  16%|█▌        | 9/58 [00:04<00:18,  2.66it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.17 GB):  16%|█▌        | 9/58 [00:04<00:18,  2.66it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=24.17 GB):  17%|█▋        | 10/58 [00:04<00:17,  2.77it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.17 GB):  17%|█▋        | 10/58 [00:04<00:17,  2.77it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=24.17 GB):  19%|█▉        | 11/58 [00:04<00:17,  2.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.17 GB):  19%|█▉        | 11/58 [00:04<00:17,  2.68it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=24.17 GB):  21%|██        | 12/58 [00:05<00:17,  2.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.16 GB):  21%|██        | 12/58 [00:05<00:17,  2.61it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=24.16 GB):  22%|██▏       | 13/58 [00:05<00:17,  2.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=24.16 GB):  22%|██▏       | 13/58 [00:05<00:17,  2.58it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=24.16 GB):  24%|██▍       | 14/58 [00:05<00:15,  2.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=24.16 GB):  24%|██▍       | 14/58 [00:05<00:15,  2.77it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=24.16 GB):  26%|██▌       | 15/58 [00:06<00:16,  2.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.68 GB):  26%|██▌       | 15/58 [00:06<00:16,  2.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.68 GB):  28%|██▊       | 16/58 [00:06<00:13,  3.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.68 GB):  28%|██▊       | 16/58 [00:06<00:13,  3.16it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=42.68 GB):  29%|██▉       | 17/58 [00:06<00:10,  3.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.68 GB):  29%|██▉       | 17/58 [00:06<00:10,  3.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.68 GB):  31%|███       | 18/58 [00:06<00:08,  4.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.67 GB):  31%|███       | 18/58 [00:06<00:08,  4.51it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=42.67 GB):  33%|███▎      | 19/58 [00:06<00:07,  5.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.67 GB):  33%|███▎      | 19/58 [00:06<00:07,  5.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.66 GB):  33%|███▎      | 19/58 [00:07<00:07,  5.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.66 GB):  36%|███▌      | 21/58 [00:07<00:05,  6.64it/s]Capturing num tokens (num_tokens=960 avail_mem=42.66 GB):  36%|███▌      | 21/58 [00:07<00:05,  6.64it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=42.66 GB):  38%|███▊      | 22/58 [00:07<00:05,  7.17it/s]Capturing num tokens (num_tokens=896 avail_mem=42.65 GB):  38%|███▊      | 22/58 [00:07<00:05,  7.17it/s]Capturing num tokens (num_tokens=832 avail_mem=42.65 GB):  38%|███▊      | 22/58 [00:07<00:05,  7.17it/s]Capturing num tokens (num_tokens=832 avail_mem=42.65 GB):  41%|████▏     | 24/58 [00:07<00:03,  8.66it/s]Capturing num tokens (num_tokens=768 avail_mem=42.65 GB):  41%|████▏     | 24/58 [00:07<00:03,  8.66it/s]

    Capturing num tokens (num_tokens=704 avail_mem=42.64 GB):  41%|████▏     | 24/58 [00:07<00:03,  8.66it/s]Capturing num tokens (num_tokens=704 avail_mem=42.64 GB):  45%|████▍     | 26/58 [00:07<00:03,  9.56it/s]Capturing num tokens (num_tokens=640 avail_mem=42.64 GB):  45%|████▍     | 26/58 [00:07<00:03,  9.56it/s]Capturing num tokens (num_tokens=576 avail_mem=42.64 GB):  45%|████▍     | 26/58 [00:07<00:03,  9.56it/s]

    Capturing num tokens (num_tokens=576 avail_mem=42.64 GB):  48%|████▊     | 28/58 [00:07<00:02, 10.77it/s]Capturing num tokens (num_tokens=512 avail_mem=42.63 GB):  48%|████▊     | 28/58 [00:07<00:02, 10.77it/s]Capturing num tokens (num_tokens=480 avail_mem=42.63 GB):  48%|████▊     | 28/58 [00:07<00:02, 10.77it/s]Capturing num tokens (num_tokens=480 avail_mem=42.63 GB):  52%|█████▏    | 30/58 [00:07<00:02, 11.41it/s]Capturing num tokens (num_tokens=448 avail_mem=42.63 GB):  52%|█████▏    | 30/58 [00:07<00:02, 11.41it/s]

    Capturing num tokens (num_tokens=416 avail_mem=42.62 GB):  52%|█████▏    | 30/58 [00:07<00:02, 11.41it/s]Capturing num tokens (num_tokens=416 avail_mem=42.62 GB):  55%|█████▌    | 32/58 [00:08<00:02, 12.31it/s]Capturing num tokens (num_tokens=384 avail_mem=42.62 GB):  55%|█████▌    | 32/58 [00:08<00:02, 12.31it/s]Capturing num tokens (num_tokens=352 avail_mem=42.61 GB):  55%|█████▌    | 32/58 [00:08<00:02, 12.31it/s]Capturing num tokens (num_tokens=352 avail_mem=42.61 GB):  59%|█████▊    | 34/58 [00:08<00:01, 13.25it/s]Capturing num tokens (num_tokens=320 avail_mem=42.61 GB):  59%|█████▊    | 34/58 [00:08<00:01, 13.25it/s]

    Capturing num tokens (num_tokens=288 avail_mem=42.61 GB):  59%|█████▊    | 34/58 [00:08<00:01, 13.25it/s]Capturing num tokens (num_tokens=288 avail_mem=42.61 GB):  62%|██████▏   | 36/58 [00:08<00:01, 14.26it/s]Capturing num tokens (num_tokens=256 avail_mem=42.61 GB):  62%|██████▏   | 36/58 [00:08<00:01, 14.26it/s]Capturing num tokens (num_tokens=240 avail_mem=42.60 GB):  62%|██████▏   | 36/58 [00:08<00:01, 14.26it/s]Capturing num tokens (num_tokens=240 avail_mem=42.60 GB):  66%|██████▌   | 38/58 [00:08<00:01, 14.70it/s]Capturing num tokens (num_tokens=224 avail_mem=42.60 GB):  66%|██████▌   | 38/58 [00:08<00:01, 14.70it/s]

    Capturing num tokens (num_tokens=208 avail_mem=42.60 GB):  66%|██████▌   | 38/58 [00:08<00:01, 14.70it/s]Capturing num tokens (num_tokens=208 avail_mem=42.60 GB):  69%|██████▉   | 40/58 [00:08<00:01, 14.24it/s]Capturing num tokens (num_tokens=192 avail_mem=42.59 GB):  69%|██████▉   | 40/58 [00:08<00:01, 14.24it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.19 GB):  69%|██████▉   | 40/58 [00:08<00:01, 14.24it/s]Capturing num tokens (num_tokens=176 avail_mem=43.19 GB):  72%|███████▏  | 42/58 [00:08<00:01, 11.04it/s]Capturing num tokens (num_tokens=160 avail_mem=43.18 GB):  72%|███████▏  | 42/58 [00:08<00:01, 11.04it/s]Capturing num tokens (num_tokens=144 avail_mem=43.18 GB):  72%|███████▏  | 42/58 [00:08<00:01, 11.04it/s]Capturing num tokens (num_tokens=144 avail_mem=43.18 GB):  76%|███████▌  | 44/58 [00:08<00:01, 11.72it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.18 GB):  76%|███████▌  | 44/58 [00:08<00:01, 11.72it/s]Capturing num tokens (num_tokens=112 avail_mem=43.18 GB):  76%|███████▌  | 44/58 [00:09<00:01, 11.72it/s]Capturing num tokens (num_tokens=112 avail_mem=43.18 GB):  79%|███████▉  | 46/58 [00:09<00:00, 12.43it/s]Capturing num tokens (num_tokens=96 avail_mem=43.17 GB):  79%|███████▉  | 46/58 [00:09<00:00, 12.43it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=43.17 GB):  79%|███████▉  | 46/58 [00:09<00:00, 12.43it/s]Capturing num tokens (num_tokens=80 avail_mem=43.17 GB):  83%|████████▎ | 48/58 [00:09<00:00, 12.82it/s]Capturing num tokens (num_tokens=64 avail_mem=43.16 GB):  83%|████████▎ | 48/58 [00:09<00:00, 12.82it/s]Capturing num tokens (num_tokens=48 avail_mem=43.16 GB):  83%|████████▎ | 48/58 [00:09<00:00, 12.82it/s]Capturing num tokens (num_tokens=48 avail_mem=43.16 GB):  86%|████████▌ | 50/58 [00:09<00:00, 14.05it/s]Capturing num tokens (num_tokens=32 avail_mem=43.16 GB):  86%|████████▌ | 50/58 [00:09<00:00, 14.05it/s]

    Capturing num tokens (num_tokens=28 avail_mem=43.16 GB):  86%|████████▌ | 50/58 [00:09<00:00, 14.05it/s]Capturing num tokens (num_tokens=28 avail_mem=43.16 GB):  90%|████████▉ | 52/58 [00:09<00:00, 13.90it/s]Capturing num tokens (num_tokens=24 avail_mem=43.15 GB):  90%|████████▉ | 52/58 [00:09<00:00, 13.90it/s]Capturing num tokens (num_tokens=20 avail_mem=43.15 GB):  90%|████████▉ | 52/58 [00:09<00:00, 13.90it/s]

    Capturing num tokens (num_tokens=20 avail_mem=43.15 GB):  93%|█████████▎| 54/58 [00:09<00:00, 13.34it/s]Capturing num tokens (num_tokens=16 avail_mem=43.15 GB):  93%|█████████▎| 54/58 [00:09<00:00, 13.34it/s]Capturing num tokens (num_tokens=12 avail_mem=43.14 GB):  93%|█████████▎| 54/58 [00:09<00:00, 13.34it/s]Capturing num tokens (num_tokens=12 avail_mem=43.14 GB):  97%|█████████▋| 56/58 [00:09<00:00, 14.10it/s]Capturing num tokens (num_tokens=8 avail_mem=43.14 GB):  97%|█████████▋| 56/58 [00:09<00:00, 14.10it/s] Capturing num tokens (num_tokens=4 avail_mem=43.14 GB):  97%|█████████▋| 56/58 [00:09<00:00, 14.10it/s]

    Capturing num tokens (num_tokens=4 avail_mem=43.14 GB): 100%|██████████| 58/58 [00:09<00:00, 14.46it/s]Capturing num tokens (num_tokens=4 avail_mem=43.14 GB): 100%|██████████| 58/58 [00:09<00:00,  5.83it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I perform the addition by combining the two numbers: 1 plus 3 equals 4.<br><br>Finally, I present the answer as the result of the addition.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I perform the addition by combining the two numbers: 1 plus 3 equals 4.<br><br>Finally, I present the answer as the result of the addition.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
