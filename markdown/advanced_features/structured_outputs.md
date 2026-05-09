# Structured Outputs

You can specify a JSON schema, [regular expression](https://en.wikipedia.org/wiki/Regular_expression) or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to constrain the model output. The model output will be guaranteed to follow the given constraints. Only one constraint parameter (`json_schema`, `regex`, or `ebnf`) can be specified for a request.

SGLang supports three grammar backends:

- [XGrammar](https://github.com/mlc-ai/xgrammar)(default): Supports JSON schema, regular expression, and EBNF constraints.
- [Outlines](https://github.com/dottxt-ai/outlines): Supports JSON schema and regular expression constraints.
- [Llguidance](https://github.com/guidance-ai/llguidance): Supports JSON schema, regular expression, and EBNF constraints.

We suggest using XGrammar for its better performance and utility. XGrammar currently uses the [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md). For more details, see [XGrammar technical overview](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar).

To use Outlines, simply add `--grammar-backend outlines` when launching the server.
To use llguidance, add `--grammar-backend llguidance`  when launching the server.
If no backend is specified, XGrammar will be used as the default.

For better output quality, **It's advisable to explicitly include instructions in the prompt to guide the model to generate the desired format.** For example, you can specify, 'Please generate the output in the following JSON format: ...'.


## OpenAI Compatible API


```python
import openai
import os

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"


server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-09 01:32:38] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-09 01:32:41] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-09 01:32:44] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-09 01:32:45] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-09 01:32:48] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-09 01:32:49] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-09 01:32:50] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.59it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.45it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.40it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.92it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.71it/s]


    2026-05-09 01:32:56,585 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 01:32:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:40,  5.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:40,  5.98s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:25,  2.60s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:25,  2.60s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:23,  1.52s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:23,  1.52s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:53,  1.00it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:53,  1.00it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:27,  1.91it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:27,  1.91it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:21,  2.38it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:21,  2.38it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:17,  2.93it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:17,  2.93it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:14,  3.45it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:14,  3.45it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:11,  4.28it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:11,  4.28it/s]

    Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:07<00:11,  4.28it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:07,  5.91it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:07,  5.91it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:07<00:07,  5.91it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:05,  7.37it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:05,  7.37it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:08<00:05,  7.37it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:04,  9.06it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:04,  9.06it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:08<00:04,  9.06it/s]

    Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:08<00:04,  9.06it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:03, 12.09it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:03, 12.09it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:03, 12.09it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:08<00:03, 12.09it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:08<00:03, 12.09it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:02, 16.84it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:02, 16.84it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:08<00:02, 16.84it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:08<00:02, 16.84it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:08<00:02, 16.84it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:08<00:02, 16.84it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:08<00:01, 23.55it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:08<00:01, 23.55it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:08<00:01, 23.55it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:08<00:01, 23.55it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:08<00:01, 23.55it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:08<00:01, 23.55it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:08<00:01, 23.55it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:08<00:00, 31.12it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:08<00:00, 31.12it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:08<00:00, 31.12it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:08<00:00, 31.12it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:08<00:00, 31.12it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:08<00:00, 31.12it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:08<00:00, 31.12it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:08<00:00, 31.12it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:08<00:00, 39.76it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:08<00:00, 39.76it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:08<00:00, 39.76it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:08<00:00, 39.76it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:08<00:00, 39.76it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:08<00:00, 39.76it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:08<00:00, 39.76it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:08<00:00, 39.76it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]

    Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:08<00:00, 46.12it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:08<00:00, 54.65it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:08<00:00, 54.65it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:08<00:00, 54.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=18.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=18.87 GB):   2%|▏         | 1/58 [00:00<00:19,  2.91it/s]Capturing num tokens (num_tokens=7680 avail_mem=18.84 GB):   2%|▏         | 1/58 [00:00<00:19,  2.91it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=18.84 GB):   3%|▎         | 2/58 [00:00<00:18,  3.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=18.84 GB):   3%|▎         | 2/58 [00:00<00:18,  3.08it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=18.84 GB):   5%|▌         | 3/58 [00:00<00:16,  3.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=18.84 GB):   5%|▌         | 3/58 [00:00<00:16,  3.31it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=18.84 GB):   7%|▋         | 4/58 [00:01<00:15,  3.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=18.79 GB):   7%|▋         | 4/58 [00:01<00:15,  3.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=18.79 GB):   9%|▊         | 5/58 [00:01<00:14,  3.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=18.79 GB):   9%|▊         | 5/58 [00:01<00:14,  3.62it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=18.79 GB):  10%|█         | 6/58 [00:01<00:13,  3.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=18.79 GB):  10%|█         | 6/58 [00:01<00:13,  3.89it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=18.79 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=18.79 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=18.79 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=18.77 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.48it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=18.77 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=18.73 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=18.73 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=18.70 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.02it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=18.70 GB):  19%|█▉        | 11/58 [00:02<00:10,  4.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=18.68 GB):  19%|█▉        | 11/58 [00:02<00:10,  4.28it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=18.68 GB):  21%|██        | 12/58 [00:03<00:11,  3.91it/s]Capturing num tokens (num_tokens=3072 avail_mem=18.18 GB):  21%|██        | 12/58 [00:03<00:11,  3.91it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=18.18 GB):  22%|██▏       | 13/58 [00:03<00:13,  3.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=18.17 GB):  22%|██▏       | 13/58 [00:03<00:13,  3.34it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=18.17 GB):  24%|██▍       | 14/58 [00:03<00:13,  3.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.37 GB):  24%|██▍       | 14/58 [00:03<00:13,  3.20it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=41.37 GB):  26%|██▌       | 15/58 [00:04<00:12,  3.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.37 GB):  26%|██▌       | 15/58 [00:04<00:12,  3.50it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=41.37 GB):  28%|██▊       | 16/58 [00:04<00:11,  3.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.37 GB):  28%|██▊       | 16/58 [00:04<00:11,  3.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.37 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.36 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.15it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=41.36 GB):  31%|███       | 18/58 [00:04<00:08,  4.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.36 GB):  31%|███       | 18/58 [00:04<00:08,  4.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.36 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.36 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.07it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=41.36 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.36 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.36 GB):  36%|███▌      | 21/58 [00:04<00:06,  6.16it/s]Capturing num tokens (num_tokens=960 avail_mem=41.34 GB):  36%|███▌      | 21/58 [00:04<00:06,  6.16it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=41.34 GB):  38%|███▊      | 22/58 [00:05<00:05,  6.70it/s]Capturing num tokens (num_tokens=896 avail_mem=41.34 GB):  38%|███▊      | 22/58 [00:05<00:05,  6.70it/s]Capturing num tokens (num_tokens=896 avail_mem=41.34 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.18it/s]Capturing num tokens (num_tokens=832 avail_mem=41.34 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.18it/s]

    Capturing num tokens (num_tokens=832 avail_mem=41.34 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.56it/s]Capturing num tokens (num_tokens=768 avail_mem=41.33 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.56it/s]Capturing num tokens (num_tokens=768 avail_mem=41.33 GB):  43%|████▎     | 25/58 [00:05<00:04,  7.71it/s]Capturing num tokens (num_tokens=704 avail_mem=41.33 GB):  43%|████▎     | 25/58 [00:05<00:04,  7.71it/s]

    Capturing num tokens (num_tokens=704 avail_mem=41.33 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.12it/s]Capturing num tokens (num_tokens=640 avail_mem=41.32 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.12it/s]Capturing num tokens (num_tokens=640 avail_mem=41.32 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.59it/s]Capturing num tokens (num_tokens=576 avail_mem=41.32 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.59it/s]Capturing num tokens (num_tokens=512 avail_mem=41.32 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.59it/s]

    Capturing num tokens (num_tokens=512 avail_mem=41.32 GB):  50%|█████     | 29/58 [00:05<00:03,  9.41it/s]Capturing num tokens (num_tokens=480 avail_mem=41.31 GB):  50%|█████     | 29/58 [00:05<00:03,  9.41it/s]Capturing num tokens (num_tokens=480 avail_mem=41.31 GB):  52%|█████▏    | 30/58 [00:05<00:02,  9.48it/s]Capturing num tokens (num_tokens=448 avail_mem=41.31 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.48it/s]

    Capturing num tokens (num_tokens=448 avail_mem=41.31 GB):  53%|█████▎    | 31/58 [00:06<00:03,  8.56it/s]Capturing num tokens (num_tokens=416 avail_mem=41.31 GB):  53%|█████▎    | 31/58 [00:06<00:03,  8.56it/s]Capturing num tokens (num_tokens=384 avail_mem=41.30 GB):  53%|█████▎    | 31/58 [00:06<00:03,  8.56it/s]Capturing num tokens (num_tokens=384 avail_mem=41.30 GB):  57%|█████▋    | 33/58 [00:06<00:02,  9.12it/s]

    Capturing num tokens (num_tokens=352 avail_mem=41.30 GB):  57%|█████▋    | 33/58 [00:06<00:02,  9.12it/s]Capturing num tokens (num_tokens=320 avail_mem=41.30 GB):  57%|█████▋    | 33/58 [00:06<00:02,  9.12it/s]Capturing num tokens (num_tokens=320 avail_mem=41.30 GB):  60%|██████    | 35/58 [00:06<00:02, 10.85it/s]Capturing num tokens (num_tokens=288 avail_mem=41.29 GB):  60%|██████    | 35/58 [00:06<00:02, 10.85it/s]Capturing num tokens (num_tokens=256 avail_mem=41.29 GB):  60%|██████    | 35/58 [00:06<00:02, 10.85it/s]

    Capturing num tokens (num_tokens=240 avail_mem=41.28 GB):  60%|██████    | 35/58 [00:06<00:02, 10.85it/s]Capturing num tokens (num_tokens=240 avail_mem=41.28 GB):  66%|██████▌   | 38/58 [00:06<00:01, 14.36it/s]Capturing num tokens (num_tokens=224 avail_mem=41.28 GB):  66%|██████▌   | 38/58 [00:06<00:01, 14.36it/s]Capturing num tokens (num_tokens=208 avail_mem=41.27 GB):  66%|██████▌   | 38/58 [00:06<00:01, 14.36it/s]Capturing num tokens (num_tokens=192 avail_mem=41.27 GB):  66%|██████▌   | 38/58 [00:06<00:01, 14.36it/s]Capturing num tokens (num_tokens=192 avail_mem=41.27 GB):  71%|███████   | 41/58 [00:06<00:01, 16.27it/s]Capturing num tokens (num_tokens=176 avail_mem=41.26 GB):  71%|███████   | 41/58 [00:06<00:01, 16.27it/s]

    Capturing num tokens (num_tokens=160 avail_mem=41.26 GB):  71%|███████   | 41/58 [00:06<00:01, 16.27it/s]Capturing num tokens (num_tokens=160 avail_mem=41.26 GB):  74%|███████▍  | 43/58 [00:06<00:01, 14.06it/s]Capturing num tokens (num_tokens=144 avail_mem=41.25 GB):  74%|███████▍  | 43/58 [00:06<00:01, 14.06it/s]Capturing num tokens (num_tokens=128 avail_mem=41.25 GB):  74%|███████▍  | 43/58 [00:06<00:01, 14.06it/s]

    Capturing num tokens (num_tokens=128 avail_mem=41.25 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.42it/s]Capturing num tokens (num_tokens=112 avail_mem=41.25 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.42it/s]Capturing num tokens (num_tokens=96 avail_mem=41.25 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.42it/s] Capturing num tokens (num_tokens=96 avail_mem=41.25 GB):  81%|████████  | 47/58 [00:07<00:00, 12.95it/s]Capturing num tokens (num_tokens=80 avail_mem=41.25 GB):  81%|████████  | 47/58 [00:07<00:00, 12.95it/s]

    Capturing num tokens (num_tokens=64 avail_mem=41.24 GB):  81%|████████  | 47/58 [00:07<00:00, 12.95it/s]Capturing num tokens (num_tokens=64 avail_mem=41.24 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.65it/s]Capturing num tokens (num_tokens=48 avail_mem=41.24 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.65it/s]Capturing num tokens (num_tokens=32 avail_mem=41.24 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.65it/s]

    Capturing num tokens (num_tokens=32 avail_mem=41.24 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.80it/s]Capturing num tokens (num_tokens=28 avail_mem=41.23 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.80it/s]Capturing num tokens (num_tokens=24 avail_mem=41.23 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.80it/s]Capturing num tokens (num_tokens=24 avail_mem=41.23 GB):  91%|█████████▏| 53/58 [00:07<00:00, 12.98it/s]Capturing num tokens (num_tokens=20 avail_mem=41.22 GB):  91%|█████████▏| 53/58 [00:07<00:00, 12.98it/s]

    Capturing num tokens (num_tokens=16 avail_mem=41.22 GB):  91%|█████████▏| 53/58 [00:07<00:00, 12.98it/s]Capturing num tokens (num_tokens=16 avail_mem=41.22 GB):  95%|█████████▍| 55/58 [00:07<00:00, 13.06it/s]Capturing num tokens (num_tokens=12 avail_mem=41.21 GB):  95%|█████████▍| 55/58 [00:07<00:00, 13.06it/s]Capturing num tokens (num_tokens=8 avail_mem=41.21 GB):  95%|█████████▍| 55/58 [00:07<00:00, 13.06it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=41.21 GB):  98%|█████████▊| 57/58 [00:07<00:00, 12.81it/s]Capturing num tokens (num_tokens=4 avail_mem=41.21 GB):  98%|█████████▊| 57/58 [00:07<00:00, 12.81it/s]Capturing num tokens (num_tokens=4 avail_mem=41.21 GB): 100%|██████████| 58/58 [00:08<00:00,  7.17it/s]


    [2026-05-09 01:33:16] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-09 01:33:19] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


### JSON

you can directly define a JSON schema or use [Pydantic](https://docs.pydantic.dev/latest/) to define and validate the response.

**Using Pydantic**


```python
from pydantic import BaseModel, Field


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Please generate the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            # convert the pydantic model to json schema
            "schema": CapitalInfo.model_json_schema(),
        },
    },
)

response_content = response.choices[0].message.content
# validate the JSON response by the pydantic model
capital_info = CapitalInfo.model_validate_json(response_content)
print_highlight(f"Validated response: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>Validated response: {"name":"Paris","population":2147000}</strong>


**JSON Schema Directly**



```python
import json

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Give me the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>{"name": "Paris", "population": 2147000}</strong>


### EBNF


```python
ebnf_grammar = """
root ::= city | description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful geography bot."},
        {
            "role": "user",
            "content": "Give me the information of the capital of France.",
        },
    ],
    temperature=0,
    max_tokens=32,
    extra_body={"ebnf": ebnf_grammar},
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Paris is the capital of France</strong>


### Regular expression


```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0,
    max_tokens=128,
    extra_body={"regex": "(Paris|London)"},
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Paris</strong>


### Structural Tag


```python
tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

tool_get_current_date = {
    "type": "function",
    "function": {
        "name": "get_current_date",
        "description": "Get the current date and time for a given timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                }
            },
            "required": ["timezone"],
        },
    },
}

schema_get_current_weather = tool_get_current_weather["function"]["parameters"]
schema_get_current_date = tool_get_current_date["function"]["parameters"]


def get_messages():
    return [
        {
            "role": "system",
            "content": f"""
# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant.""",
        },
        {
            "role": "user",
            "content": "You are in New York. Please get the current date and time, and the weather.",
        },
    ]


messages = get_messages()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<function=get_current_weather>",
                "schema": schema_get_current_weather,
                "end": "</function>",
            },
            {
                "begin": "<function=get_current_date>",
                "schema": schema_get_current_date,
                "end": "</function>",
            },
        ],
        "triggers": ["<function="],
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Sources:<br>- get_current_date function<br>- get_current_weather function</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_current_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_get_current_weather,
                    },
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_get_current_date,
                    },
                    "end": "</function>",
                },
            ],
            "at_least_one": False,
            "stop_after_first": False,
        },
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


## Native API and SGLang Runtime (SRT)

### JSON

**Using Pydantic**


```python
import requests
import json
from pydantic import BaseModel, Field

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


# Make API request
messages = [
    {
        "role": "user",
        "content": "Here is the information of the capital of France in the JSON format.\n",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json.dumps(CapitalInfo.model_json_schema()),
        },
    },
)
print_highlight(response.json())


response_data = json.loads(response.json()["text"])
# validate the response by the pydantic model
capital_info = CapitalInfo.model_validate(response_data)
print_highlight(f"Validated response: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': 'd6b6c3cda49640bcbb9d667484eea2d7', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.24813836999237537, 'response_sent_to_client_ts': 1778290411.599608}}</strong>



<strong style='color: #00008B;'>Validated response: {"name":"Paris","population":2147000}</strong>


**JSON Schema Directly**


```python
json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

# JSON
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '52bc1c16b180464a84c0c0b8558b4fdf', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 49, 'cached_tokens_details': {'device': 49, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2505022040568292, 'response_sent_to_client_ts': 1778290411.8604732}}</strong>


### EBNF


```python
messages = [
    {
        "role": "user",
        "content": "Give me the information of the capital of France.",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "max_new_tokens": 128,
            "temperature": 0,
            "n": 3,
            "ebnf": (
                "root ::= city | description\n"
                'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
                'description ::= city " is " status\n'
                'status ::= "the capital of " country\n'
                'country ::= "England" | "France" | "Germany" | "Italy"'
            ),
        },
        "stream": False,
        "return_logprob": False,
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>[{'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': 'b59dfc868e2e4024a225d89afcbcf268', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12494699493981898, 'response_sent_to_client_ts': 1778290412.0321887}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': 'f2575e038b34460d9577ff9f71c771d5', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12485722196288407, 'response_sent_to_client_ts': 1778290412.0322008}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': 'cb2a2cd41f564ed2b0799af72ea98c70', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12481168401427567, 'response_sent_to_client_ts': 1778290412.032205}}]</strong>


### Regular expression


```python
messages = [
    {
        "role": "user",
        "content": "Paris is the capital of",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France', 'output_ids': [50100, 128009], 'meta_info': {'id': 'b8be23e279d241f4afcae3b69e12b67b', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 2, 'cached_tokens': 31, 'cached_tokens_details': {'device': 31, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.05464552901685238, 'response_sent_to_client_ts': 1778290412.0948617}}</strong>


### Structural Tag


```python
from transformers import AutoTokenizer

# generate an answer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
payload = {
    "text": text,
    "sampling_params": {
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<function=get_current_weather>",
                        "schema": schema_get_current_weather,
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "schema": schema_get_current_date,
                        "end": "</function>",
                    },
                ],
                "triggers": ["<function="],
            }
        )
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': 'b233e5d6aded49fda8f2d8d2e984b640', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1313397060148418, 'response_sent_to_client_ts': 1778290413.8247564}}</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
payload = {
    "text": text,
    "sampling_params": {
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "triggered_tags",
                    "triggers": ["<function="],
                    "tags": [
                        {
                            "begin": "<function=get_current_weather>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": schema_get_current_weather,
                            },
                            "end": "</function>",
                        },
                        {
                            "begin": "<function=get_current_date>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": schema_get_current_date,
                            },
                            "end": "</function>",
                        },
                    ],
                    "at_least_one": False,
                    "stop_after_first": False,
                },
            }
        )
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': 'ffb4a34ca08b48e29d927bb5f25e3970', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.07298092404380441, 'response_sent_to_client_ts': 1778290413.9062402}}</strong>



```python
terminate_process(server_process)
```

## Offline Engine API


```python
import sglang as sgl

llm = sgl.Engine(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", grammar_backend="xgrammar"
)
```

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.34it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.31it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.30it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.79it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.58it/s]


    2026-05-09 01:33:56,646 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 01:33:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:36,  5.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:36,  5.91s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:42,  2.91s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:42,  2.91s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:44,  1.91s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:44,  1.91s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:08<01:16,  1.41s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:08<01:16,  1.41s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:08<00:59,  1.13s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:08<00:59,  1.13s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:09<00:48,  1.08it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:09<00:48,  1.08it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:09<00:39,  1.28it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:09<00:39,  1.28it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:10<00:33,  1.48it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:10<00:33,  1.48it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:10<00:28,  1.70it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:10<00:28,  1.70it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:10<00:24,  1.94it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:10<00:24,  1.94it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:11<00:21,  2.17it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:11<00:21,  2.17it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:11<00:19,  2.40it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:11<00:19,  2.40it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:11<00:16,  2.66it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:11<00:16,  2.66it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:12<00:15,  2.92it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:12<00:15,  2.92it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:12<00:13,  3.23it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:12<00:13,  3.23it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:12<00:11,  3.60it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:12<00:11,  3.60it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:12<00:10,  3.96it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:12<00:10,  3.96it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:12<00:08,  4.46it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:12<00:08,  4.46it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:13<00:07,  4.95it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:13<00:07,  4.95it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:13<00:06,  5.54it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:13<00:06,  5.54it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:13<00:05,  6.28it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:13<00:05,  6.28it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:13<00:05,  6.28it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:13<00:04,  7.63it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:13<00:04,  7.63it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:13<00:04,  7.63it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:13<00:03,  8.73it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:13<00:03,  8.73it/s]

    Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:13<00:03,  8.73it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:13<00:03,  9.85it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:13<00:03,  9.85it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:13<00:03,  9.85it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:13<00:02, 11.03it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:13<00:02, 11.03it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:14<00:02, 11.03it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:14<00:02, 12.18it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:14<00:02, 12.18it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:14<00:02, 12.18it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:14<00:02, 12.36it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:14<00:02, 12.36it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:14<00:02, 12.36it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:14<00:01, 13.20it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:14<00:01, 13.20it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:14<00:01, 13.20it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:14<00:01, 14.36it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:14<00:01, 14.36it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:14<00:01, 14.36it/s]

    Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:14<00:01, 15.05it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:14<00:01, 15.05it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:14<00:01, 15.05it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:14<00:01, 15.63it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:14<00:01, 15.63it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:14<00:01, 15.63it/s]

    Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:14<00:00, 15.99it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:14<00:00, 15.99it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:14<00:00, 15.99it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:14<00:00, 16.15it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:14<00:00, 16.15it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:15<00:00, 16.15it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:15<00:00, 16.33it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:15<00:00, 16.33it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:15<00:00, 16.33it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:15<00:00, 16.46it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:15<00:00, 16.46it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:15<00:00, 16.46it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:15<00:00, 16.56it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:15<00:00, 16.56it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:15<00:00, 16.56it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:15<00:00, 16.56it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:15<00:00, 18.14it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:15<00:00, 18.14it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:15<00:00, 18.14it/s]

    Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:15<00:00, 18.57it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:15<00:00, 18.57it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:15<00:00, 18.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:15<00:00,  3.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=20.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=20.29 GB):   2%|▏         | 1/58 [00:01<01:07,  1.18s/it]Capturing num tokens (num_tokens=7680 avail_mem=20.26 GB):   2%|▏         | 1/58 [00:01<01:07,  1.18s/it]

    Capturing num tokens (num_tokens=7680 avail_mem=20.26 GB):   3%|▎         | 2/58 [00:02<01:03,  1.14s/it]Capturing num tokens (num_tokens=7168 avail_mem=20.26 GB):   3%|▎         | 2/58 [00:02<01:03,  1.14s/it]

    Capturing num tokens (num_tokens=7168 avail_mem=20.26 GB):   5%|▌         | 3/58 [00:03<00:57,  1.04s/it]Capturing num tokens (num_tokens=6656 avail_mem=20.26 GB):   5%|▌         | 3/58 [00:03<00:57,  1.04s/it]

    Capturing num tokens (num_tokens=6656 avail_mem=20.26 GB):   7%|▋         | 4/58 [00:04<00:52,  1.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=20.26 GB):   7%|▋         | 4/58 [00:04<00:52,  1.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=20.26 GB):   9%|▊         | 5/58 [00:04<00:46,  1.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=20.26 GB):   9%|▊         | 5/58 [00:04<00:46,  1.15it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=20.26 GB):  10%|█         | 6/58 [00:05<00:42,  1.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=20.26 GB):  10%|█         | 6/58 [00:05<00:42,  1.24it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=20.26 GB):  12%|█▏        | 7/58 [00:06<00:38,  1.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.26 GB):  12%|█▏        | 7/58 [00:06<00:38,  1.32it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=20.26 GB):  14%|█▍        | 8/58 [00:06<00:34,  1.43it/s]Capturing num tokens (num_tokens=4096 avail_mem=20.26 GB):  14%|█▍        | 8/58 [00:06<00:34,  1.43it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=20.26 GB):  16%|█▌        | 9/58 [00:07<00:31,  1.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.26 GB):  16%|█▌        | 9/58 [00:07<00:31,  1.55it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=20.26 GB):  17%|█▋        | 10/58 [00:08<00:36,  1.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=20.26 GB):  17%|█▋        | 10/58 [00:08<00:36,  1.32it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=20.26 GB):  19%|█▉        | 11/58 [00:08<00:29,  1.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.26 GB):  19%|█▉        | 11/58 [00:08<00:29,  1.60it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=20.26 GB):  21%|██        | 12/58 [00:09<00:27,  1.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.35 GB):  21%|██        | 12/58 [00:09<00:27,  1.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.35 GB):  22%|██▏       | 13/58 [00:09<00:20,  2.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.35 GB):  22%|██▏       | 13/58 [00:09<00:20,  2.18it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=41.35 GB):  24%|██▍       | 14/58 [00:09<00:15,  2.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.70 GB):  24%|██▍       | 14/58 [00:09<00:15,  2.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.70 GB):  26%|██▌       | 15/58 [00:09<00:12,  3.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.70 GB):  26%|██▌       | 15/58 [00:09<00:12,  3.32it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=59.70 GB):  26%|██▌       | 15/58 [00:09<00:12,  3.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.70 GB):  29%|██▉       | 17/58 [00:09<00:08,  4.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.69 GB):  29%|██▉       | 17/58 [00:09<00:08,  4.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.69 GB):  29%|██▉       | 17/58 [00:09<00:08,  4.82it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=59.69 GB):  33%|███▎      | 19/58 [00:09<00:05,  6.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.69 GB):  33%|███▎      | 19/58 [00:09<00:05,  6.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.69 GB):  33%|███▎      | 19/58 [00:09<00:05,  6.62it/s]Capturing num tokens (num_tokens=960 avail_mem=59.67 GB):  33%|███▎      | 19/58 [00:09<00:05,  6.62it/s] Capturing num tokens (num_tokens=960 avail_mem=59.67 GB):  38%|███▊      | 22/58 [00:09<00:03,  9.70it/s]Capturing num tokens (num_tokens=896 avail_mem=59.67 GB):  38%|███▊      | 22/58 [00:09<00:03,  9.70it/s]Capturing num tokens (num_tokens=832 avail_mem=59.67 GB):  38%|███▊      | 22/58 [00:10<00:03,  9.70it/s]

    Capturing num tokens (num_tokens=768 avail_mem=59.66 GB):  38%|███▊      | 22/58 [00:10<00:03,  9.70it/s]Capturing num tokens (num_tokens=768 avail_mem=59.66 GB):  43%|████▎     | 25/58 [00:10<00:02, 12.63it/s]Capturing num tokens (num_tokens=704 avail_mem=59.66 GB):  43%|████▎     | 25/58 [00:10<00:02, 12.63it/s]Capturing num tokens (num_tokens=640 avail_mem=59.65 GB):  43%|████▎     | 25/58 [00:10<00:02, 12.63it/s]Capturing num tokens (num_tokens=576 avail_mem=59.65 GB):  43%|████▎     | 25/58 [00:10<00:02, 12.63it/s]Capturing num tokens (num_tokens=576 avail_mem=59.65 GB):  48%|████▊     | 28/58 [00:10<00:01, 15.60it/s]Capturing num tokens (num_tokens=512 avail_mem=59.65 GB):  48%|████▊     | 28/58 [00:10<00:01, 15.60it/s]Capturing num tokens (num_tokens=480 avail_mem=59.64 GB):  48%|████▊     | 28/58 [00:10<00:01, 15.60it/s]

    Capturing num tokens (num_tokens=448 avail_mem=59.64 GB):  48%|████▊     | 28/58 [00:10<00:01, 15.60it/s]Capturing num tokens (num_tokens=448 avail_mem=59.64 GB):  53%|█████▎    | 31/58 [00:10<00:01, 18.30it/s]Capturing num tokens (num_tokens=416 avail_mem=59.64 GB):  53%|█████▎    | 31/58 [00:10<00:01, 18.30it/s]Capturing num tokens (num_tokens=384 avail_mem=59.63 GB):  53%|█████▎    | 31/58 [00:10<00:01, 18.30it/s]Capturing num tokens (num_tokens=352 avail_mem=59.63 GB):  53%|█████▎    | 31/58 [00:10<00:01, 18.30it/s]Capturing num tokens (num_tokens=352 avail_mem=59.63 GB):  59%|█████▊    | 34/58 [00:10<00:01, 20.84it/s]Capturing num tokens (num_tokens=320 avail_mem=59.63 GB):  59%|█████▊    | 34/58 [00:10<00:01, 20.84it/s]Capturing num tokens (num_tokens=288 avail_mem=59.62 GB):  59%|█████▊    | 34/58 [00:10<00:01, 20.84it/s]

    Capturing num tokens (num_tokens=256 avail_mem=59.62 GB):  59%|█████▊    | 34/58 [00:10<00:01, 20.84it/s]Capturing num tokens (num_tokens=240 avail_mem=59.61 GB):  59%|█████▊    | 34/58 [00:10<00:01, 20.84it/s]Capturing num tokens (num_tokens=240 avail_mem=59.61 GB):  66%|██████▌   | 38/58 [00:10<00:00, 23.92it/s]Capturing num tokens (num_tokens=224 avail_mem=59.61 GB):  66%|██████▌   | 38/58 [00:10<00:00, 23.92it/s]Capturing num tokens (num_tokens=208 avail_mem=59.60 GB):  66%|██████▌   | 38/58 [00:10<00:00, 23.92it/s]Capturing num tokens (num_tokens=192 avail_mem=59.60 GB):  66%|██████▌   | 38/58 [00:10<00:00, 23.92it/s]Capturing num tokens (num_tokens=176 avail_mem=59.59 GB):  66%|██████▌   | 38/58 [00:10<00:00, 23.92it/s]Capturing num tokens (num_tokens=176 avail_mem=59.59 GB):  72%|███████▏  | 42/58 [00:10<00:00, 26.50it/s]Capturing num tokens (num_tokens=160 avail_mem=59.59 GB):  72%|███████▏  | 42/58 [00:10<00:00, 26.50it/s]

    Capturing num tokens (num_tokens=144 avail_mem=59.59 GB):  72%|███████▏  | 42/58 [00:10<00:00, 26.50it/s]Capturing num tokens (num_tokens=128 avail_mem=59.58 GB):  72%|███████▏  | 42/58 [00:10<00:00, 26.50it/s]Capturing num tokens (num_tokens=112 avail_mem=59.59 GB):  72%|███████▏  | 42/58 [00:10<00:00, 26.50it/s]Capturing num tokens (num_tokens=112 avail_mem=59.59 GB):  79%|███████▉  | 46/58 [00:10<00:00, 28.57it/s]Capturing num tokens (num_tokens=96 avail_mem=59.58 GB):  79%|███████▉  | 46/58 [00:10<00:00, 28.57it/s] Capturing num tokens (num_tokens=80 avail_mem=59.58 GB):  79%|███████▉  | 46/58 [00:10<00:00, 28.57it/s]Capturing num tokens (num_tokens=64 avail_mem=59.57 GB):  79%|███████▉  | 46/58 [00:10<00:00, 28.57it/s]Capturing num tokens (num_tokens=48 avail_mem=59.57 GB):  79%|███████▉  | 46/58 [00:10<00:00, 28.57it/s]

    Capturing num tokens (num_tokens=48 avail_mem=59.57 GB):  86%|████████▌ | 50/58 [00:10<00:00, 30.06it/s]Capturing num tokens (num_tokens=32 avail_mem=59.57 GB):  86%|████████▌ | 50/58 [00:10<00:00, 30.06it/s]Capturing num tokens (num_tokens=28 avail_mem=59.56 GB):  86%|████████▌ | 50/58 [00:10<00:00, 30.06it/s]Capturing num tokens (num_tokens=24 avail_mem=59.56 GB):  86%|████████▌ | 50/58 [00:10<00:00, 30.06it/s]Capturing num tokens (num_tokens=20 avail_mem=59.55 GB):  86%|████████▌ | 50/58 [00:11<00:00, 30.06it/s]Capturing num tokens (num_tokens=20 avail_mem=59.55 GB):  93%|█████████▎| 54/58 [00:11<00:00, 31.00it/s]Capturing num tokens (num_tokens=16 avail_mem=59.55 GB):  93%|█████████▎| 54/58 [00:11<00:00, 31.00it/s]Capturing num tokens (num_tokens=12 avail_mem=59.54 GB):  93%|█████████▎| 54/58 [00:11<00:00, 31.00it/s]Capturing num tokens (num_tokens=8 avail_mem=59.54 GB):  93%|█████████▎| 54/58 [00:11<00:00, 31.00it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=59.54 GB):  93%|█████████▎| 54/58 [00:11<00:00, 31.00it/s]Capturing num tokens (num_tokens=4 avail_mem=59.54 GB): 100%|██████████| 58/58 [00:11<00:00, 30.48it/s]Capturing num tokens (num_tokens=4 avail_mem=59.54 GB): 100%|██████████| 58/58 [00:11<00:00,  5.19it/s]


### JSON

**Using Pydantic**


```python
import json
from pydantic import BaseModel, Field

prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


sampling_params = {
    "temperature": 0.1,
    "top_p": 0.95,
    "json_schema": json.dumps(CapitalInfo.model_json_schema()),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}")  # validate the output by the pydantic model
    capital_info = CapitalInfo.model_validate_json(output["text"])
    print_highlight(f"Validated output: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of China in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Beijing","population":21500000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Paris","population":2141000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Ireland in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Dublin","population":527617}</strong>


**JSON Schema Directly**


```python
prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

sampling_params = {"temperature": 0.1, "top_p": 0.95, "json_schema": json_schema}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of China in the JSON format.<br>Generated text: {"name": "Beijing", "population": 21500000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France in the JSON format.<br>Generated text: {"name": "Paris", "population": 2141000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Ireland in the JSON format.<br>Generated text: {"name": "Dublin", "population": 527617}</strong>


### EBNF



```python
prompts = [
    "Give me the information of the capital of France.",
    "Give me the information of the capital of Germany.",
    "Give me the information of the capital of Italy.",
]

sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "ebnf": (
        "root ::= city | description\n"
        'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
        'description ::= city " is " status\n'
        'status ::= "the capital of " country\n'
        'country ::= "England" | "France" | "Germany" | "Italy"'
    ),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France.<br>Generated text: Paris is the capital of France</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Germany.<br>Generated text: Berlin is the capital of Germany</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Italy.<br>Generated text: Paris is the capital of France</strong>


### Regular expression


```python
prompts = [
    "Please provide information about London as a major global city:",
    "Please provide information about Paris as a major global city:",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95, "regex": "(France|England)"}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Please provide information about London as a major global city:<br>Generated text: England</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Please provide information about Paris as a major global city:<br>Generated text: France</strong>


### Structural Tag


```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
prompts = [text]


sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_current_weather>",
                    "schema": schema_get_current_weather,
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "schema": schema_get_current_date,
                    "end": "</function>",
                },
            ],
            "triggers": ["<function="],
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|><br><br>Cutting Knowledge Date: December 2023<br>Today Date: 26 Jul 2024<br><br><|eot_id|><|start_header_id|>user<|end_header_id|><br><br>Paris is the capital of<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br><br>Generated text: France.</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=get_current_weather>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema_get_current_weather,
                        },
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema_get_current_date,
                        },
                        "end": "</function>",
                    },
                ],
                "at_least_one": False,
                "stop_after_first": False,
            },
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|><br><br>Cutting Knowledge Date: December 2023<br>Today Date: 26 Jul 2024<br><br><|eot_id|><|start_header_id|>user<|end_header_id|><br><br>Paris is the capital of<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br><br>Generated text: France.</strong>



```python
llm.shutdown()
```
