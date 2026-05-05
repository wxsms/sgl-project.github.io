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


    [2026-05-05 22:07:04] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-05 22:07:06] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-05 22:07:10] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-05 22:07:10] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-05 22:07:13] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.
    [2026-05-05 22:07:13] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.49it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.44it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.43it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.95it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.73it/s]


    2026-05-05 22:07:18,806 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 22:07:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:27,  2.63s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:27,  2.63s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:29,  1.64s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:29,  1.64s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:02,  1.16s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:02,  1.16s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:47,  1.13it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:28,  1.78it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:28,  1.78it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:23,  2.16it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:23,  2.16it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:18,  2.59it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:18,  2.59it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:15,  3.04it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:15,  3.04it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.46it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.46it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:11,  4.00it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:11,  4.00it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:09,  4.62it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:09,  4.62it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:08,  5.09it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:08,  5.09it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:07,  5.95it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:07,  5.95it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:06,  6.65it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:06,  6.65it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:09<00:06,  6.65it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:04,  8.28it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:04,  8.28it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:04,  8.28it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:03,  9.98it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:03,  9.98it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:03,  9.98it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:09<00:03,  9.98it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:10<00:02, 13.42it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:10<00:02, 13.42it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:10<00:02, 13.42it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:10<00:02, 13.42it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:10<00:01, 16.77it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:10<00:01, 16.77it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:10<00:01, 16.77it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:10<00:01, 16.77it/s]

    Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:10<00:01, 16.77it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:10<00:01, 21.74it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:10<00:01, 21.74it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:10<00:01, 21.74it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:10<00:01, 21.74it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:10<00:01, 21.74it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:10<00:00, 24.30it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:10<00:00, 24.30it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:10<00:00, 24.30it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:10<00:00, 24.30it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:10<00:00, 24.30it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:10<00:00, 24.30it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:10<00:00, 29.38it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:10<00:00, 29.38it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:10<00:00, 29.38it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:10<00:00, 29.38it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:10<00:00, 29.38it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:10<00:00, 28.87it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:10<00:00, 28.87it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:10<00:00, 28.87it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:10<00:00, 28.87it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:10<00:00, 27.45it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:10<00:00, 27.45it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:10<00:00, 27.45it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:10<00:00, 27.45it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:10<00:00, 26.88it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:10<00:00, 26.88it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:10<00:00, 26.88it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:10<00:00, 26.88it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:11<00:00, 26.88it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:11<00:00, 27.34it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:11<00:00, 27.34it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:11<00:00, 27.34it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:11<00:00, 27.34it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:11<00:00, 27.34it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:11<00:00, 27.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00, 32.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.12 GB):   2%|▏         | 1/58 [00:00<00:25,  2.26it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.07 GB):   2%|▏         | 1/58 [00:00<00:25,  2.26it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.07 GB):   3%|▎         | 2/58 [00:00<00:23,  2.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.06 GB):   3%|▎         | 2/58 [00:00<00:23,  2.38it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.06 GB):   5%|▌         | 3/58 [00:01<00:20,  2.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.04 GB):   5%|▌         | 3/58 [00:01<00:20,  2.62it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.04 GB):   7%|▋         | 4/58 [00:01<00:19,  2.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.03 GB):   7%|▋         | 4/58 [00:01<00:19,  2.77it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.03 GB):   9%|▊         | 5/58 [00:01<00:16,  3.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.02 GB):   9%|▊         | 5/58 [00:01<00:16,  3.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.02 GB):  10%|█         | 6/58 [00:01<00:14,  3.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.03 GB):  10%|█         | 6/58 [00:01<00:14,  3.63it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.03 GB):  12%|█▏        | 7/58 [00:02<00:12,  4.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.02 GB):  12%|█▏        | 7/58 [00:02<00:12,  4.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.02 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.02 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.57it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.02 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.02 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.02 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.02 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.02 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.02 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.02 GB):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.02 GB):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.02 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.01 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.01 GB):  22%|██▏       | 13/58 [00:03<00:06,  7.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.01 GB):  26%|██▌       | 15/58 [00:03<00:04,  8.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.01 GB):  26%|██▌       | 15/58 [00:03<00:04,  8.71it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=42.01 GB):  26%|██▌       | 15/58 [00:03<00:04,  8.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.01 GB):  29%|██▉       | 17/58 [00:03<00:04, 10.18it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.00 GB):  29%|██▉       | 17/58 [00:03<00:04, 10.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.00 GB):  29%|██▉       | 17/58 [00:03<00:04, 10.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.00 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.00 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=42.00 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]Capturing num tokens (num_tokens=960 avail_mem=41.98 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s] Capturing num tokens (num_tokens=960 avail_mem=41.98 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.21it/s]Capturing num tokens (num_tokens=896 avail_mem=41.98 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.21it/s]Capturing num tokens (num_tokens=832 avail_mem=41.98 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.21it/s]Capturing num tokens (num_tokens=768 avail_mem=41.97 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.21it/s]

    Capturing num tokens (num_tokens=768 avail_mem=41.97 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Capturing num tokens (num_tokens=704 avail_mem=41.97 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Capturing num tokens (num_tokens=640 avail_mem=41.96 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Capturing num tokens (num_tokens=576 avail_mem=41.96 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Capturing num tokens (num_tokens=576 avail_mem=41.96 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.71it/s]Capturing num tokens (num_tokens=512 avail_mem=41.96 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.71it/s]Capturing num tokens (num_tokens=480 avail_mem=41.95 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.71it/s]Capturing num tokens (num_tokens=448 avail_mem=41.95 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.71it/s]

    Capturing num tokens (num_tokens=416 avail_mem=41.95 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.71it/s]Capturing num tokens (num_tokens=416 avail_mem=41.95 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.80it/s]Capturing num tokens (num_tokens=384 avail_mem=41.94 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.80it/s]Capturing num tokens (num_tokens=352 avail_mem=41.94 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.80it/s]Capturing num tokens (num_tokens=320 avail_mem=41.93 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.80it/s]Capturing num tokens (num_tokens=288 avail_mem=41.93 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.80it/s]Capturing num tokens (num_tokens=288 avail_mem=41.93 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.57it/s]Capturing num tokens (num_tokens=256 avail_mem=41.92 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.57it/s]Capturing num tokens (num_tokens=240 avail_mem=41.92 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.57it/s]

    Capturing num tokens (num_tokens=224 avail_mem=41.92 GB):  62%|██████▏   | 36/58 [00:04<00:00, 26.57it/s]Capturing num tokens (num_tokens=208 avail_mem=41.91 GB):  62%|██████▏   | 36/58 [00:04<00:00, 26.57it/s]Capturing num tokens (num_tokens=208 avail_mem=41.91 GB):  69%|██████▉   | 40/58 [00:04<00:00, 29.06it/s]Capturing num tokens (num_tokens=192 avail_mem=41.91 GB):  69%|██████▉   | 40/58 [00:04<00:00, 29.06it/s]Capturing num tokens (num_tokens=176 avail_mem=41.90 GB):  69%|██████▉   | 40/58 [00:04<00:00, 29.06it/s]Capturing num tokens (num_tokens=160 avail_mem=41.86 GB):  69%|██████▉   | 40/58 [00:04<00:00, 29.06it/s]Capturing num tokens (num_tokens=160 avail_mem=41.86 GB):  74%|███████▍  | 43/58 [00:04<00:00, 26.82it/s]Capturing num tokens (num_tokens=144 avail_mem=41.86 GB):  74%|███████▍  | 43/58 [00:04<00:00, 26.82it/s]

    Capturing num tokens (num_tokens=128 avail_mem=41.85 GB):  74%|███████▍  | 43/58 [00:04<00:00, 26.82it/s]Capturing num tokens (num_tokens=112 avail_mem=41.86 GB):  74%|███████▍  | 43/58 [00:04<00:00, 26.82it/s]Capturing num tokens (num_tokens=96 avail_mem=41.85 GB):  74%|███████▍  | 43/58 [00:04<00:00, 26.82it/s] Capturing num tokens (num_tokens=96 avail_mem=41.85 GB):  81%|████████  | 47/58 [00:04<00:00, 29.45it/s]Capturing num tokens (num_tokens=80 avail_mem=41.85 GB):  81%|████████  | 47/58 [00:04<00:00, 29.45it/s]Capturing num tokens (num_tokens=64 avail_mem=41.85 GB):  81%|████████  | 47/58 [00:04<00:00, 29.45it/s]Capturing num tokens (num_tokens=48 avail_mem=41.84 GB):  81%|████████  | 47/58 [00:04<00:00, 29.45it/s]Capturing num tokens (num_tokens=32 avail_mem=41.84 GB):  81%|████████  | 47/58 [00:04<00:00, 29.45it/s]Capturing num tokens (num_tokens=32 avail_mem=41.84 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.42it/s]Capturing num tokens (num_tokens=28 avail_mem=41.83 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.42it/s]

    Capturing num tokens (num_tokens=24 avail_mem=41.83 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.42it/s]Capturing num tokens (num_tokens=20 avail_mem=41.83 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.42it/s]Capturing num tokens (num_tokens=16 avail_mem=41.82 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.42it/s]Capturing num tokens (num_tokens=16 avail_mem=41.82 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.67it/s]Capturing num tokens (num_tokens=12 avail_mem=41.82 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.67it/s]Capturing num tokens (num_tokens=8 avail_mem=41.81 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.67it/s] Capturing num tokens (num_tokens=4 avail_mem=41.81 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.67it/s]Capturing num tokens (num_tokens=4 avail_mem=41.81 GB): 100%|██████████| 58/58 [00:04<00:00, 12.52it/s]


    [2026-05-05 22:07:36] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-05 22:07:38] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


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


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Sources:<br>- get_current_date function<br>- get_current_weather function</strong>


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


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '2893ba9b306247bc85d027a9634ff586', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.13287725765258074, 'response_sent_to_client_ts': 1778018867.977646}}</strong>



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


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '1edcc448c3d047daa058bdc4962bbcaf', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 49, 'cached_tokens_details': {'device': 49, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1287818467244506, 'response_sent_to_client_ts': 1778018868.1184804}}</strong>


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


<strong style='color: #00008B;'>[{'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '72c84927c5d242ca82dea458e11b061c', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.07048535533249378, 'response_sent_to_client_ts': 1778018868.2128453}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '9100548d2cc2467e9d04012f00a38314', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.07042238116264343, 'response_sent_to_client_ts': 1778018868.2128558}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '8deb153bf0744fd9ab09dddabf971331', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.07037872262299061, 'response_sent_to_client_ts': 1778018868.2128594}}]</strong>


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


<strong style='color: #00008B;'>{'text': 'France', 'output_ids': [50100, 128009], 'meta_info': {'id': '44083b741b264113913e3b295502678c', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 2, 'cached_tokens': 31, 'cached_tokens_details': {'device': 31, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.02764022722840309, 'response_sent_to_client_ts': 1778018868.247976}}</strong>


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


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': 'ab1bccfbc067473984b9ff6ea2986471', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11719468329101801, 'response_sent_to_client_ts': 1778018869.6028476}}</strong>



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


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': 'f73b21c546d44a4cbaf5bdb1dd379301', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.03670417424291372, 'response_sent_to_client_ts': 1778018869.6483526}}</strong>



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

    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.53it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.39it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.38it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.90it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.69it/s]


    2026-05-05 22:08:06,836 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 22:08:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:00,  5.27s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:00,  5.27s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:48,  1.12it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:48,  1.12it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:34,  1.56it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:34,  1.56it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:28,  1.85it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:28,  1.85it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:23,  2.13it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:23,  2.13it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:20,  2.41it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:20,  2.41it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:18,  2.72it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:18,  2.72it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:15,  3.07it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:15,  3.07it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:13,  3.39it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:13,  3.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:11,  3.85it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:11,  3.85it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:08<00:11,  3.85it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:07,  5.67it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:07,  5.67it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:08<00:07,  5.67it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:05,  7.61it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:05,  7.61it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:08<00:05,  7.61it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:08<00:04,  9.78it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:08<00:04,  9.78it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:08<00:04,  9.78it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:08<00:04,  9.78it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:08<00:02, 13.62it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:08<00:02, 13.62it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:08<00:02, 13.62it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:08<00:02, 13.62it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:08<00:02, 13.62it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:08<00:02, 13.62it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:08<00:01, 20.83it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:08<00:01, 20.83it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:08<00:01, 20.83it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:08<00:01, 20.83it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:08<00:01, 20.83it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:08<00:01, 20.83it/s]

    Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:08<00:01, 20.83it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:08<00:00, 38.50it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:08<00:00, 38.50it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:08<00:00, 38.50it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:08<00:00, 38.50it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:08<00:00, 38.50it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:08<00:00, 38.50it/s]

    Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:08<00:00, 38.50it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:08<00:00, 38.50it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:08<00:00, 45.94it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:08<00:00, 45.94it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:08<00:00, 45.94it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:08<00:00, 45.94it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:08<00:00, 45.94it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:08<00:00, 45.94it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:08<00:00, 45.94it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:08<00:00, 45.94it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:09<00:00, 45.94it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:09<00:00, 45.94it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:09<00:00, 55.76it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:09<00:00, 55.76it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:09<00:00, 55.76it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:09<00:00, 55.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=59.75 GB):   2%|▏         | 1/58 [00:00<00:19,  2.93it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.72 GB):   2%|▏         | 1/58 [00:00<00:19,  2.93it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=59.72 GB):   3%|▎         | 2/58 [00:00<00:17,  3.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.72 GB):   3%|▎         | 2/58 [00:00<00:17,  3.12it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=59.72 GB):   5%|▌         | 3/58 [00:00<00:16,  3.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.72 GB):   5%|▌         | 3/58 [00:00<00:16,  3.37it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.72 GB):   7%|▋         | 4/58 [00:01<00:15,  3.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.72 GB):   7%|▋         | 4/58 [00:01<00:15,  3.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.72 GB):   9%|▊         | 5/58 [00:01<00:13,  3.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.72 GB):   9%|▊         | 5/58 [00:01<00:13,  3.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.72 GB):  10%|█         | 6/58 [00:01<00:12,  4.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.72 GB):  10%|█         | 6/58 [00:01<00:12,  4.15it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=59.72 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.70 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.70 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.21 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.80it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=59.21 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.04 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.04 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.04 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.04 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.04 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.04 GB):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.04 GB):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=59.04 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.04 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.03 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.03 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.03 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.69it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=59.03 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.03 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.03 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.02 GB):  29%|██▉       | 17/58 [00:03<00:04, 10.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.02 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.94it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.02 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.94it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=59.02 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.94it/s]Capturing num tokens (num_tokens=960 avail_mem=59.00 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.94it/s] Capturing num tokens (num_tokens=960 avail_mem=59.00 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.21it/s]Capturing num tokens (num_tokens=896 avail_mem=59.00 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.21it/s]Capturing num tokens (num_tokens=832 avail_mem=59.00 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.21it/s]Capturing num tokens (num_tokens=768 avail_mem=59.00 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.21it/s]Capturing num tokens (num_tokens=768 avail_mem=59.00 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Capturing num tokens (num_tokens=704 avail_mem=58.99 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.99 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Capturing num tokens (num_tokens=576 avail_mem=58.98 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Capturing num tokens (num_tokens=576 avail_mem=58.98 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.80it/s]Capturing num tokens (num_tokens=512 avail_mem=58.98 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.80it/s]Capturing num tokens (num_tokens=480 avail_mem=58.97 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.80it/s]Capturing num tokens (num_tokens=448 avail_mem=58.97 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.80it/s]Capturing num tokens (num_tokens=416 avail_mem=58.97 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.80it/s]Capturing num tokens (num_tokens=416 avail_mem=58.97 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.93it/s]Capturing num tokens (num_tokens=384 avail_mem=58.97 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.93it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.96 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.93it/s]Capturing num tokens (num_tokens=320 avail_mem=58.96 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.93it/s]Capturing num tokens (num_tokens=288 avail_mem=58.95 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.93it/s]Capturing num tokens (num_tokens=288 avail_mem=58.95 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.75it/s]Capturing num tokens (num_tokens=256 avail_mem=58.95 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.75it/s]Capturing num tokens (num_tokens=240 avail_mem=58.95 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.75it/s]Capturing num tokens (num_tokens=224 avail_mem=58.94 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.75it/s]Capturing num tokens (num_tokens=208 avail_mem=58.94 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.75it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.94 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=192 avail_mem=58.93 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=176 avail_mem=58.93 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=160 avail_mem=58.92 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=144 avail_mem=58.92 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=144 avail_mem=58.92 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.20it/s]Capturing num tokens (num_tokens=128 avail_mem=58.91 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.20it/s]Capturing num tokens (num_tokens=112 avail_mem=58.92 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.20it/s]Capturing num tokens (num_tokens=96 avail_mem=58.91 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.20it/s] Capturing num tokens (num_tokens=80 avail_mem=58.91 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.20it/s]

    Capturing num tokens (num_tokens=80 avail_mem=58.91 GB):  83%|████████▎ | 48/58 [00:03<00:00, 32.70it/s]Capturing num tokens (num_tokens=64 avail_mem=58.91 GB):  83%|████████▎ | 48/58 [00:03<00:00, 32.70it/s]Capturing num tokens (num_tokens=48 avail_mem=58.90 GB):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Capturing num tokens (num_tokens=32 avail_mem=58.90 GB):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Capturing num tokens (num_tokens=28 avail_mem=58.89 GB):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Capturing num tokens (num_tokens=28 avail_mem=58.89 GB):  90%|████████▉ | 52/58 [00:04<00:00, 33.78it/s]Capturing num tokens (num_tokens=24 avail_mem=57.83 GB):  90%|████████▉ | 52/58 [00:04<00:00, 33.78it/s]Capturing num tokens (num_tokens=20 avail_mem=57.83 GB):  90%|████████▉ | 52/58 [00:04<00:00, 33.78it/s]Capturing num tokens (num_tokens=16 avail_mem=57.83 GB):  90%|████████▉ | 52/58 [00:04<00:00, 33.78it/s]Capturing num tokens (num_tokens=12 avail_mem=57.82 GB):  90%|████████▉ | 52/58 [00:04<00:00, 33.78it/s]

    Capturing num tokens (num_tokens=12 avail_mem=57.82 GB):  97%|█████████▋| 56/58 [00:04<00:00, 34.44it/s]Capturing num tokens (num_tokens=8 avail_mem=57.82 GB):  97%|█████████▋| 56/58 [00:04<00:00, 34.44it/s] Capturing num tokens (num_tokens=4 avail_mem=55.80 GB):  97%|█████████▋| 56/58 [00:04<00:00, 34.44it/s]Capturing num tokens (num_tokens=4 avail_mem=55.80 GB): 100%|██████████| 58/58 [00:04<00:00, 13.59it/s]


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



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Italy.<br>Generated text: London is the capital of England</strong>


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
