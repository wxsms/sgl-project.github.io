# SGLang Frontend Language

SGLang frontend language can be used to define simple and easy prompts in a convenient, structured way.

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint
from sglang.lang.api import set_default_backend
from sglang.srt.utils import load_image
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import print_highlight, terminate_process, wait_for_server

server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-11 00:51:44] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.56it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.43it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.40it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.42it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.43it/s]


    2026-05-11 00:51:50,770 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 00:51:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:54,  5.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:54,  5.17s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:06,  2.26s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:06,  2.26s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.15it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.82it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.82it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.56it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.56it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.39it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.39it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.39it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.06it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.06it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.06it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.58it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.58it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.58it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.13it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.13it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.13it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.00it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.00it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.00it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.00it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.52it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.52it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.52it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.52it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.52it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.52it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.74it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.74it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.74it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.74it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.74it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.74it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.74it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:07<00:00, 34.87it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:07<00:00, 45.12it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 52.57it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 52.57it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 52.57it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:07<00:00, 52.57it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:07<00:00, 52.57it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:07<00:00, 52.57it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:07<00:00, 52.57it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:07<00:00, 52.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.49 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.49 GB):   2%|▏         | 1/58 [00:00<00:16,  3.38it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.46 GB):   2%|▏         | 1/58 [00:00<00:16,  3.38it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.46 GB):   3%|▎         | 2/58 [00:00<00:15,  3.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.45 GB):   3%|▎         | 2/58 [00:00<00:15,  3.54it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.45 GB):   5%|▌         | 3/58 [00:00<00:14,  3.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.45 GB):   5%|▌         | 3/58 [00:00<00:14,  3.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.45 GB):   7%|▋         | 4/58 [00:01<00:13,  4.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.45 GB):   7%|▋         | 4/58 [00:01<00:13,  4.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.45 GB):   9%|▊         | 5/58 [00:01<00:12,  4.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.45 GB):   9%|▊         | 5/58 [00:01<00:12,  4.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.45 GB):  10%|█         | 6/58 [00:01<00:11,  4.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.45 GB):  10%|█         | 6/58 [00:01<00:11,  4.66it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.45 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.45 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.45 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.45 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.49it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=45.45 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.45 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.45 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.45 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.19it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=45.45 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.45 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.45 GB):  21%|██        | 12/58 [00:02<00:06,  7.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.44 GB):  21%|██        | 12/58 [00:02<00:06,  7.35it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=45.44 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.44 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.44 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.44 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.44 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.75it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.44 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.44 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.44 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.43 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.82it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=45.43 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.43 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.42 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=960 avail_mem=45.41 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.95it/s] Capturing num tokens (num_tokens=960 avail_mem=45.41 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.88it/s]Capturing num tokens (num_tokens=896 avail_mem=45.41 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.88it/s]

    Capturing num tokens (num_tokens=832 avail_mem=45.41 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.88it/s]Capturing num tokens (num_tokens=832 avail_mem=45.41 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.09it/s]Capturing num tokens (num_tokens=768 avail_mem=45.40 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.09it/s]Capturing num tokens (num_tokens=704 avail_mem=45.40 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.09it/s]Capturing num tokens (num_tokens=640 avail_mem=45.40 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.09it/s]Capturing num tokens (num_tokens=640 avail_mem=45.40 GB):  47%|████▋     | 27/58 [00:03<00:01, 17.58it/s]Capturing num tokens (num_tokens=576 avail_mem=45.39 GB):  47%|████▋     | 27/58 [00:03<00:01, 17.58it/s]

    Capturing num tokens (num_tokens=512 avail_mem=45.39 GB):  47%|████▋     | 27/58 [00:03<00:01, 17.58it/s]Capturing num tokens (num_tokens=480 avail_mem=45.39 GB):  47%|████▋     | 27/58 [00:03<00:01, 17.58it/s]Capturing num tokens (num_tokens=448 avail_mem=45.38 GB):  47%|████▋     | 27/58 [00:03<00:01, 17.58it/s]Capturing num tokens (num_tokens=448 avail_mem=45.38 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.84it/s]Capturing num tokens (num_tokens=416 avail_mem=45.38 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.84it/s]Capturing num tokens (num_tokens=384 avail_mem=45.38 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.84it/s]Capturing num tokens (num_tokens=352 avail_mem=45.37 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.84it/s]

    Capturing num tokens (num_tokens=352 avail_mem=45.37 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.19it/s]Capturing num tokens (num_tokens=320 avail_mem=45.37 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.19it/s]Capturing num tokens (num_tokens=288 avail_mem=45.37 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.19it/s]Capturing num tokens (num_tokens=256 avail_mem=45.13 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.19it/s]Capturing num tokens (num_tokens=256 avail_mem=45.13 GB):  64%|██████▍   | 37/58 [00:03<00:01, 19.87it/s]Capturing num tokens (num_tokens=240 avail_mem=44.26 GB):  64%|██████▍   | 37/58 [00:03<00:01, 19.87it/s]

    Capturing num tokens (num_tokens=224 avail_mem=44.26 GB):  64%|██████▍   | 37/58 [00:03<00:01, 19.87it/s]Capturing num tokens (num_tokens=208 avail_mem=44.25 GB):  64%|██████▍   | 37/58 [00:03<00:01, 19.87it/s]Capturing num tokens (num_tokens=208 avail_mem=44.25 GB):  69%|██████▉   | 40/58 [00:03<00:01, 17.54it/s]Capturing num tokens (num_tokens=192 avail_mem=44.25 GB):  69%|██████▉   | 40/58 [00:03<00:01, 17.54it/s]Capturing num tokens (num_tokens=176 avail_mem=44.25 GB):  69%|██████▉   | 40/58 [00:03<00:01, 17.54it/s]Capturing num tokens (num_tokens=160 avail_mem=44.24 GB):  69%|██████▉   | 40/58 [00:03<00:01, 17.54it/s]Capturing num tokens (num_tokens=144 avail_mem=44.24 GB):  69%|██████▉   | 40/58 [00:03<00:01, 17.54it/s]

    Capturing num tokens (num_tokens=144 avail_mem=44.24 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.72it/s]Capturing num tokens (num_tokens=128 avail_mem=44.24 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.72it/s]Capturing num tokens (num_tokens=112 avail_mem=44.24 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.72it/s]Capturing num tokens (num_tokens=96 avail_mem=44.23 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.72it/s] Capturing num tokens (num_tokens=80 avail_mem=44.23 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.72it/s]Capturing num tokens (num_tokens=80 avail_mem=44.23 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.41it/s]Capturing num tokens (num_tokens=64 avail_mem=44.23 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.41it/s]Capturing num tokens (num_tokens=48 avail_mem=44.22 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.41it/s]Capturing num tokens (num_tokens=32 avail_mem=44.22 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.41it/s]Capturing num tokens (num_tokens=28 avail_mem=44.22 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.41it/s]

    Capturing num tokens (num_tokens=28 avail_mem=44.22 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.82it/s]Capturing num tokens (num_tokens=24 avail_mem=44.21 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.82it/s]Capturing num tokens (num_tokens=20 avail_mem=44.21 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.82it/s]Capturing num tokens (num_tokens=16 avail_mem=44.21 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.82it/s]Capturing num tokens (num_tokens=12 avail_mem=44.20 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.82it/s]Capturing num tokens (num_tokens=12 avail_mem=44.20 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.64it/s]Capturing num tokens (num_tokens=8 avail_mem=44.20 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.64it/s] Capturing num tokens (num_tokens=4 avail_mem=44.20 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.64it/s]Capturing num tokens (num_tokens=4 avail_mem=44.20 GB): 100%|██████████| 58/58 [00:04<00:00, 13.24it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32207


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-11 00:52:16] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


## Basic Usage

The most simple way of using SGLang frontend language is a simple question answer dialog between a user and an assistant.


```python
@function
def basic_qa(s, question):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))
```


```python
state = basic_qa("List 3 countries and their capitals.")
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>Sure! Here are three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Italy** - Rome<br>3. **Japan** - Tokyo</strong>


## Multi-turn Dialog

SGLang frontend language can also be used to define multi-turn dialogs.


```python
@function
def multi_turn_qa(s):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user("Please give me a list of 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=512))
    s += user("Please give me another list of 3 countries and their capitals.")
    s += assistant(gen("second_answer", max_tokens=512))
    return s


state = multi_turn_qa()
print_highlight(state["first_answer"])
print_highlight(state["second_answer"])
```


<strong style='color: #00008B;'>Certainly! Here is a list of 3 countries and their respective capitals:<br><br>1. **Germany** - Berlin<br>2. **France** - Paris<br>3. **Italy** - Rome</strong>



<strong style='color: #00008B;'>Of course! Here is another list of 3 countries and their capitals:<br><br>1. **Spain** - Madrid<br>2. **Canada** - Ottawa<br>3. **Australia** - Canberra</strong>


## Control flow

You may use any Python code within the function to define more complex control flows.


```python
@function
def tool_use(s, question):
    s += assistant(
        "To answer this question: "
        + question
        + ". I need to use a "
        + gen("tool", choices=["calculator", "search engine"])
        + ". "
    )

    if s["tool"] == "calculator":
        s += assistant("The math expression is: " + gen("expression"))
    elif s["tool"] == "search engine":
        s += assistant("The key word to search is: " + gen("word"))


state = tool_use("What is 2 * 2?")
print_highlight(state["tool"])
print_highlight(state["expression"])
```


<strong style='color: #00008B;'>calculator</strong>



<strong style='color: #00008B;'>2 * 2.<br><br>Let's calculate it directly without the need for a physical calculator:<br><br>2 * 2 = 4<br><br>So, the answer is 4.</strong>


## Parallelism

Use `fork` to launch parallel prompts. Because `sgl.gen` is non-blocking, the for loop below issues two generation calls in parallel.


```python
@function
def tip_suggestion(s):
    s += assistant(
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += assistant(
            f"Now, expand tip {i+1} into a paragraph:\n"
            + gen("detailed_tip", max_tokens=256, stop="\n\n")
        )

    s += assistant("Tip 1:" + forks[0]["detailed_tip"] + "\n")
    s += assistant("Tip 2:" + forks[1]["detailed_tip"] + "\n")
    s += assistant(
        "To summarize the above two tips, I can say:\n" + gen("summary", max_tokens=512)
    )


state = tip_suggestion()
print_highlight(state["summary"])
```


<strong style='color: #00008B;'>1. **Balanced Diet**: <br>   - **Key Components**: Focus on a variety of nutritious foods including fruits, vegetables, whole grains, lean proteins, and healthy fats.<br>   - **Limitations**: Reduce processed foods, sugary snacks, and high-fat foods.<br>   - **Importance**: Essential for maintaining a healthy weight, reducing the risk of chronic diseases, and ensuring your body receives necessary nutrients.<br><br>2. **Regular Exercise**:<br>   - **Aim**: Aim for at least 150 to 300 minutes of moderate aerobic activity or 75 to 150 minutes of vigorous activity each week, plus muscle-strengthening exercises on two or more days.<br>   - **Benefits**: Strengthens the heart, improves mood, enhances sleep quality, boosts energy, and reduces the risk of chronic diseases.<br>   - **Conclusion**: Regular physical activity significantly improves overall health and quality of life.<br><br>By following these two tips, you can maintain good health and lead a more vibrant life.</strong>


## Constrained Decoding

Use `regex` to specify a regular expression as a decoding constraint. This is only supported for local models.


```python
@function
def regular_expression_gen(s):
    s += user("What is the IP address of the Google DNS servers?")
    s += assistant(
        gen(
            "answer",
            temperature=0,
            regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
        )
    )


state = regular_expression_gen()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>208.67.222.222</strong>


Use `regex` to define a `JSON` decoding schema.


```python
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)


@function
def character_gen(s, name):
    s += user(
        f"{name} is a character in Harry Potter. Please fill in the following information about this character."
    )
    s += assistant(gen("json_output", max_tokens=256, regex=character_regex))


state = character_gen("Harry Potter")
print_highlight(state["json_output"])
```


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Ash",<br>        "core": "Phoenix feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Tom Marvolo Ridd"<br>}</strong>


## Batching 

Use `run_batch` to run a batch of prompts.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)

for i, state in enumerate(states):
    print_highlight(f"Answer {i+1}: {states[i]['answer']}")
```

      0%|          | 0/3 [00:00<?, ?it/s]

    100%|██████████| 3/3 [00:00<00:00, 28.76it/s]

    100%|██████████| 3/3 [00:00<00:00, 28.51it/s]

    



<strong style='color: #00008B;'>Answer 1: The capital of the United Kingdom is London.</strong>



<strong style='color: #00008B;'>Answer 2: The capital of France is Paris.</strong>



<strong style='color: #00008B;'>Answer 3: The capital of Japan is Tokyo.</strong>


## Streaming 

Use `stream` to stream the output to the user.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


state = text_qa.run(
    question="What is the capital of France?", temperature=0.1, stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)
```

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant


    The

     capital

     of

     France

     is

     Paris

    .

    <|im_end|>


## Complex Prompts

You may use `{system|user|assistant}_{begin|end}` to define complex prompts.


```python
@function
def chat_example(s):
    s += system("You are a helpful assistant.")
    # Same as: s += s.system("You are a helpful assistant.")

    with s.user():
        s += "Question: What is the capital of France?"

    s += assistant_begin()
    s += "Answer: " + gen("answer", max_tokens=100, stop="\n")
    s += assistant_end()


state = chat_example()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'> The capital of France is Paris.</strong>



```python
terminate_process(server_process)
```

## Multi-modal Generation

You may use SGLang frontend language to define multi-modal prompts.
See [here](https://docs.sglang.io/supported_models/text_generation/multimodal_language_models.html) for supported models.


```python
server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-11 00:52:38] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [2026-05-11 00:52:40] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-11 00:52:43] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [2026-05-11 00:52:47] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:02,  1.47it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.46it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.41it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:02<00:00,  1.44it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.87it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.65it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-05-11 00:52:56,964 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 00:52:56] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32925



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-11 00:53:00] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


Ask a question about an image.


```python
@function
def image_qa(s, image_file, question):
    s += user(image(image_file) + question)
    s += assistant(gen("answer", max_tokens=256))


image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
image_bytes, _ = load_image(image_url)
state = image_qa(image_bytes, "What is in the image?")
print_highlight(state["answer"])
```

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:849: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1586.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a person attempting to iron clothing in the back of a yellow SUV in a busy urban environment. The person is standing on a step ladder placed on the back of the vehicle, and the SUV's rear trunk is open to hold the garments being ironed. Yellow taxi cabs can be seen in the background, along with tall buildings and streetlights, indicating a city street scene.</strong>



```python
terminate_process(server_process)
```
