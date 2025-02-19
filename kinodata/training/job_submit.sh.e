Traceback (most recent call last):
  File "/lila/data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training/train.py", line 2, in <module>
    import pytorch_lightning as pl
  File "/home/lopezrir/miniconda3/envs/kinodata_gpu_lilac/lib/python3.10/site-packages/pytorch_lightning/__init__.py", line 34, in <module>
    from lightning_lite.utilities.seed import seed_everything  # noqa: E402
  File "/home/lopezrir/miniconda3/envs/kinodata_gpu_lilac/lib/python3.10/site-packages/lightning_lite/__init__.py", line 23, in <module>
    from lightning_lite.lite import LightningLite  # noqa: E402
  File "/home/lopezrir/miniconda3/envs/kinodata_gpu_lilac/lib/python3.10/site-packages/lightning_lite/lite.py", line 21, in <module>
    import torch
  File "/home/lopezrir/miniconda3/envs/kinodata_gpu_lilac/lib/python3.10/site-packages/torch/__init__.py", line 778, in <module>
    _C._initExtension(manager_path())
  File "/home/lopezrir/miniconda3/envs/kinodata_gpu_lilac/lib/python3.10/site-packages/torch/cuda/__init__.py", line 23, in <module>
    from .streams import ExternalStream, Stream, Event
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 879, in exec_module
  File "<frozen importlib._bootstrap_external>", line 975, in get_code
  File "<frozen importlib._bootstrap_external>", line 1074, in get_data
KeyboardInterrupt
