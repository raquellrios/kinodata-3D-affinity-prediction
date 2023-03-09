import inspect

from typing import Any, Dict, Hashable, Iterable, Optional, Union, TypeVar
import traceback
from collections import defaultdict
from pathlib import Path
from warnings import warn

import torch
import yaml

T = TypeVar("T")


class Config(dict):
    def __getattribute__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        return super().__getattribute__(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        self[__name] == __value

    def intersect(self, other: Union[dict, "Config"]) -> "Config":
        return Config({k: v for k, v in self.items() if k in other})

    def subset(self, keys: Iterable[Hashable]) -> "Config":
        return Config({key: self[key] for key in keys if key in self})

    def update(
        self, other: Union[dict, "Config"], allow_duplicates: bool = True
    ) -> "Config":
        if not allow_duplicates:
            intersection = self.intersect(other)
            if len(intersection) > 0:
                raise ValueError(f"Duplicate keys detected: {intersection.keys()}")
        return Config(self | other)

    def init(self, obj: type[T], *args, **kwargs) -> T:
        obj_signature = inspect.signature(obj)
        sub_config = self.subset(obj_signature.parameters.keys())
        sub_config.update(kwargs, allow_duplicates=False)
        bound_arguments = obj_signature.bind(*args, **sub_config)
        return obj(*bound_arguments.args, **bound_arguments.kwargs)

    def __repr__(self) -> str:
        inner = ", ".join([f"{key}={value}" for key, value in self.items()])
        return f"{self.__class__.__name__}({inner})"


configs: Dict[str, Config] = dict()


def register(config_name: str, config: Optional[Config] = None, **kwargs: Any):
    if config is None:
        config = Config(**kwargs)
    configs[config_name] = config.update(kwargs)


def extend(config_name: str, **kwargs: Any):
    configs[config_name] |= kwargs


def load_from_file(fp: Path) -> Config:
    try:
        file_stream = fp.read_text()
        config = yaml.full_load(file_stream)
    except:
        print(f"Not a valid config yaml file: {fp}")
        traceback.print_exc()
        exit(1)
    return config


def overwrite_from_file(config: Config, fp: Union[str, Path], verbose: bool = True):
    fp = Path(fp)
    if not fp.exists():
        warn(f"Config file does not exist: {fp}")
        return config
    file_config = load_from_file(fp)
    if verbose:
        print(f"Reading additional config from {fp}:")
        print("\n".join([f"{key}: {value}" for key, value in file_config.items()]))
    return config.update(file_config)


def get(*config_names: str) -> Config:
    key_counts = defaultdict(int)
    for name in config_names:
        assert name in configs, f'Unknown config: "{name}"'
        for key in configs[name]:
            key_counts[key] += 1

    duplicate_keys = [key for key in key_counts if key_counts[key] > 1]
    if len(duplicate_keys) > 0:
        raise ValueError(f"Duplicate config keys: {duplicate_keys}")

    config = Config()
    for name in config_names:
        config = config.update(configs[name])
    return config


register("meta", model_type="egin")

register(
    "data",
    interaction_radius=5.0,
    node_types=["ligand", "pocket"],
    edge_types=[
        ("ligand", "interacts", "ligand"),
        ("ligand", "interacts", "pocket"),
        ("pocket", "interacts", "ligand"),
    ],
    seed=420,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    use_bonds=True,
    add_artificial_decoys=False,
    cold_split=False,
)

register(
    "egnn",
    num_mp_layers=4,
    hidden_channels=64,
    act="silu",
    final_act="softplus",
    mp_type="rbf",
    mp_reduce="sum",
    rbf_size=64,
    readout_aggregation_type="sum",
)

register(
    "egin",
    hidden_channels=128,
    num_layers=4,
    d_cut=5.0,
    edge_dim=None,
    readout_aggregation_type="sum",
    act="relu",
    final_act="softplus",
)

register(
    "training",
    lr=3e-4,
    weight_decay=1e-5,
    batch_size=32,
    accumulate_grad_batches=2,
    epochs=100,
    num_workers=32,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    loss_type="mse",
    lr_factor=0.7,
    lr_patience=5,
    min_lr=1e-7,
    perturb_ligand_positions=None,
    perturb_pocket_positions=None,
    add_docking_scores=False,
)
