from __future__ import annotations
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import WandbConfig, TrainingConfig, Settings


def init_wandb(
    config: "WandbConfig",
    training_config: "TrainingConfig | None" = None,
    settings: "Settings | None" = None,
):
    if not config.enabled:
        return None

    import wandb

    run_config: dict = {}
    if training_config is not None:
        run_config.update(asdict(training_config))
    if settings is not None:
        run_config.update(asdict(settings))

    run = wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.run_name,
        config=run_config,
    )
    return run


def finish_wandb():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
