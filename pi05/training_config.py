"""OpenPI JAX configurations for full and LoRA training."""
import dataclasses
from dataclasses import dataclass
from pathlib import Path

from openpi import transforms as ot
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as oc
from openpi.training import optimizer, weight_loaders
from pi05.transforms import YAMInputs, YAMOutputs


@dataclass(frozen=True)
class ABCDataConfig(oc.DataConfigFactory):
    repo_id: str = "yam_abc"

    def create(self, assets_dirs, model_config):
        group = ot.Group(inputs=[YAMInputs()], outputs=[YAMOutputs()])
        mask = ot.make_bool_mask(6, -1, 6, -1)
        group = group.push(inputs=[ot.DeltaActions(mask)], outputs=[ot.AbsoluteActions(mask)])
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config),
                                   data_transforms=group,
                                   model_transforms=oc.ModelTransformFactory()(model_config))


def make_config(mode="lora", horizon=50, assets="pi05/assets", checkpoints="pi05/checkpoints",
                exp_name="yam", batch_size=8, fsdp_devices=1, steps=30000, workers=4,
                resume=False, save_interval=1000):
    if mode not in ("full", "lora"):
        raise ValueError(mode)
    kwargs = dict(pi05=True, action_horizon=horizon)
    if mode == "lora":
        kwargs.update(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    model = Pi0Config(**kwargs)
    config = oc.TrainConfig(
        name=f"pi05_yam_{mode}", exp_name=exp_name, model=model, data=ABCDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=str(Path(assets).resolve()), checkpoint_base_dir=str(Path(checkpoints).resolve()),
        batch_size=batch_size, fsdp_devices=fsdp_devices, num_train_steps=steps, num_workers=workers,
        resume=resume, save_interval=save_interval, wandb_enabled=False,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5,
                                                decay_steps=30000, decay_lr=5e-6),
        ema_decay=None,
        policy_metadata={"robot": "yam", "fps": 30, "action_dim": 14,
                         "action_space": "absolute_joint_targets", "camera_order": ["top", "left", "right"]})
    if mode == "lora":
        config = dataclasses.replace(config, freeze_filter=model.get_freeze_filter())
    return config
