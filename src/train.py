import torch
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    callbacks,
)
from ultralytics.cfg import TASK2DATA
from src.utils.trainer import Trainer


class PoseTrain():
    def __init__(self, model):
        self.callbacks = callbacks.get_default_callbacks()
        self.overrides = model.overrides
        self.task = model.task
        self.model = model

    def train(self, **kwargs,):
        overrides = self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}
        self.trainer = Trainer(self.model, overrides=args, _callbacks=self.callbacks)
        if torch.cuda.is_available():
            world_size = 1
        self.trainer._setup_train(world_size)
        self.trainer._do_train(world_size=world_size)


