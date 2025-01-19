import inspect
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from src.utils.predicter import PosePredictor
from ultralytics.engine.results import Results
from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.utils import (
    ARGV,
    ASSETS,
    LOGGER,
    callbacks,
    emojis,
)


class PosePredict():
    def __init__(self, model, depth=False):
        self.callbacks = callbacks.get_default_callbacks()
        self.overrides = model.overrides
        self.task = model.task
        self.predictor = None
        self.model = model.model
        self.depth = depth

    def _smart_load(self):
        try:
            return PosePredictor
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(
                    f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list,
                      tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> List[Results]:

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        # highest priority args on the right
        args = {**self.overrides, **custom, **kwargs}
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = (predictor or self._smart_load())(
                overrides=args, _callbacks=self.callbacks, depth=self.depth)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)
