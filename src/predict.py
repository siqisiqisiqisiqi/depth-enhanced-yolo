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
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source
                of the image(s) to make predictions on. Accepts various types including file paths, URLs, PIL
                images, numpy arrays, and torch tensors.
            stream (bool): If True, treats the input source as a continuous stream for predictions.
            predictor (BasePredictor | None): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor.
            **kwargs (Any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # print detection bounding boxes

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(
                f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

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
