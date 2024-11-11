from pathlib import Path

from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.utils import (
    SETTINGS,
    checks,
)

# Model information extraction
# class CustomModel(nn.Module):
class ModelExt():
    def __init__(self, model, task: str = None, verbose: bool = False):
        self.overrides = {}
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        Resets specific arguments when loading a PyTorch model checkpoint.

        This static method filters the input arguments dictionary to retain only a specific set of keys that are
        considered important for model loading. It's used to ensure that only relevant arguments are preserved
        when loading a model from a checkpoint, discarding any unnecessary or potentially conflicting settings.

        Args:
            args (dict): A dictionary containing various model arguments and settings.

        Returns:
            (dict): A new dictionary containing only the specified include keys from the input arguments.

        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}
    
    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Load the model parameter from the yaml
        """
        cfg_dict = yaml_model_load(cfg)
        self.task = task or guess_model_task(cfg_dict)
        self.cfg = cfg_dict
        self.overrides["model"] = cfg
        self.overrides["task"] = self.task

    def _load(self, weights: str, task=None) -> None:
        """
        Loads a model from a checkpoint file or initializes it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str | None): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolov8n -> yolov8n.pt

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights
