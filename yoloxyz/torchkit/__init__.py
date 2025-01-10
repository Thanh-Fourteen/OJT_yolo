# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import json
import uuid
import platform
from pathlib import Path
from threading import Lock
from typing import Union

from yolov9.utils.general import LOGGER

RANK = int(os.getenv("RANK", -1))
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)

def get_user_config_dir(sub_dir="Ultralytics"):
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(path.parent):
        LOGGER.warning(
            f"WARNING ⚠️ user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
            "Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path."
        )
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path

def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d

USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())  # Ultralytics settings dir
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"
GIT_DIR = get_git_dir()

class JSONDict(dict):
    """
    A dictionary-like class that provides JSON persistence for its contents.

    This class extends the built-in dictionary to automatically save its contents to a JSON file whenever they are
    modified. It ensures thread-safe operations using a lock.

    Attributes:
        file_path (Path): The path to the JSON file used for persistence.
        lock (threading.Lock): A lock object to ensure thread-safe operations.

    Methods:
        _load: Loads the data from the JSON file into the dictionary.
        _save: Saves the current state of the dictionary to the JSON file.
        __setitem__: Stores a key-value pair and persists it to disk.
        __delitem__: Removes an item and updates the persistent storage.
        update: Updates the dictionary and persists changes.
        clear: Clears all entries and updates the persistent storage.

    Examples:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: Union[str, Path] = "data.json"):
        """Initialize a JSONDict object with a specified file path for JSON persistence."""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        """Load the data from the JSON file into the dictionary."""
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    self.update(json.load(f))
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.file_path}. Starting with an empty dictionary.")
        except Exception as e:
            print(f"Error reading from {self.file_path}: {e}")

    def _save(self):
        """Save the current state of the dictionary to the JSON file."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            print(f"Error writing to {self.file_path}: {e}")

    @staticmethod
    def _json_default(obj):
        """Handle JSON serialization of Path objects."""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        """Store a key-value pair and persist to disk."""
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        """Remove an item and update the persistent storage."""
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        """Return a pretty-printed JSON string representation of the dictionary."""
        return f'JSONDict("{self.file_path}"):\n{json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)}'

    def update(self, *args, **kwargs):
        """Update the dictionary and persist changes."""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """Clear all entries and update the persistent storage."""
        with self.lock:
            super().clear()
            self._save()

class SettingsManager(JSONDict):
    """
    SettingsManager class for managing and persisting Ultralytics settings.

    This class extends JSONDict to provide JSON persistence for settings, ensuring thread-safe operations and default
    values. It validates settings on initialization and provides methods to update or reset settings.

    Attributes:
        file (Path): The path to the JSON file used for persistence.
        version (str): The version of the settings schema.
        defaults (Dict): A dictionary containing default settings.
        help_msg (str): A help message for users on how to view and update settings.

    Methods:
        _validate_settings: Validates the current settings and resets if necessary.
        update: Updates settings, validating keys and types.
        reset: Resets the settings to default and saves them.

    Examples:
        Initialize and update settings:
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="/new/runs/dir")
        >>> print(settings["runs_dir"])
        /new/runs/dir
    """

    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        """Initializes the SettingsManager with default settings and loads user settings."""
        import hashlib

        from backbones.yolov9.utils.torch_utils import torch_distributed_zero_first

        root = GIT_DIR or Path()
        datasets_root = (root.parent if GIT_DIR and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,  # Settings schema version
            "datasets_dir": str(datasets_root / "datasets"),  # Datasets directory
            "weights_dir": str(root / "weights"),  # Model weights directory
            "runs_dir": str(root / "runs"),  # Experiment runs directory
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # SHA-256 anonymized UUID hash
            "sync": True,  # Enable synchronization
            "api_key": "",  # Ultralytics API Key
            "openai_api_key": "",  # OpenAI API Key
            "clearml": True,  # ClearML integration
            "comet": True,  # Comet integration
            "dvc": True,  # DVC integration
            "hub": True,  # Ultralytics HUB integration
            "mlflow": True,  # MLflow integration
            "neptune": True,  # Neptune integration
            "raytune": True,  # Ray Tune integration
            "tensorboard": True,  # TensorBoard logging
            "wandb": False,  # Weights & Biases logging
            "vscode_msg": True,  # VSCode messaging
        }

        self.help_msg = (
            f"\nView Ultralytics Settings with 'yolo settings' or at '{self.file}'"
            "\nUpdate Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
            "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
        )

        with torch_distributed_zero_first(RANK):
            super().__init__(self.file)

            if not self.file.exists() or not self:  # Check if file doesn't exist or is empty
                LOGGER.info(f"Creating new Ultralytics Settings v{version} file ✅ {self.help_msg}")
                self.reset()

            self._validate_settings()

    def _validate_settings(self):
        """Validate the current settings and reset if necessary."""
        correct_keys = set(self.keys()) == set(self.defaults.keys())
        correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
        correct_version = self.get("settings_version", "") == self.version

        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                "WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem "
                f"with your settings or a recent ultralytics package update. {self.help_msg}"
            )
            self.reset()

        if self.get("datasets_dir") == self.get("runs_dir"):
            LOGGER.warning(
                f"WARNING ⚠️ Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                f"Please change one to avoid possible issues during training. {self.help_msg}"
            )

    def update(self, *args, **kwargs):
        """Updates settings, validating keys and types."""
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
            t = type(self.defaults[k])
            if not isinstance(v, t):
                raise TypeError(f"Ultralytics setting '{k}' must be of type '{t}', not '{type(v)}'. {self.help_msg}")
        super().update(*args, **kwargs)

    def reset(self):
        """Resets the settings to default and saves them."""
        self.clear()
        self.update(self.defaults)
