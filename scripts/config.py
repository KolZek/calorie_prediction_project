from dataclasses import dataclass


@dataclass
class Config:
    DATA_ROOT: str = "/content/data"
    CHECKPOINT_DIR: str = "/content/checkpoints"
    MODEL_SAVE_PATH: str = "/content/checkpoints/best_model.pt"

    SEED: int = 42

    IMG_SIZE: int = 224
    MAX_LEN: int = 128
    VAL_FRAC: float = 0.15
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2

    IMAGE_MODEL: str = "efficientnet_b0"
    TEXT_MODEL: str = "distilbert-base-uncased"
    HIDDEN_DIM: int = 512
    DROPOUT: float = 0.2

    EPOCHS: int = 15
    IMAGE_LR: float = 1e-5
    TEXT_LR: float = 2e-5
    CLASSIFIER_LR: float = 1e-3

    DEVICE: str = "cuda"
