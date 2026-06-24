# ============================================================
# config.py — Central Configuration for BanglaBERT++ 
# Hybrid Multi-Task Framework
# ============================================================
# 
# Research Project: "A Hybrid Multi-Task BanglaBERT Framework 
# for Fake News Detection, Hate Speech Identification,
# Sentiment Analysis, and News Classification in Bangla"
#
# Architecture: BanglaBERT++ → BiLSTM → CNN → Multi-Head 
# Attention → Shared Representation → Task Heads
# ============================================================

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
PRETRAIN_SAVE  = os.path.join(CHECKPOINT_DIR, "banglabert_plus_plus")

# Create directories if they don't exist
for d in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR, PRETRAIN_SAVE]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Base encoder: csebuetnlp/banglabert (recommended for Bangla NLP)
    base_model_name: str = "csebuetnlp/banglabert"

    # After Phase-1 MLM pretraining, the adapted model is saved here
    adapted_model_path: str = PRETRAIN_SAVE

    # BERT encoder output dimension
    hidden_size: int = 768

    # BiLSTM configuration
    lstm_hidden_size: int = 256       # hidden size per direction
    lstm_num_layers: int = 2          # stacked BiLSTM layers
    lstm_dropout: float = 0.3

    # CNN configuration (applied on top of BiLSTM output)
    cnn_out_channels: int = 256       # number of filters
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_dropout: float = 0.3

    # Multi-Head Attention configuration
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Shared representation (bottleneck after attention)
    shared_hidden_size: int = 512
    shared_dropout: float = 0.3

    # Task head dropout
    head_dropout: float = 0.2

    # Maximum token sequence length
    max_seq_length: int = 256

    # Whether to freeze BERT encoder during Task training (Phase 2 warm-up)
    freeze_encoder_epochs: int = 2    # freeze for first N epochs


# ─────────────────────────────────────────────
# TASK CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class TaskConfig:
    # ── Task 1: Hate Speech Detection ──────────
    hate_speech_labels: List[str] = field(
        default_factory=lambda: ["NoHate", "Hate"]
    )
    hate_speech_num_classes: int = 2

    # ── Task 2: Fake News Detection ────────────
    fake_news_labels: List[str] = field(
        default_factory=lambda: ["Real", "Fake"]
    )
    fake_news_num_classes: int = 2

    # ── Task 3: Sentiment Analysis ─────────────
    sentiment_labels: List[str] = field(
        default_factory=lambda: ["Negative", "Neutral", "Positive"]
    )
    sentiment_num_classes: int = 3

    # ── Task 4: News Category Classification ───
    # Will be inferred from dataset; this is a placeholder
    news_category_labels: List[str] = field(
        default_factory=lambda: [
            "politics", "sports", "entertainment",
            "science", "technology", "business", "international"
        ]
    )
    news_category_num_classes: int = 7   # updated dynamically from data

    # Multi-task loss weights (sum to 1.0; tune based on task priority)
    task_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "hate_speech": 0.30,
            "fake_news":   0.30,
            "sentiment":   0.20,
            "news_cat":    0.20,
        }
    )


# ─────────────────────────────────────────────
# PHASE 1 — MLM PRE-TRAINING CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class PretrainConfig:
    # CC-100 Bangla corpus (download separately; place in data/)
    cc100_data_path: str = os.path.join(DATA_DIR, "cc100_bn.txt")

    # MLM masking probability
    mlm_probability: float = 0.15

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 4    # effective batch = 64
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_steps: int = 100_000             # set -1 to use epochs instead
    save_steps: int = 5_000
    logging_steps: int = 500
    fp16: bool = True                    # mixed precision (requires CUDA)
    dataloader_num_workers: int = 4
    max_seq_length: int = 256


# ─────────────────────────────────────────────
# PHASE 2 — MULTI-TASK FINE-TUNING CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Training schedule
    num_train_epochs: int = 15
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2     # effective batch = 32
    max_grad_norm: float = 1.0               # gradient clipping

    # Optimizer
    learning_rate: float = 2e-5
    bert_learning_rate: float = 1e-5         # lower LR for BERT layers
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8

    # Scheduler: linear warmup then cosine decay
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # Mixed precision & hardware
    fp16: bool = True
    use_cuda: bool = torch.cuda.is_available()
    num_gpu: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu: bool = torch.cuda.device_count() > 1
    dataloader_num_workers: int = 4
    seed: int = 42

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_macro_f1"  # monitored metric
    early_stopping_mode: str = "max"

    # Checkpoint saving
    save_best_only: bool = True
    checkpoint_save_path: str = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    resume_from_checkpoint: Optional[str] = None  # path to resume

    # Logging
    log_dir: str = LOG_DIR
    logging_steps: int = 100
    eval_steps: int = 500
    tensorboard: bool = True
    wandb: bool = False                       # set True to enable WandB
    wandb_project: str = "banglabert-multitask"
    wandb_run_name: str = "hybrid-banglabert-plus-plus"

    # Data split ratios
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Class imbalance strategy: "smote", "oversample", "class_weight", "none"
    imbalance_strategy: str = "class_weight"

    # Output directory for plots, tables, etc.
    output_dir: str = OUTPUT_DIR


# ─────────────────────────────────────────────
# DATASET PATHS (populate after downloading)
# ─────────────────────────────────────────────
@dataclass
class DatasetConfig:
    # Dataset 1 — Bengali Text Classification (Kaggle)
    # kaggle datasets download -d raselmeya/bengali-text-classification-dataset
    text_classification_path: str = os.path.join(DATA_DIR, "bengali_text_classification")

    # Dataset 2 — Bangla Hate Speech (HuggingFace)
    # FariaAFrinTisha/Banglahatespeech
    hate_speech_hf_name: str = "FariaAFrinTisha/Banglahatespeech"

    # Dataset 3 — Multi-Label Bangla Hate Speech (HuggingFace)
    # sumaiya-afroze/Multi-Label_Bangla_Hate_Speech_Data
    hate_speech_multilabel_hf_name: str = "sumaiya-afroze/Multi-Label_Bangla_Hate_Speech_Data"

    # Dataset 4 — CC-100 Bangla (for MLM pretraining)
    # Download from: https://data.statmt.org/cc-100/bn.txt.xz
    cc100_path: str = os.path.join(DATA_DIR, "cc100_bn.txt")

    # Dataset 5 — Bangla Fake News (Kaggle)
    fake_news_path: str = os.path.join(DATA_DIR, "bangla_fake_news")

    # Dataset 6 — Bengali News Classification (Kaggle)
    news_classification_path: str = os.path.join(DATA_DIR, "bengali_news_classification")

    # Dataset 7 — Bangla Sentiment (Kaggle / csebuetnlp)
    sentiment_path: str = os.path.join(DATA_DIR, "bangla_sentiment")


# ─────────────────────────────────────────────
# INFERENCE CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class InferenceConfig:
    model_path: str = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    base_model_path: str = PRETRAIN_SAVE
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length: int = 256
    batch_size: int = 32


# ─────────────────────────────────────────────
# GLOBAL CONFIG INSTANCES (import these in other files)
# ─────────────────────────────────────────────
MODEL_CFG    = ModelConfig()
TASK_CFG     = TaskConfig()
PRETRAIN_CFG = PretrainConfig()
TRAIN_CFG    = TrainConfig()
DATA_CFG     = DatasetConfig()
INFER_CFG    = InferenceConfig()


# ─────────────────────────────────────────────
# DEVICE SETUP UTILITY
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Config] Using CUDA — {torch.cuda.device_count()} GPU(s) detected.")
        for i in range(torch.cuda.device_count()):
            print(f"         GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        print("[Config] CUDA not available — using CPU.")
    return device


if __name__ == "__main__":
    print("=" * 60)
    print("BanglaBERT++ Multi-Task Config Summary")
    print("=" * 60)
    print(f"Base Model     : {MODEL_CFG.base_model_name}")
    print(f"Adapted Model  : {MODEL_CFG.adapted_model_path}")
    print(f"Max Seq Length : {MODEL_CFG.max_seq_length}")
    print(f"Tasks          : Hate Speech | Fake News | Sentiment | News Cat")
    print(f"FP16           : {TRAIN_CFG.fp16}")
    print(f"Multi-GPU      : {TRAIN_CFG.use_multi_gpu}")
    print(f"Device         : {get_device()}")
    print("=" * 60)