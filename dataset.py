# ============================================================
# dataset.py — Data Loading, Cleaning, Tokenization, and
# Multi-Task Dataset Classes for BanglaBERT++ Framework
# ============================================================

import os
import re
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE

from config import MODEL_CFG, TRAIN_CFG, TASK_CFG, DATA_CFG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BANGLA TEXT CLEANING UTILITIES
# ─────────────────────────────────────────────

# Unicode range for Bangla script
BANGLA_RANGE = re.compile(r'[\u0980-\u09FF]')

# Common noise patterns to remove
URL_RE      = re.compile(r'http[s]?://\S+')
EMAIL_RE    = re.compile(r'\S+@\S+\.\S+')
MENTION_RE  = re.compile(r'@\w+')
HASHTAG_RE  = re.compile(r'#\w+')
EMOJI_RE    = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    "]+", flags=re.UNICODE
)
MULTI_SPACE_RE = re.compile(r'\s+')
# English character filter (keep only Bangla + punctuation + digits)
NON_BANGLA_RE  = re.compile(r'[^\u0980-\u09FF\u0964\u0965\s.,!?।\-\'\"0-9]')


def normalize_unicode(text: str) -> str:
    """
    Apply NFC Unicode normalization to ensure consistent
    Bangla character encoding across different sources.
    """
    return unicodedata.normalize("NFC", text)


def clean_bangla_text(
    text: str,
    remove_english: bool = False,
    remove_numbers: bool = False,
) -> str:
    """
    Comprehensive cleaning pipeline for Bangla text:
      1. NFC Unicode normalization
      2. Remove URLs, emails, mentions, hashtags
      3. Remove emojis
      4. Optionally remove non-Bangla characters
      5. Optionally remove digits
      6. Collapse multiple whitespace
      7. Strip leading/trailing whitespace

    Args:
        text            : Raw input string
        remove_english  : Strip English/Latin characters
        remove_numbers  : Strip digit characters

    Returns:
        Cleaned Bangla text string
    """
    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)

    if remove_english:
        text = NON_BANGLA_RE.sub(" ", text)

    if remove_numbers:
        # Remove both ASCII and Bangla digits
        text = re.sub(r'[0-9\u09E6-\u09EF]', " ", text)

    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ─────────────────────────────────────────────
# LABEL ENCODERS (fit on full data; saved for inference)
# ─────────────────────────────────────────────
class MultiTaskLabelEncoders:
    """
    Container for per-task label encoders.
    Keeps consistent integer → string mapping across splits.
    """
    def __init__(self):
        self.hate     = LabelEncoder()
        self.fake     = LabelEncoder()
        self.sentiment= LabelEncoder()
        self.news_cat = LabelEncoder()

    def fit_hate(self, labels):
        self.hate.fit(labels)
        TASK_CFG.hate_speech_labels = list(self.hate.classes_)
        TASK_CFG.hate_speech_num_classes = len(self.hate.classes_)

    def fit_fake(self, labels):
        self.fake.fit(labels)
        TASK_CFG.fake_news_labels = list(self.fake.classes_)
        TASK_CFG.fake_news_num_classes = len(self.fake.classes_)

    def fit_sentiment(self, labels):
        self.sentiment.fit(labels)
        TASK_CFG.sentiment_labels = list(self.sentiment.classes_)
        TASK_CFG.sentiment_num_classes = len(self.sentiment.classes_)

    def fit_news_cat(self, labels):
        self.news_cat.fit(labels)
        TASK_CFG.news_category_labels = list(self.news_cat.classes_)
        TASK_CFG.news_category_num_classes = len(self.news_cat.classes_)


LABEL_ENCODERS = MultiTaskLabelEncoders()


# ─────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────

def load_hate_speech_data() -> pd.DataFrame:
    """
    Load and merge the two HuggingFace hate speech datasets:
      - FariaAFrinTisha/Banglahatespeech   (binary)
      - sumaiya-afroze/Multi-Label_Bangla_Hate_Speech_Data (multi-label → binarised)

    Returns:
        DataFrame with columns: ['text', 'hate_label']
    """
    records = []

    # ── Dataset 2: FariaAFrinTisha/Banglahatespeech ─────────
    try:
        logger.info("Loading FariaAFrinTisha/Banglahatespeech ...")
        ds = load_dataset("FariaAFrinTisha/Banglahatespeech", trust_remote_code=True)
        for split in ds:
            df = ds[split].to_pandas()
            # Inspect available columns and map accordingly
            # Expected columns: 'text' / 'label' (0=NoHate, 1=Hate)
            text_col  = next((c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()), df.columns[0])
            label_col = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), df.columns[1])
            df = df.rename(columns={text_col: "text", label_col: "hate_label_raw"})
            df["text"] = df["text"].apply(clean_bangla_text)
            # Standardise label to binary string
            df["hate_label"] = df["hate_label_raw"].apply(
                lambda x: "Hate" if str(x).lower() in ["1", "hate", "yes"] else "NoHate"
            )
            records.append(df[["text", "hate_label"]])
            logger.info(f"  Split '{split}': {len(df)} samples")
    except Exception as e:
        logger.warning(f"Could not load FariaAFrinTisha dataset: {e}")

    # ── Dataset 3: Multi-Label Bangla Hate Speech ───────────
    try:
        logger.info("Loading sumaiya-afroze/Multi-Label_Bangla_Hate_Speech_Data ...")
        ds2 = load_dataset("sumaiya-afroze/Multi-Label_Bangla_Hate_Speech_Data", trust_remote_code=True)
        for split in ds2:
            df2 = ds2[split].to_pandas()
            text_col = next((c for c in df2.columns if "text" in c.lower() or "sentence" in c.lower()), df2.columns[0])
            df2 = df2.rename(columns={text_col: "text"})
            df2["text"] = df2["text"].apply(clean_bangla_text)
            # For multi-label: any hate label → "Hate", else "NoHate"
            # Check if there's any hate-indicating column
            hate_cols = [c for c in df2.columns if c.lower() not in ["text", "id", "index"]]
            if hate_cols:
                df2["hate_label"] = df2[hate_cols].apply(
                    lambda row: "Hate" if any(
                        str(v).lower() in ["1", "hate", "yes", "true"] for v in row
                    ) else "NoHate",
                    axis=1
                )
            else:
                df2["hate_label"] = "NoHate"
            records.append(df2[["text", "hate_label"]])
            logger.info(f"  Split '{split}': {len(df2)} samples")
    except Exception as e:
        logger.warning(f"Could not load sumaiya-afroze dataset: {e}")

    if not records:
        logger.error("No hate speech data loaded. Check dataset availability.")
        return pd.DataFrame(columns=["text", "hate_label"])

    df_hate = pd.concat(records, ignore_index=True)
    df_hate = df_hate.dropna(subset=["text", "hate_label"])
    df_hate = df_hate[df_hate["text"].str.len() > 3]  # remove very short texts
    logger.info(f"Total hate speech samples: {len(df_hate)}")
    logger.info(df_hate["hate_label"].value_counts().to_string())
    return df_hate


def load_fake_news_data() -> pd.DataFrame:
    """
    Load Bangla Fake News dataset from local CSV files.
    Expected structure: CSV with 'text'/'content' and 'label'/'category' columns.

    Place the Kaggle dataset files inside:
        data/bangla_fake_news/

    Returns:
        DataFrame with columns: ['text', 'fake_label']
    """
    data_path = DATA_CFG.fake_news_path
    records   = []

    if os.path.exists(data_path):
        for fname in os.listdir(data_path):
            if fname.endswith(".csv"):
                fpath = os.path.join(data_path, fname)
                try:
                    df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
                    # Detect text and label columns flexibly
                    text_col  = next((c for c in df.columns if any(
                        kw in c.lower() for kw in ["text", "content", "article", "body", "news"]
                    )), None)
                    label_col = next((c for c in df.columns if any(
                        kw in c.lower() for kw in ["label", "class", "type", "fake", "category"]
                    )), None)

                    if text_col and label_col:
                        df = df.rename(columns={text_col: "text", label_col: "fake_label_raw"})
                        df["text"] = df["text"].apply(clean_bangla_text)
                        df["fake_label"] = df["fake_label_raw"].apply(
                            lambda x: "Fake" if str(x).lower() in [
                                "1", "fake", "false", "মিথ্যা"
                            ] else "Real"
                        )
                        records.append(df[["text", "fake_label"]])
                        logger.info(f"Loaded fake news file: {fname} ({len(df)} rows)")
                except Exception as e:
                    logger.warning(f"Error reading {fname}: {e}")
    else:
        logger.warning(
            f"Fake news data directory not found: {data_path}\n"
            "  → Download from Kaggle and place CSV files in data/bangla_fake_news/"
        )

    if not records:
        logger.warning("Creating synthetic fake news placeholder. Replace with real data.")
        # Synthetic placeholder so pipeline runs end-to-end
        df_fake = pd.DataFrame({
            "text": ["এটি একটি নকল সংবাদ।"] * 100 + ["এটি সত্যিকারের সংবাদ।"] * 100,
            "fake_label": ["Fake"] * 100 + ["Real"] * 100,
        })
        return df_fake

    df_fake = pd.concat(records, ignore_index=True)
    df_fake = df_fake.dropna(subset=["text", "fake_label"])
    df_fake = df_fake[df_fake["text"].str.len() > 3]
    logger.info(f"Total fake news samples: {len(df_fake)}")
    logger.info(df_fake["fake_label"].value_counts().to_string())
    return df_fake


def load_sentiment_data() -> pd.DataFrame:
    """
    Load Bangla Sentiment dataset (Kaggle / csebuetnlp split).

    Place CSV files in: data/bangla_sentiment/

    Returns:
        DataFrame with columns: ['text', 'sentiment_label']
    """
    data_path = DATA_CFG.sentiment_path
    records   = []

    if os.path.exists(data_path):
        for fname in os.listdir(data_path):
            if fname.endswith(".csv"):
                fpath = os.path.join(data_path, fname)
                try:
                    df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
                    text_col  = next((c for c in df.columns if any(
                        kw in c.lower() for kw in ["text", "sentence", "review", "comment"]
                    )), None)
                    label_col = next((c for c in df.columns if any(
                        kw in c.lower() for kw in ["label", "sentiment", "class", "polarity"]
                    )), None)

                    if text_col and label_col:
                        df = df.rename(columns={text_col: "text", label_col: "sentiment_raw"})
                        df["text"] = df["text"].apply(clean_bangla_text)
                        # Normalise to Positive / Negative / Neutral
                        def map_sentiment(x):
                            x = str(x).lower().strip()
                            if x in ["positive", "pos", "1", "2", "ইতিবাচক"]:
                                return "Positive"
                            elif x in ["negative", "neg", "-1", "নেতিবাচক"]:
                                return "Negative"
                            else:
                                return "Neutral"
                        df["sentiment_label"] = df["sentiment_raw"].apply(map_sentiment)
                        records.append(df[["text", "sentiment_label"]])
                        logger.info(f"Loaded sentiment file: {fname} ({len(df)} rows)")
                except Exception as e:
                    logger.warning(f"Error reading {fname}: {e}")
    else:
        logger.warning(
            f"Sentiment data directory not found: {data_path}\n"
            "  → Download from Kaggle and place CSV files in data/bangla_sentiment/"
        )

    if not records:
        logger.warning("Creating synthetic sentiment placeholder. Replace with real data.")
        df_sent = pd.DataFrame({
            "text": ["ভালো।"] * 100 + ["খারাপ।"] * 100 + ["ঠিক আছে।"] * 100,
            "sentiment_label": ["Positive"] * 100 + ["Negative"] * 100 + ["Neutral"] * 100,
        })
        return df_sent

    df_sent = pd.concat(records, ignore_index=True)
    df_sent = df_sent.dropna(subset=["text", "sentiment_label"])
    df_sent = df_sent[df_sent["text"].str.len() > 3]
    logger.info(f"Total sentiment samples: {len(df_sent)}")
    logger.info(df_sent["sentiment_label"].value_counts().to_string())
    return df_sent


def load_news_classification_data() -> pd.DataFrame:
    """
    Load Bengali News Classification dataset (Kaggle: raselmeya).
    Covers Datasets 1 and 6.

    Place CSV files in: data/bengali_text_classification/ or
                         data/bengali_news_classification/

    Returns:
        DataFrame with columns: ['text', 'news_category']
    """
    paths   = [DATA_CFG.text_classification_path, DATA_CFG.news_classification_path]
    records = []

    for data_path in paths:
        if os.path.exists(data_path):
            for fname in os.listdir(data_path):
                if fname.endswith(".csv"):
                    fpath = os.path.join(data_path, fname)
                    try:
                        df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
                        text_col = next((c for c in df.columns if any(
                            kw in c.lower() for kw in ["text", "content", "article", "body", "news"]
                        )), None)
                        cat_col  = next((c for c in df.columns if any(
                            kw in c.lower() for kw in ["category", "label", "class", "topic", "type"]
                        )), None)

                        if text_col and cat_col:
                            df = df.rename(columns={text_col: "text", cat_col: "news_category"})
                            df["text"] = df["text"].apply(clean_bangla_text)
                            df["news_category"] = df["news_category"].astype(str).str.strip().str.lower()
                            records.append(df[["text", "news_category"]])
                            logger.info(f"Loaded news file: {fname} ({len(df)} rows)")
                    except Exception as e:
                        logger.warning(f"Error reading {fname}: {e}")

    if not records:
        logger.warning(
            "News classification data not found. Creating placeholder.\n"
            "  → Download Datasets 1 & 6 from Kaggle into data/ directories."
        )
        categories = ["politics", "sports", "entertainment", "science", "technology"]
        df_news = pd.DataFrame({
            "text": [f"এটি {c} বিষয়ক সংবাদ।" for c in categories] * 40,
            "news_category": categories * 40,
        })
        return df_news

    df_news = pd.concat(records, ignore_index=True)
    df_news = df_news.dropna(subset=["text", "news_category"])
    df_news = df_news[df_news["text"].str.len() > 3]
    # Remove extremely rare categories (< 10 samples)
    counts = df_news["news_category"].value_counts()
    valid_cats = counts[counts >= 10].index
    df_news = df_news[df_news["news_category"].isin(valid_cats)]
    logger.info(f"Total news classification samples: {len(df_news)}")
    logger.info(df_news["news_category"].value_counts().to_string())
    return df_news


# ─────────────────────────────────────────────
# MULTI-TASK DATAFRAME BUILDER
# ─────────────────────────────────────────────

def build_multitask_dataframe() -> pd.DataFrame:
    """
    Merge all four task datasets into a single DataFrame.
    Each row represents one text sample and has per-task labels.
    Columns not applicable to a row are set to -1 (ignored in loss).

    Returns:
        Unified DataFrame with columns:
          text, hate_label, fake_label, sentiment_label, news_category
    """
    logger.info("=" * 60)
    logger.info("Loading and merging all task datasets ...")
    logger.info("=" * 60)

    df_hate   = load_hate_speech_data()
    df_fake   = load_fake_news_data()
    df_sent   = load_sentiment_data()
    df_news   = load_news_classification_data()

    # Fit label encoders on full data BEFORE splitting
    LABEL_ENCODERS.fit_hate(df_hate["hate_label"])
    LABEL_ENCODERS.fit_fake(df_fake["fake_label"])
    LABEL_ENCODERS.fit_sentiment(df_sent["sentiment_label"])
    LABEL_ENCODERS.fit_news_cat(df_news["news_category"])

    # Encode labels to integers
    df_hate["hate_label"]       = LABEL_ENCODERS.hate.transform(df_hate["hate_label"])
    df_fake["fake_label"]       = LABEL_ENCODERS.fake.transform(df_fake["fake_label"])
    df_sent["sentiment_label"]  = LABEL_ENCODERS.sentiment.transform(df_sent["sentiment_label"])
    df_news["news_category"]    = LABEL_ENCODERS.news_cat.transform(df_news["news_category"])

    # Build unified format with sentinel -1 for missing task labels
    records = []

    for _, row in df_hate.iterrows():
        records.append({
            "text": row["text"],
            "hate_label": int(row["hate_label"]),
            "fake_label": -1,
            "sentiment_label": -1,
            "news_category": -1,
        })

    for _, row in df_fake.iterrows():
        records.append({
            "text": row["text"],
            "hate_label": -1,
            "fake_label": int(row["fake_label"]),
            "sentiment_label": -1,
            "news_category": -1,
        })

    for _, row in df_sent.iterrows():
        records.append({
            "text": row["text"],
            "hate_label": -1,
            "fake_label": -1,
            "sentiment_label": int(row["sentiment_label"]),
            "news_category": -1,
        })

    for _, row in df_news.iterrows():
        records.append({
            "text": row["text"],
            "hate_label": -1,
            "fake_label": -1,
            "sentiment_label": -1,
            "news_category": int(row["news_category"]),
        })

    df_all = pd.DataFrame(records)
    df_all = df_all.sample(frac=1, random_state=TRAIN_CFG.seed).reset_index(drop=True)
    logger.info(f"Total merged samples: {len(df_all)}")
    return df_all


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split on a combined task indicator so that
    each split has balanced task representation.

    Returns:
        (train_df, val_df, test_df)
    """
    # Create a composite key for stratification
    df["_strat_key"] = (
        df["hate_label"].clip(lower=0).astype(str) + "_" +
        df["fake_label"].clip(lower=0).astype(str) + "_" +
        df["sentiment_label"].clip(lower=0).astype(str) + "_" +
        df["news_category"].clip(lower=0).astype(str)
    )

    val_test_ratio = TRAIN_CFG.val_ratio + TRAIN_CFG.test_ratio
    test_of_valtest = TRAIN_CFG.test_ratio / val_test_ratio

    try:
        train_df, temp_df = train_test_split(
            df, test_size=val_test_ratio,
            stratify=df["_strat_key"],
            random_state=TRAIN_CFG.seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_of_valtest,
            stratify=temp_df["_strat_key"],
            random_state=TRAIN_CFG.seed
        )
    except ValueError:
        # Fallback: unstratified split when some classes are too rare
        logger.warning("Stratified split failed; using random split.")
        train_df, temp_df = train_test_split(
            df, test_size=val_test_ratio, random_state=TRAIN_CFG.seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_of_valtest, random_state=TRAIN_CFG.seed
        )

    train_df = train_df.drop(columns=["_strat_key"])
    val_df   = val_df.drop(columns=["_strat_key"])
    test_df  = test_df.drop(columns=["_strat_key"])

    logger.info(f"Split → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# CLASS WEIGHT COMPUTATION
# ─────────────────────────────────────────────

def compute_task_class_weights(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """
    Compute inverse-frequency class weights for each task.
    Used with CrossEntropyLoss(weight=...) to handle imbalance.

    Args:
        df: Training DataFrame (must not include val/test to avoid leakage)

    Returns:
        Dict mapping task name → weight tensor on CPU
    """
    weights = {}
    task_col_map = {
        "hate_speech": ("hate_label", TASK_CFG.hate_speech_num_classes),
        "fake_news":   ("fake_label", TASK_CFG.fake_news_num_classes),
        "sentiment":   ("sentiment_label", TASK_CFG.sentiment_num_classes),
        "news_cat":    ("news_category", TASK_CFG.news_category_num_classes),
    }

    for task, (col, n_classes) in task_col_map.items():
        valid_labels = df[col][df[col] >= 0].values
        if len(valid_labels) == 0:
            weights[task] = torch.ones(n_classes)
            continue
        classes = np.arange(n_classes)
        present = np.unique(valid_labels)
        w = compute_class_weight("balanced", classes=present, y=valid_labels)
        full_w = np.ones(n_classes)
        for cls, weight in zip(present, w):
            full_w[cls] = weight
        weights[task] = torch.FloatTensor(full_w)
        logger.info(f"[{task}] class weights: {full_w.round(3)}")

    return weights


# ─────────────────────────────────────────────
# PYTORCH DATASET CLASS
# ─────────────────────────────────────────────

class BanglaMultiTaskDataset(Dataset):
    """
    PyTorch Dataset for the BanglaBERT++ multi-task framework.
    Tokenizes text on-the-fly using the BanglaBERT tokenizer.
    Supports dynamic padding via a custom collate function.

    Args:
        df        : DataFrame with columns [text, hate_label, fake_label,
                    sentiment_label, news_category]
        tokenizer : HuggingFace AutoTokenizer for BanglaBERT
        max_length: Maximum token sequence length (default from ModelConfig)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = None):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length= max_length or MODEL_CFG.max_seq_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row  = self.df.iloc[idx]
        text = str(row["text"])

        # Tokenize with truncation; padding handled by collate_fn
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,           # dynamic padding in collate_fn
            return_tensors=None,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        item = {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"],  dtype=torch.long),
            "token_type_ids": torch.tensor(encoding["token_type_ids"],  dtype=torch.long),
            # Task labels (-1 = not applicable for this sample)
            "hate_label":       torch.tensor(int(row["hate_label"]),       dtype=torch.long),
            "fake_label":       torch.tensor(int(row["fake_label"]),       dtype=torch.long),
            "sentiment_label":  torch.tensor(int(row["sentiment_label"]),  dtype=torch.long),
            "news_category":    torch.tensor(int(row["news_category"]),    dtype=torch.long),
        }
        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Dynamic padding collate function.
    Pads all sequences in a batch to the length of the longest sequence,
    which is typically shorter than max_seq_length → faster training.
    """
    pad_id    = 0  # BanglaBERT pad token id
    max_len   = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list      = []
    attention_mask_list = []
    token_type_ids_list = []
    hate_labels, fake_labels, sentiment_labels, news_categories = [], [], [], []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Right-pad with zeros
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask_list.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        token_type_ids_list.append(
            torch.cat([item["token_type_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        hate_labels.append(item["hate_label"])
        fake_labels.append(item["fake_label"])
        sentiment_labels.append(item["sentiment_label"])
        news_categories.append(item["news_category"])

    return {
        "input_ids":        torch.stack(input_ids_list),
        "attention_mask":   torch.stack(attention_mask_list),
        "token_type_ids":   torch.stack(token_type_ids_list),
        "hate_label":       torch.stack(hate_labels),
        "fake_label":       torch.stack(fake_labels),
        "sentiment_label":  torch.stack(sentiment_labels),
        "news_category":    torch.stack(news_categories),
    }


# ─────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────

def get_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    tokenizer,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders.
    Training loader uses WeightedRandomSampler to balance task representation.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = BanglaMultiTaskDataset(train_df, tokenizer)
    val_ds   = BanglaMultiTaskDataset(val_df,   tokenizer)
    test_ds  = BanglaMultiTaskDataset(test_df,  tokenizer)

    # Weighted sampler: give each sample a weight based on task rarity
    # Tasks with fewer samples get up-sampled proportionally
    task_counts = {
        "hate":  (train_df["hate_label"] >= 0).sum(),
        "fake":  (train_df["fake_label"] >= 0).sum(),
        "sent":  (train_df["sentiment_label"] >= 0).sum(),
        "news":  (train_df["news_category"] >= 0).sum(),
    }
    max_count = max(task_counts.values())

    sample_weights = []
    for _, row in train_df.iterrows():
        if row["hate_label"] >= 0:
            w = max_count / max(task_counts["hate"], 1)
        elif row["fake_label"] >= 0:
            w = max_count / max(task_counts["fake"], 1)
        elif row["sentiment_label"] >= 0:
            w = max_count / max(task_counts["sent"], 1)
        else:
            w = max_count / max(task_counts["news"], 1)
        sample_weights.append(w)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CFG.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=TRAIN_CFG.dataloader_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_CFG.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=TRAIN_CFG.dataloader_num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=TRAIN_CFG.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=TRAIN_CFG.dataloader_num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders ready → "
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)} | "
        f"Test batches: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# TOKENIZER LOADER
# ─────────────────────────────────────────────

def load_tokenizer(model_name_or_path: str = None) -> AutoTokenizer:
    """
    Load the BanglaBERT tokenizer.
    After Phase 1 pretraining, the tokenizer is saved alongside the model,
    so this function loads from the adapted model path if it exists.

    Args:
        model_name_or_path: Override path/name; defaults to adapted model.

    Returns:
        Loaded AutoTokenizer instance
    """
    from config import PRETRAIN_SAVE

    if model_name_or_path is None:
        # Use adapted model if Phase 1 is complete, else base model
        if os.path.exists(os.path.join(PRETRAIN_SAVE, "tokenizer_config.json")):
            model_name_or_path = PRETRAIN_SAVE
            logger.info(f"Tokenizer loaded from adapted model: {PRETRAIN_SAVE}")
        else:
            model_name_or_path = MODEL_CFG.base_model_name
            logger.info(f"Tokenizer loaded from base model: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("Testing dataset pipeline ...")

    tokenizer = load_tokenizer()
    df_all    = build_multitask_dataframe()
    train_df, val_df, test_df = split_dataset(df_all)
    weights   = compute_task_class_weights(train_df)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, tokenizer)

    # Inspect one batch
    batch = next(iter(train_loader))
    print("\nSample batch keys:", list(batch.keys()))
    print("input_ids shape  :", batch["input_ids"].shape)
    print("hate_label       :", batch["hate_label"][:5])
    print("fake_label       :", batch["fake_label"][:5])
    print("sentiment_label  :", batch["sentiment_label"][:5])
    print("news_category    :", batch["news_category"][:5])
    print("Dataset pipeline test PASSED ✓")