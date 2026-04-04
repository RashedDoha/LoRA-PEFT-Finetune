from datasets import load_dataset
from datasets import Dataset, DatasetDict
from config import get_settings


def get_dataset() -> DatasetDict:
    config = get_settings()
    dataset = load_dataset(config.dataset_name)
    return dataset

def get_train_dataset() -> Dataset:
    dataset = get_dataset()
    return dataset["train"]

def get_eval_dataset() -> Dataset:
    dataset = get_dataset()
    return dataset["test"]