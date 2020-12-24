"""
Load natural questions dataset which is already prepared and downloaded to a folder
"""
from datasets import load_from_disk, DatasetDict


def load_data(path: str, train_split_num: int) -> DatasetDict:
    """load and split data from path

    Load dataset from path folder, split into train, validation and test

    Args:
        path (str): path to a folder where dataset is saved
        train_split_num (int): example number at which we want train dataset to be split into train and validation

    Returns:
        dataset_loaded (DatasetDict)L loaded and split dataset
    """
    dataset_loaded = load_from_disk(path)
    total_train_examples = len(dataset_loaded['train'])

    # splitting train examples into train and validation and treating validation examples as test set
    train_split = dataset_loaded['train'].select(range(0, train_split_num), keep_in_memory=True)
    validation_split = dataset_loaded['train'].select(range(train_split_num, total_train_examples), keep_in_memory=True)
    test_split = dataset_loaded['validation']
    return DatasetDict({"train": train_split, "validation": validation_split, "test": test_split})
