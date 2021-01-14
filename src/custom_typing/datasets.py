# class definitions of custom dataset types
from torch.utils.data import Dataset


class SimplifiedNaturalQADataset(Dataset):
    def __init__(self, id_list) -> object:
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        return self.id_list[index]
