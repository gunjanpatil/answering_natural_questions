# class definitions of custom dataset types
from torch.utils.data import Dataset


class SimplifiedNaturalQADataset(Dataset):
    def __init__(self, id_list) -> object:
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        return self.id_list[index]

class QADatasetWithNegativeExamples(SimplifiedNaturalQADataset):
    def __init__(self, id_list, neg_id_list) -> object:
        super().__init__(id_list)
        self.neg_id_list = neg_id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        return self.id_list[index]