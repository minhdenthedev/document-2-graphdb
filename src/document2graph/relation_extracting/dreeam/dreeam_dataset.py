from torch.utils.data import Dataset


class DreeamDataset(Dataset):
    def __init__(self, features: list):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]
