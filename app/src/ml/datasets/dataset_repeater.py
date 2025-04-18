from torch.utils.data import Dataset


class DatasetRepeater(Dataset):
    """Pass several times over the same dataset for better i/o performance."""

    def __init__(self, dataset, num_repeats: int = 100) -> None:
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self) -> int:
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx: int):
        return self.dataset[idx % self.dataset.__len__()]
