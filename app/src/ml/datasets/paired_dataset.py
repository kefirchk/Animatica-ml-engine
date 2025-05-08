import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    """Dataset of pairs for services."""

    def __init__(self, initial_dataset, number_of_pairs: int, seed=0) -> None:
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if not pairs_list:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs["source"].isin(videos), pairs["driving"].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append((name_to_index[pairs["driving"].iloc[ind]], name_to_index[pairs["source"].iloc[ind]]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {"driving_" + key: value for key, value in first.items()}
        second = {"source_" + key: value for key, value in second.items()}

        return {**first, **second}
