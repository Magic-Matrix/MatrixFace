import torch

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, datas, piplines=None):
        self.datas = datas
        if piplines is None:
            self.piplines = []
        else:
            self.piplines = piplines

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        data = self.get_one_data(idx)

        for pipline in self.piplines:
            data = pipline(data)
        return data
    
    def get_one_data(self, idx):
        data = self.datas[idx]
        return data



