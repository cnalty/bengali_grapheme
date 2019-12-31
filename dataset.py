import pandas as pd
import torch
import numpy as np

class Dataset():
    def __init__(self, type="train"):
        self.type = type
        self.data = {}
        self.labels = pd.read_csv("{}.csv".format(type))
        for i in range(1):
            curr = pd.read_parquet("{}_image_data_{}.parquet".format(type, i))
            curr.set_index("image_id")
            self.data.update(curr.T.to_dict('list'))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data["{}_{}".format(self.type.capitalize(), index)]/255.0
        labels = self.data[index][1:-1]
        data = data.reshap(137, -1)
        data = torch.tensor(data).float()

        return data, labels[0], labels[1], labels[2]

def main():
    from PIL import Image

    w, h = 137, 236
    dataset = Dataset()
    im, l1, l2, l3 = dataset.__getitem__(0)
    im = im.numpy()
    im = im * 255
    im = im.astype(np.uint8)[0]
    img = Image.fromarray(im)

    img.show()

if __name__ == "__main__":
    main()
