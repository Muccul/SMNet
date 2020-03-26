import os
import os.path
import numpy as np
import random
import h5py
import cv2
import glob
import torch.utils.data as udata
import torch

def normalize(data):
    return data/255.

def data_rotate(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def prepare_data(data_path='data', train_filename='train_data.h5'):
    # train
    print('process training data')
    files = glob.glob(os.path.join(data_path, 'pristine_images_color', '*'))
    files.sort()
    h5f = h5py.File(train_filename, 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        Img = torch.tensor(Img)
        Img = Img.permute(2,0,1)
        Img = Img.numpy()
        Img = np.float32(normalize(Img))
        # rorate image
        for j in range(8):
            Img = data_rotate(Img, j)
            h5f.create_dataset(str(train_num), data=Img)
            train_num += 1
        print(train_num)
    h5f.close()
    print('training set, # samples %d\n' % train_num)

    # val
    print('\nprocess validation data')
    files = glob.glob(os.path.join(data_path, 'CBSD68', '*.bmp'))
    files.sort()
    h5f = h5py.File('val_data.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)
        img = img.numpy()
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True, data_root='train_data.h5'):
        super(Dataset, self).__init__()
        self.train = train
        self.data_root = data_root
        if self.train:
            h5f = h5py.File(data_root, 'r')
        else:
            h5f = h5py.File('val_data.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.data_root, 'r')
        else:
            h5f = h5py.File('val_data.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

if __name__ == '__main__':

    prepare_data(data_path='data', train_filename='train_data.h5')

    # import visdom
    #
    # viz = visdom.Visdom()
    # dataset = Dataset(train=True, data_root="train_data.h5")
    # loader = udata.DataLoader(dataset, batch_size=8, num_workers=8, drop_last=True)
    # print(len(loader.dataset))
    # for x in loader:
    #     print(x.size())
    #     viz.images(x, nrow=2, win='image')
