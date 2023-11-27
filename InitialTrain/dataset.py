from torch.utils.data import Dataset
import numpy as np
import h5py

class MyDataset(Dataset):
    def __init__(self,labels):
       self.data_label = labels

    def __getitem__(self, idx):
        labels = np.array(self.data_label[idx])
        labels = labels.astype(np.float32)

        features = []
        for label in labels:
            for index,ll in enumerate(label):
                if(abs(ll)==0.0):
                    label[index]=0.0

            path = './DATA'
            f = h5py.File(f'{path}/NEW_RING_allspace_num3_RC_192_256.hdf5', 'r')

            array = f[str(label[0]) + '_' + str(label[1]) + '_' + str(label[2]) + '_' + str(label[3])]
            features.append(np.array(array))
            f.close()

        features = np.array(features)
        features = features.astype(np.float32)

        return features, labels

    def __len__(self):
        return len(self.data_label)








