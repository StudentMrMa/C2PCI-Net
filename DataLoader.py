

import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from configurations import Config


class CustomDataset(Dataset):
    def __init__(self, eeg_map_data, eeg_stat_data, peri_data, labels):
        self.eeg_map_data = eeg_map_data
        self.eeg_stat_data = eeg_stat_data
        self.peri_data = peri_data
        self.labels = labels

    def __len__(self):
        return len(self.eeg_map_data)

    def __getitem__(self, index):
        eeg_map = self.eeg_map_data[index]
        eeg_stat = self.eeg_stat_data[index]
        peri = self.peri_data[index]
        label = self.labels[index]
        return eeg_map, eeg_stat, peri, label

def load_data(dataset, label_type='valence'):
    try:
        if dataset == "DEAP":
            eeg_map_data = loadmat("D:\\MaZhuang_Workspace\\Project\\Dataset_psd\\DEAP_PSD.mat") 
            # eeg_stat_data = loadmat("D:\\MaZhuang_Workspace\\Project\\Dataset_stat\\DEAP_stat.mat")
            peri_data = loadmat("D:\\MaZhuang_Workspace\\Project\\stand_and_normal\\stand_and_normal_data\\DEAP_peri_normal.mat")
            eeg_stat_data = loadmat("D:\MaZhuang_Workspace\Project\DEAP_stat_normal.mat")
            # peri_data = loadmat("D:\\MaZhuang_Workspace\\Project\\Dataset_peri\\DEAP_peri.mat")
            labels = loadmat("D:\\MaZhuang_Workspace\\Project\\Labels\\DEAP_labels.mat")

            eeg_map_data = eeg_map_data['PSD_seg_nobl_log_sde_zc4topo32_features']
            # eeg_stat_data = eeg_stat_data['eeg_en_stat'].reshape(-1,32,7)
            eeg_stat_data = eeg_stat_data['data'].reshape(-1,32,7)
            # peri_data = peri_data['peri_feature'].reshape(-1,55,1)
            peri_data = peri_data['DEAP_peri_normal'].reshape(-1,55,1)
            labels = labels['labels'] # 19200,2  (valence arousal)

            # eeg_map_data = eeg_map_data[:,0:1,:,:]
            # eeg_map_data = eeg_map_data[:,1:2,:,:]
            # eeg_map_data = eeg_map_data[:,2:3,:,:]
            # eeg_map_data = eeg_map_data[:,3:4,:,:]
            # eeg_map_data = eeg_map_data[:,4:5,:,:]

            # peri_data = peri_data[:,:17]
            # peri_data = peri_data[:,17:22]
            # peri_data = peri_data[:,22:27]
            # peri_data = peri_data[:,27:35]
            # peri_data = peri_data[:,35:]

            # labels = labels[:,0] # valence
            # labels = labels[:,1] # arousal
            

            print("load DEAP data successfully !!")
            print(eeg_map_data.shape,eeg_stat_data.shape,peri_data.shape,labels.shape)

        elif dataset == "HCI":
            eeg_map_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_psd\HCI_PSD.mat")
            # eeg_stat_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_stat\HCI_en_stat.mat")
            eeg_stat_data = loadmat("D:\MaZhuang_Workspace\Project\HCI_stat_normal.mat") # normal stat data
            peri_data = loadmat("D:\MaZhuang_Workspace\Project\HCI_peri_normal.mat") # normal peri data
            # peri_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_peri\HCI_peri.mat")
            labels = loadmat("D:\MaZhuang_Workspace\Project\Labels\HCI_labels.mat")

            eeg_map_data = eeg_map_data['PSD_seg_nobl_log_sde_zc4topo32_features']
            eeg_stat_data = eeg_stat_data['data'].reshape(-1,32,7)
            peri_data = peri_data['HCI_peri_normal'].reshape(-1,49,1)
            labels = labels['label']
            # labels = labels[:,0]
            # labels = labels[:,1]

            # eeg_map_data = eeg_map_data[:,0:1,:,:]
            # eeg_map_data = eeg_map_data[:,1:2,:,:]
            # eeg_map_data = eeg_map_data[:,2:3,:,:] 
            # eeg_map_data = eeg_map_data[:,3:4,:,:]
            # eeg_map_data = eeg_map_data[:,4:5,:,:]


            
            # peri_data = peri_data[:,:18]
            # peri_data = peri_data[:,18:23]
            # peri_data = peri_data[:,23:28]
            # peri_data = peri_data[:,28:49]
            


            print("load HCI data successfully !!")
            print(eeg_map_data.shape,eeg_stat_data.shape,peri_data.shape,labels.shape)

        elif dataset == "SEED-IV":
            eeg_map_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_psd\SEED_IV_PSD.mat") 
            # eeg_stat_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_stat\SEED_IV_stat.mat")
            eeg_stat_data = loadmat("D:\MaZhuang_Workspace\Project\SEED_IV_stat_normal.mat") # normal stat
            peri_data = loadmat("D:\MaZhuang_Workspace\Project\SEED_IV_peri_normal.mat") # normal peri data
            # peri_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_peri\SEED_IV_eye_feature.mat")
            labels = loadmat("D:\MaZhuang_Workspace\Project\Labels\SEED_IV_labels.mat")

            eeg_map_data = eeg_map_data['PSD_seg_nobl_log_sde_zc4topo32_features']
            eeg_stat_data = eeg_stat_data['SEED_IV_stat_normal'].reshape(-1,62,7)
            peri_data = peri_data['SEED_IV_peri_normal'].reshape(-1,31,1)
            labels = labels['labels'] # 37575,1


            # eeg_map_data = eeg_map_data[:,0:1,:,:]
            # eeg_map_data = eeg_map_data[:,1:2,:,:]
            # eeg_map_data = eeg_map_data[:,2:3,:,:]
            # eeg_map_data = eeg_map_data[:,3:4,:,:] 
            # eeg_map_data = eeg_map_data[:,4:5,:,:] 

            print("load SEED-IV data successfully !")
            print(eeg_map_data.shape,eeg_stat_data.shape,peri_data.shape,labels.shape)

        elif dataset == "SEED-V":
            eeg_map_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_psd\SEED_V_PSD.mat") 
            eeg_stat_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_stat\SEED_V_stat.mat")
            # eeg_stat_data = loadmat("D:\MaZhuang_Workspace\Project\stand_and_normal\stand_and_normal_data\SEED_V_stat_stand.mat")
            # peri_data = loadmat("D:\\MaZhuang_Workspace\\Project\\SEED_V_peri_normal.mat")
            peri_data = loadmat("D:\MaZhuang_Workspace\Project\Dataset_peri\SEED_V_eye_feature.mat")
            labels = loadmat("D:\MaZhuang_Workspace\Project\Labels\SEED_V_labels.mat")
            
            eeg_map_data = eeg_map_data['PSD_seg_nobl_log_sde_zc4topo32_features']
            eeg_stat_data = eeg_stat_data['SEED_V_Stat'].reshape(-1,62,7) 
            peri_data = peri_data['eye_feature'].reshape(-1,33,1) 
            labels = labels['labels'] 


            # eeg_map_theat = eeg_map_data[:,0:1,:,:] 
            # eeg_map_alpha = eeg_map_data[:,1:2,:,:] 
            # eeg_map_slow_alpha = eeg_map_data[:,2:3,:,:] 
            # eeg_map_Beta = eeg_map_data[:,3:4,:,:] 
            # eeg_map_Gamma = eeg_map_data[:,4:5,:,:] 


            # eeg_map_data = eeg_map_data[:,0:1,:,:] 
            # eeg_map_data = eeg_map_data[:,1:2,:,:] 
            # eeg_map_data = eeg_map_data[:,2:3,:,:]
            # eeg_map_data = eeg_map_data[:,3:4,:,:]
            # eeg_map_data = eeg_map_data[:,4:5,:,:]

            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

            print("load SEED-V data successfully !")
            print(eeg_map_data.shape,eeg_stat_data.shape,peri_data.shape,labels.shape)

        if dataset == "DEAP":
            peri_data = peri_data
        elif dataset == "HCI":
            peri_data = peri_data
        elif dataset == "SEED-IV":
            peri_data = peri_data[:,:22]
        elif dataset == "SEED-V":
            peri_data = peri_data[:,:24]

        if dataset == "DEAP":
            if label_type == 'valence':
                labels = labels[:,0]
            elif label_type == 'arousal':
                labels = labels[:,1]
        elif dataset == "HCI":
            if label_type == 'valence':
                labels = labels[:,0]
            elif label_type == 'arousal':
                labels = labels[:,1]
        else: 
            labels = labels

        # print(eeg_map_data.shape,eeg_stat_data.shape,peri_data.shape,labels.shape)
        return eeg_map_data, eeg_stat_data, peri_data, labels
        

    except KeyError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


def split_dataset(eeg_map_data, eeg_stat_data, peri_data, labels, dataset_name):

    num_subjects = {"DEAP": 32,"SEED-IV": 15,"SEED-V": 16, "HCI" : 24 }.get(dataset_name, 0)
    samples_per_subject = {"DEAP": 600,"SEED-IV": 2505,"SEED-V": 1823, "HCI" :528 }.get(dataset_name, 0)

    train_datasets = []
    test_datasets = []

    for p in range(num_subjects):
        indices = np.arange(p * samples_per_subject, (p + 1) * samples_per_subject)
        test_indices = list(indices)
        train_indices = list(set(range(len(eeg_map_data))) - set(indices))

        # do not load the whole datasets , the cuda memonary is not enough

        trainset_eeg_map = eeg_map_data[train_indices]
        trainset_eeg_stat = eeg_stat_data[train_indices]
        trainset_peri = peri_data[train_indices]
        trainset_labels = labels[train_indices]

        testset_eeg_map = eeg_map_data[test_indices]
        testset_eeg_stat = eeg_stat_data[test_indices]
        testset_peri = peri_data[test_indices]
        testset_labels = labels[test_indices]


        train_datasets.append(CustomDataset(trainset_eeg_map, trainset_eeg_stat, trainset_peri, trainset_labels))
        test_datasets.append(CustomDataset(testset_eeg_map, testset_eeg_stat, testset_peri, testset_labels))

    return train_datasets, test_datasets


# def dataset_loaders(dataset_name, batch_size=128):
#     eeg_map_data, eeg_stat_data, peri_data, labels = load_data(dataset_name, label_type='valence')
#     num_subjects = {"DEAP": 32, "HCI": 24, "SEED-IV": 15, "SEED-V": 16}.get(dataset_name, 0)

#     train_loaders = []
#     test_loaders = []

#     for test_subject in range(num_subjects):
#         train_data_tuple, test_data_tuple = split_dataset(eeg_map_data, eeg_stat_data, peri_data, labels, dataset_name)

#         train_dataset = CustomDataset(train_data_tuple[test_subject][0], train_data_tuple[test_subject][1], train_data_tuple[test_subject][2], train_data_tuple[test_subject][3])
#         test_dataset = CustomDataset(test_data_tuple[test_subject][0], test_data_tuple[test_subject][1], test_data_tuple[test_subject][2], test_data_tuple[test_subject][3])

#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#         train_loaders.append(train_loader)
#         test_loaders.append(test_loader)

#     return train_loaders, test_loaders

# def data_generator(data_loader):
#     while True:
#         for batch in data_loader:
#             yield batch

def dataset_loaders(dataset_name,batch_size=128):
    if dataset_name == "DEAP":
        num_subjects = 32
        samples_per_subject = 600
        batch_size = 128 
    elif dataset_name == "HCI":
        num_subjects = 24
        samples_per_subject = 528
        batch_size = 128 
    elif dataset_name == "SEED-IV":
        num_subjects = 15
        samples_per_subject = 2505
        batch_size = 128  
    elif dataset_name == "SEED-V":
        num_subjects = 16
        samples_per_subject = 1823
        batch_size = 128  
    else:
        raise ValueError(f"there is no database here--->: {dataset_name}")
  

    eeg_map_data, eeg_stat_data, peri_data, labels = load_data(dataset_name, label_type='valence')
    train_datasets, test_datasets = split_dataset(eeg_map_data, eeg_stat_data, peri_data, labels, dataset_name)

    train_loaders = []
    test_loaders = []

    for test_subject in range(num_subjects):
    
        train_data = ConcatDataset([train_datasets[i] for i in range(num_subjects) if i != test_subject])
        test_data = test_datasets[test_subject]


        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

        # return train_loader, test_loader
    return train_loaders, test_loaders


dataset_loaders("SEED-V",batch_size=128)  # or "DEAP" "SEED-IV", "SEED-V","HCI"
