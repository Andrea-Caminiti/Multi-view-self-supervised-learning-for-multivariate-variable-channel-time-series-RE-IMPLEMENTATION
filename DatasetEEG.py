import torch
from torch.utils.data import DatasetTorch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os
from Load_data_pretrain import extract_info
from Load_data_val import read_cassette
from dn3.data.dataset import Dataset, Thinker, EpochTorchRecording
import mne

class EEGdataset(DatasetTorch):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        eeg, label = self.data[index]
        return eeg, label
    
def get_thinkers(raw, subject):
    sessions = dict()
    events = mne.events_from_annotations(raw, chunk_duration=30)[0]
    epochs = mne.Epochs(raw, events)
    recording = EpochTorchRecording(epochs, force_label=True)
    sessions[subject] = recording
    if len(sessions.keys()) > 0:
        return Thinker(sessions)
    
def split(thinkers):
    train, val = train_test_split(list(thinkers.keys()), test_size = 0.2, random_state=0)
    train_thinkers = dict()
    for subj in train:
        train_thinkers[subj] = thinkers[subj]
    val_thinkers = dict()
    for subj in val:
        val_thinkers[subj] = thinkers[subj]
    return train_thinkers, val_thinkers

def build_dataset(pretrain_path, finetune_path):
    thinker_pretrain = dict()
    thinker_finetune= dict()
    for subject in os.listdir(pretrain_path):
        raw = extract_info(os.path.join(pretrain_path, subject))
        thinker_pretrain[subject] = get_thinkers(raw, subject)

    thinker_pretrain_train, thinker_pretrain_val = split(thinker_pretrain) 
    pretrain_train_dset = Dataset(thinker_pretrain_train)
    pretrain_val_dset = Dataset(thinker_pretrain_val)

    pretrain_train_dset = EEGdataset(pretrain_train_dset)
    pretrain_val_dset = EEGdataset(pretrain_val_dset)

    pretrain_train_loader = DataLoader(pretrain_train_dset, batch_size=64, shuffle = True, num_workers=2)
    pretrain_val_loader = DataLoader(pretrain_val_dset, batch_size=64, shuffle = True, num_workers=2)


    for subject in os.listdir(finetune_path):
        raw = read_cassette(os.path.join(finetune_path, subject))
        thinker_finetune[subject] = get_thinkers(raw, subject)
    
    thinker_finetune_trainVal, thinker_finetune_test = split(thinker_finetune)
    thinker_finetune_train, thinker_finetune_val = split(thinker_finetune_trainVal)

    finetune_train_dset = Dataset(thinker_finetune_train)
    finetune_val_dset = Dataset(thinker_finetune_val)
    finetune_test_dset = Dataset(thinker_finetune_test)

    finetune_train_dset = EEGdataset(finetune_train_dset)
    finetune_val_dset = EEGdataset(finetune_val_dset)
    finetune_test_dset = EEGdataset(finetune_test_dset)

    finetune_train_loader = DataLoader(finetune_train_dset, batch_size=64, shuffle = True, num_workers=2)
    finetune_val_loader = DataLoader(finetune_val_dset, batch_size=64, shuffle = True, num_workers=2)
    finetune_test_loader = DataLoader(finetune_test_dset, batch_size=64, shuffle = True, num_workers=2)
    
    return pretrain_train_loader, pretrain_val_loader,finetune_train_loader, finetune_val_loader, finetune_test_loader,
