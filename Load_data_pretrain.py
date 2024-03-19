import os
from scipy.io import loadmat
import h5py
import numpy as np
import scipy.signal as signal
import mne

def get_signal_names(path):
    '''Function to extract the names, frequency, and number of samples of the signals analyzed in the .hea files'''
    with open(path, 'r') as f:
        s = f.readlines()
        s = [x.split() for x in s]
        #The first line of the document contains information about the file such as number of lines in the document
        #and frequency:
        n_signals = int(s[0][1]) - 1
        freq = int(s[0][2])
        #Since we saved the information we needed we can stop considering the first and last line as they 
        #do not contain information about the name of the signal
        s = s[1:-1]
        names = [s[i][-1] for i in range( n_signals)] #The name is the last element in each line
    
    return names, freq

def get_labels(filename):
    data = h5py.File(filename, 'r')
    l = data['data']['sleep_stages']['rem'].shape[1]
    #print(data['data']['sleep_stages']['rem'].shape)
    labels = np.empty((l, 6))
    for i, label in enumerate(data['data']['sleep_stages'].keys()):
        labels[:,i] = data['data']['sleep_stages'][label]
        
    return labels, list(data['data']['sleep_stages'].keys())

def labels_to_events(labels):
    new_labels = np.argmax(labels, axis = 1)
    lab = new_labels[0]
    events = []
    start = 0
    i = 0
    while i < len(new_labels)-1:
        while new_labels[i] == lab and i < len(new_labels)-1:
            i+=1
        end = i
        dur = end +1 - start
        events.append([start, dur, lab])
        lab = new_labels[i]
        start = i+1
    return events

def extract_info(filename, subject):
    if os.path.isdir(filename):
        for file in os.listdir(filename):
            if '.hea' in file:
                names, freq = get_signal_names(os.path.join(filename, file))
            elif '-arousal.mat' in file:
                labels, labels_names = get_labels(os.path.join(filename, file))
            elif 'mat' in file:
                data = loadmat(os.path.join(filename, file))['val'][:6, :]
        
        info = mne.create_info(names[:6], freq, ch_types = 'eeg') #From here to end of function extracted
        raw = mne.io.RawArray(data, info)                         #from the paper's repository  
        raw.resample(100)
        raw.apply_function(lambda x: (x - np.mean(x)) / np.std(x))
        events = labels_to_events(labels)
        label_dict = dict(zip(np.arange(0,6), labels_names))
        events = np.array(events)
        f = lambda x: label_dict[x]
        annotations = mne.Annotations(onset = events[:,0]/freq, duration = events[:,1]/freq, description  = list(map(f,events[:,2])))
        raw.set_annotations(annotations)
        epoch_events = mne.events_from_annotations(raw, chunk_duration = 30)
        info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
        stim_data = np.zeros((1, len(raw.times)))
        stim_raw = mne.io.RawArray(stim_data, info)
        raw.add_channels([stim_raw], force_update_info=True)
        raw.add_events(epoch_events[0], stim_channel = 'STI')
        raw.save(rf'/media/andrea/Windows/Users/andre/Desktop/AIRO/NN Project/Dataset/You snooze you win/training_fif/{subject}_001_30s_raw.fif', overwrite = True)
        
if __name__ == '__main__':
    extract_info(r'/media/andrea/Windows/Users/andre/Desktop/AIRO/NN Project/Dataset/You snooze you win/training/tr03-0005')