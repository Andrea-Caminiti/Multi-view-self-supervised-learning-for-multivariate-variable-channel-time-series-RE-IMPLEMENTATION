import mne
import glob
import os

def read_cassette(file): #Extracted from the paper's repository

    raw = mne.io.read_raw_edf(file, stim_channel='Event marker', preload = True, verbose='INFO')
    file_split = file.split('\ '.strip())
    subject = file_split[-1][:5]
    session = file_split[-1][5:7]
    file_path = '\ '.strip().join(file_split[:-1])
    
    anno_path = glob.glob(f'{file_path}\{subject}{session}*-Hypnogram.edf')[0]
    
    annot_train = mne.read_annotations(anno_path)
    raw.set_annotations(annot_train, emit_warning=False)
    out_path = rf'C:\Users\andre\Desktop\AIRO\NN Project\Dataset\{subject}\ '.strip()
    os.makedirs(out_path, exist_ok = True)
    raw.save(f'{out_path}{subject}{session}_raw.fif', overwrite = True)

    return raw
    
def main():
    read_cassette(r'C:\Users\andre\Desktop\AIRO\NN Project\Multi-view-self-supervised-learning-for-multivariate-variable-channel-time-series-RE-IMPLEMENTATION\Dataset\sleep-cassette\SC4001E0-PSG.edf')
    
if __name__ == '__main__':
    main()