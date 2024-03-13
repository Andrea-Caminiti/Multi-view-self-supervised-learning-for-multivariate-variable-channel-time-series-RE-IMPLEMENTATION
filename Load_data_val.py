import mne

def read_cassette(file):

    raw = mne.io.read_raw_edf(file, stim_channel='Event marker', preload = True, verbose='INFO')
    return raw
    
def main():
    read_cassette(r'C:\Users\andre\Desktop\AIRO\NN Project\Multi-view-self-supervised-learning-for-multivariate-variable-channel-time-series-RE-IMPLEMENTATION\Dataset\sleep-cassette\SC4001E0-PSG.edf')
    
if __name__ == '__main__':
    main()