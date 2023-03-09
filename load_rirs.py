import numpy as np
from scipy.special import jv, hankel2
from scipy import signal
from config import get_parameter
import matplotlib.pyplot as plt
import os
import torch

params = get_parameter()
nfft = params['n_fft']
c_ = 343
M = 5
np.set_printoptions(threshold=np.inf)


def get_reverb_secpath():
    M = 15
    ref_channels = 3
    N_mics = 2 * M + 1 + 10000
    ns, nm = 2 * M + 1 + ref_channels, N_mics
    rir_dict = dict()
    for s in range(ns):
        print(s)
        for m in range(nm):
            datapath = os.path.join('./rir_2d_nosel_M15', 'rir_' + str(s) + '_' + str(m) + '.npy')
            rir = np.load(datapath).reshape(-1)
            
            if rir.shape[0] < nfft:
                rir = np.concatenate([rir, np.zeros(nfft - rir.shape[0])])
            elif rir.shape[0] > nfft:
                rir = rir[:nfft]
            rir_dict[(s, m)] = rir
    
    source_err = [rir_dict[(i, j)] for i in range(ref_channels) for j in range(2* M + 1)]
    source_err = np.concatenate(source_err).reshape(ref_channels, 2 * M + 1, -1)
    F_pri_rirs = np.fft.rfft(source_err).transpose(2, 1, 0)
    print(F_pri_rirs.shape)

    source_ins = [rir_dict[(i, j)] for i in range(ref_channels) for j in range(2* M + 1,N_mics)]
    source_ins = np.concatenate(source_ins).reshape(ref_channels, N_mics - (2 * M + 1), -1)
    F_pri_rirs_ins = np.fft.rfft(source_ins).transpose(2, 1, 0)
    print(F_pri_rirs_ins.shape)

    loud_err = [rir_dict[(i, j)] for i in range(ref_channels, ref_channels + 2 * M + 1) for j in range(2* M + 1)]
    loud_err = np.concatenate(loud_err).reshape(2 * M + 1, 2 * M + 1, -1)
    F_sec_rirs = np.fft.rfft(loud_err).transpose(2, 1, 0)
    print(F_sec_rirs.shape)

    loud_ins = [rir_dict[(i, j)] for i in range(ref_channels, ref_channels + 2 * M + 1) 
                            for j in range(2* M + 1,N_mics)]
    loud_ins = np.concatenate(loud_ins).reshape(2 * M + 1, N_mics - (2 * M + 1), -1)
    F_sec_rirs_ins = np.fft.rfft(loud_ins).transpose(2, 1, 0)
    print(F_sec_rirs_ins.shape)

    return F_sec_rirs, F_sec_rirs_ins, F_pri_rirs, F_pri_rirs_ins

if __name__ == '__main__':
    F_sec_rirs, F_sec_rirs_ins, F_pri_rirs, F_pri_rirs_ins = get_reverb_secpath()
    np.save('F_sec_rirs_reverb.npy', F_sec_rirs)
    np.save('F_sec_rirs_ins_reverb.npy', F_sec_rirs_ins)
    np.save('F_pri_rirs_reverb.npy', F_pri_rirs)
    np.save('F_pri_rirs_ins_reverb.npy', F_pri_rirs_ins)
