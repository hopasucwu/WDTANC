from scipy.special import jv
import numpy as np
from config import get_parameter

params = get_parameter()
nfft = params['n_fft']
c_ = 343

def generate_base_matrix(M):  #[Q, 2M+1]
    base_matrix = []
    for m in range(2 * M + 1):
        b_v = []
        for n in range(2 * M + 1):
            b_v.append(np.exp(1j * ((m - M) * 2 * np.pi / (2 * M + 1) * n)))
        base_matrix.append(b_v)
    base_matrix = np.array(base_matrix)
    base_matrix = base_matrix.transpose(1, 0) #[Q, 2*M+1]
    return base_matrix

def generate_bessel_matrix(R1, M, if_reg = True, f_index = -1):
    bessel_matrix = []
    for f in range(int(nfft // 2 + 1)):
        if f_index >= 0 and f!=f_index:
            bessel_matrix.append(np.array([[0] * (2*M+1)]))
            continue
        b_v = []
        for m in range(2 * M + 1):
            bes = jv((m - M), 2 * np.pi * f * 8000 / (c_ * int(nfft // 2 + 1)) * R1)
            # if abs(bes) < params['_lambda']:
            #     if bes >= 0:
            #         bes = params['_lambda']
            #     else:
            #         bes = -params['_lambda']
            if if_reg:
                if bes >= 0:
                    bes += params['_lambda']
                else:
                    bes -= params['_lambda']
            b_v.append(bes)
        b_v1 = np.array(b_v) #envelop(np.array(b_v))
        bessel_matrix.append(b_v1.reshape(1, -1))
    bessel_matrix = np.concatenate(bessel_matrix, axis = 0)  #[F, 2M+1]
    return bessel_matrix
