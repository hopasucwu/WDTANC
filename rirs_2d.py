import numpy as np
from scipy.special import hankel2
import sfs
import matplotlib.pyplot as plt

L = [6, 6, 2]  # room dimensions
max_order = 2 # maximum order of image sources
coeffs = [0.75, 0.75, 0.75, 0.75, 1., 1.]  # wall reflection coefficients
n_fft = 320

M = 15
source_pos_list = [[3 + 3 * np.cos(240 / 180 * np.pi), 3 + 3 * np.sin(240 / 180 * np.pi), 1],
                     [3 + 2.2, 3, 1], [3 + 2.5 * np.cos(45 / 180 * np.pi), 3 + 2.5 * np.sin(45 / 180 * np.pi), 1]]
mic_pos_list = []
for i in range(2 * M + 1):
    mic_pos_list.append([3 + 1 * np.cos(i * 2*np.pi / (2*M + 1)), 3 + 1 * np.sin(i * 2*np.pi / (2*M + 1)), 1])
    source_pos_list.append([3 + 2 * np.cos(i * 2*np.pi / (2*M + 1)), 3 + 2 * np.sin(i * 2*np.pi / (2*M + 1)), 1])

xv = np.linspace(0.1, 5.9, 100)
yv = np.linspace(0.1, 5.9, 100)
X, Y = np.meshgrid(xv, yv)
grid = np.concatenate([np.expand_dims(X, [-1]), np.expand_dims(Y, [-1])], axis = -1)
print(grid.shape)
grid = grid.reshape(-1 , 2)

mic_pos_list.extend([[grid[i][0], grid[i][1], 1] for i in range(grid.shape[0])])
print(len(mic_pos_list))

def generate_rir_from_sources(virtual_source_pos, source_strength, mic_pos, f_index = -1):
    rirs = np.zeros(n_fft // 2 + 1) + 1j * np.zeros(n_fft // 2 + 1)
    rs = np.sqrt(np.sum((virtual_source_pos - mic_pos) ** 2, axis = -1))
    # print(rs.shape)
    for i in range(n_fft // 2 + 1):
        if f_index >= 0 and i != f_index:
            continue
        k = 2 * np.pi * (i / (n_fft // 2 + 1) * 8000) / 343
        if k ==0:
            continue
        rirs[i] = np.sum(hankel2(0, k * rs) * source_strength)
        if f_index >= 0:
            rirs[i] *= (n_fft // 2 + 1)
    rirs = np.fft.irfft(rirs)
    return rirs

for s, sp in enumerate(source_pos_list):
    print(s)
    x0, y0, z0 = sp[0], sp[1], sp[2]
    xs, wall_count = sfs.util.image_sources_for_box([x0, y0], L[0:2], max_order)
    # print(xs.shape)
    source_strength = np.prod(coeffs[:4] ** wall_count, axis = 1)
    for m, mp in enumerate(mic_pos_list):
        if m % 1000 == 0:
            print(s, m)
        # index = np.sqrt(np.sum((xs - mp[:2]) ** 2, axis = -1)) / 343 * 16000 < n_fft
        # rir = generate_rir_from_sources(xs[index], source_strength[index], np.array(mp[:2]))
        rir = generate_rir_from_sources(xs, source_strength, np.array(mp[:2]))
        np.save('../rir_2d_nosel_M15/rir_'+str(s) + '_' + str(m) +'.npy', rir)
