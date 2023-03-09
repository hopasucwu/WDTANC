import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
mu = 1e-4
NO = 31
NI = 31
NE = 31
app = '_fxlms_washing1e4'

envs = '_reverb'
xv = np.linspace(0.1, 5.9, 100)
yv = np.linspace(0.1, 5.9, 100)
X, Y = np.meshgrid(xv, yv)
grid = np.concatenate([np.expand_dims(X, [-1]), np.expand_dims(Y, [-1])], axis = -1)
grid = grid.reshape(-1 , 2)
inside_pos = []
for i in range(len(grid)):
    xr = grid[i][0]
    yr = grid[i][1]
    xr = xr #(xg / 100) * 6
    yr = yr #(yg / 100) * 6
    r = np.sqrt((xr - 3) ** 2 + (yr - 3) ** 2)
    if r < 1:
        inside_pos.append(i)
inside_pos = np.array(inside_pos)

def getdata_from_rirs(F_data, rirs):
    ###
    #F_data:[F, T, (NO)]
    #rirs:[F, NE, NO]
    ###
    if len(F_data.shape) == 2:
        F_data = torch.cat([F_data.unsqueeze(-1) for i in range(rirs.shape[-1])], axis = -1)
    F_data = F_data.permute(1, 0, 2) #[T, F, NO]
    getdata = torch.matmul(rirs, F_data.unsqueeze(-1)).squeeze(-1) #[T, F, NE]
    getdata = getdata.permute(1, 0, 2) #[F, T, NE]
    return getdata

def generate_source_data(wavname = None):
    wf = sf.SoundFile(wavname)
    source_data = wf.read(dtype = 'float32')
    # source_data = np.cos(np.linspace(0, 500 * 2 * np.pi * 200, 500 * 16000)).astype(np.float32)
    source_data = source_data / np.max(np.abs(source_data))
    return source_data

def noise_control(filter_weight, input_x):
    #filter_weight:[NO, NI, F, F, 2]
    #input_x:[F, 1, NI]
    input_x = input_x.squeeze(-2).permute(1, 0).unsqueeze(-2) #[NI, 1, F]
    out_ls = (torch.matmul(input_x.real, filter_weight[:, :, :, :, 0]) + 
                1j * torch.matmul(input_x.real, filter_weight[:, :, :, :, 1]) + 
                1j * torch.matmul(input_x.imag, filter_weight[:, :, :, :, 0]) - 
                torch.matmul(input_x.imag, filter_weight[:, :, :, :, 1])).squeeze(-2)  #[NO, NI, F]
    out_ls = torch.sum(out_ls, axis = 1).permute(1, 0).unsqueeze(-2) #[F, 1, NO]
    return out_ls

def Record_adaptive(wavname = None, index = ''):
    print(wavname)
    F_pri_rirs = np.load('../F_pri_rirs'+envs+'.npy')
    F_pri_rirs = torch.from_numpy(F_pri_rirs).type(torch.complex64).to(device) #[F, NE, NO]
    F_sec_rirs = np.load('../F_sec_rirs'+envs+'.npy')
    F_sec_rirs = torch.from_numpy(F_sec_rirs).type(torch.complex64).to(device) #[F, NE, NO]

    filter_weight = torch.zeros(NO, NI, 161, 161, 2).to(device)
    filter_weight.requires_grad = True

    source_data = torch.from_numpy(generate_source_data(wavname = wavname))
    F_source_data = torch.stft(source_data, n_fft=320, win_length = 320,
                                hop_length=int(0.5 * 320), center = False, return_complex = True) #[F, T]
    in_data = F_source_data[:, :-1]
    trg_data = F_source_data[:, 1:]
    Length = in_data.shape[-1]
    rec_out_ls = []
    rec_trg_data = []
    rec_err = []
    rec_noise = []
    print(Length)
    optimizer = torch.optim.SGD([filter_weight], lr = mu)
    loss_fn = torch.nn.MSELoss().to(device)

    for i in range(1):
        for l in range(Length):
            input_x = in_data[:, l:l+1].to(device) #[F, 1]
            trg_y = trg_data[:, l:l+1].to(device) #[F, 1]
            margin_in = getdata_from_rirs(input_x, F_pri_rirs).type(torch.complex64).to(device) #[F, 1, NE]
            margin_trg = getdata_from_rirs(trg_y, F_pri_rirs).type(torch.complex64).to(device) #[F, 1, NE]

            out_ls = noise_control(filter_weight, margin_in) #[F, 1, NO]
            # print('out_ls', out_ls.shape)
            out_margin = getdata_from_rirs(out_ls, F_sec_rirs) #[F, 1, NE]
            # print('out_margin', out_margin.shape)
            # print('margin_trg', margin_trg[:6])
            ri_out_margin = torch.cat([out_margin.real.unsqueeze(-1), out_margin.imag.unsqueeze(-1)], axis = -1)
            ri_margin_trg = torch.cat([margin_trg.real.unsqueeze(-1), margin_trg.imag.unsqueeze(-1)], axis = -1)
            #[F, 1, NE, 2]
            # print(ri_out_margin.shape)
            # print(ri_margin_trg.shape)
            loss = loss_fn(-ri_out_margin, ri_margin_trg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            err_margin = out_margin + margin_trg
            rec_out_ls.append(out_ls.cpu().detach().numpy())
            rec_trg_data.append(trg_y.cpu().detach().numpy())
            rec_err.append(err_margin.cpu().detach().numpy())
            rec_noise.append(margin_trg.cpu().detach().numpy())
            if l % 1000 == 0:
                print(np.sum(np.abs(err_margin.cpu().detach().numpy()) ** 2))
                # print(np.sum(err_margin.cpu().detach().numpy()))
    rec_out_ls = np.concatenate(rec_out_ls[-1000:], axis = -2)
    rec_trg_data = np.concatenate(rec_trg_data[-1000:], axis = -1)
    print(rec_out_ls.shape)
    print(rec_trg_data.shape)
    np.save('out_ls'+app+index+'.npy', rec_out_ls)
    np.save('data_trg'+app+index+'.npy', rec_trg_data)
    rec_err = np.concatenate(rec_err[-128:], axis = -2)
    rec_noise = np.concatenate(rec_noise[-128:], axis = -2)
    print(rec_err.shape)
    print(rec_noise.shape)
    print('snr_margin', 10 * np.log10(np.sum(np.abs(rec_err) ** 2) /np.sum(np.abs(rec_noise) ** 2)))

def test(index = ''):
    out_ls = torch.from_numpy(np.load('out_ls'+app+index+'.npy')).type(torch.complex64)[:, -128:, :].to(device)
    data_trg = torch.from_numpy(np.load('data_trg'+app+index+'.npy')).type(torch.complex64)[:, -128:].to(device)

    F_pri_rirs_ins = np.load('../F_pri_rirs_ins'+envs+'.npy')[:, inside_pos, :]
    F_pri_rirs_ins = torch.from_numpy(F_pri_rirs_ins).type(torch.complex64).to(device) #[F, NE, NO]
    F_sec_rirs_ins = np.load('../F_sc_rirs'+envs+'.npy')[:, inside_pos, :]
    F_sec_rirs_ins = torch.from_numpy(F_sec_rirs_ins).type(torch.complex64).to(device) #[F, NE, NO]
    # F_pri_rirs = np.load('../F_pri_rirs'+envs+'.npy')
    # F_pri_rirs = torch.from_numpy(F_pri_rirs).type(torch.complex64).to(device) #[F, NE, NO]
    # F_sec_rirs = np.load('../F_sec_rirs'+envs+'.npy')
    # F_sec_rirs = torch.from_numpy(F_sec_rirs).type(torch.complex64).to(device) #[F, NE, NO]

    trg_ins = getdata_from_rirs(data_trg, F_pri_rirs_ins)  #[F, T, NE]
    out_ins = getdata_from_rirs(out_ls, F_sec_rirs_ins)
    trg_ins = trg_ins.permute(2, 0, 1).squeeze(0)
    out_ins = out_ins.permute(2, 0, 1).squeeze(0) #[NE, F, T]
    left_ins = trg_ins + out_ins
    snr = 10 * torch.log10(torch.sum(torch.abs(left_ins) ** 2) / torch.sum(torch.abs(trg_ins) ** 2))
    print('snr_ins', snr)

if __name__ == '__main__':
    data_root = '/mnt/hd16t/wdh/anc/simulate/data/train_datas_ori'
    scene_list = os.listdir(data_root)
    for i, scene in enumerate(scene_list):
        wav_path = os.path.join(data_root, scene, 'ch15.wav')
        Record_adaptive(wavname = wav_path, index = str(i))
        test(str(i))
        # test()
