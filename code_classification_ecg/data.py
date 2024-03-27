import numpy as np
from torch.utils.data import Dataset
import torch
import wfdb
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import copy
import pickle

class Invert:
    def __init__(self, rs=np.random):
        self.rs = rs

    def __call__(self, sample):
        if self.rs.rand() < 0.5:
            sample['data'] = -sample['data']

        return sample

import math
class Sample:
    def __init__(self, size, rs=np.random):
        self.size = size
        self.rs = rs

    def __call__(self, sample):
        data = sample['data']
        overset = data.shape[1] - self.size
        if overset == 0:
            pass
        elif overset < 0:
            pass
        else:
            start_idx = self.rs.randint(overset)
            data = data[:, start_idx:start_idx + self.size]
        sample['data'] = data
        return sample

class Zoom:
    def __init__(self, minz=.75, maxz=1.5, rs=np.random):
        self.minz = np.log(minz)
        self.maxz = np.log(maxz)
        self.rs = rs

    def __call__(self, sample):
        data = sample['data']
        if self.rs.rand() < .5:
            z = np.exp(self.rs.uniform(self.minz, self.maxz))
            data = ndimage.zoom(data, (1, z))
        sample['data'] = data

        return sample


class Rescale:
    def __init__(self, quantile=0.99):
        self.quantile = quantile
        pass
    def __call__(self, sample):
        sample['data'] = sample['data'] / np.quantile(np.abs(sample['data']), self.quantile)
        return sample

class RandScale:
    def __init__(self, minscale=1/2, maxscale=2, rs=np.random):
        self.minscale = minscale
        self.maxscale = maxscale
        self.rs = rs

    def __call__(self, sample):
        data = sample['data']
        scale = self.rs.uniform(self.minscale, self.maxscale)

        sample['data'] = data * scale
        sample['scale'] = scale
        return sample

class RandomBaselineShift:
    def __init__(self, minv=-.25, maxv=.25, rs=np.random):
        self.rs = rs
        self.vals = (minv, maxv)

    def __call__(self, sample):
        data = sample['data']
        data += self.rs.uniform(*self.vals)
        sample['data'] = data
        return sample


class RandOffset:
    def __init__(self, rs=np.random):
        self.rs = rs

    def __call__(self, sample):
        data = sample['data']
        offset = self.rs.rand()
        sample['data'] = np.roll(data, int(offset * len(data)))
        sample['offset'] = offset
        return sample


class RandomlySwitchChannels:
    def __init__(self, rs=np.random):
        self.rs = rs

    def __call__(self, sample):
        data = sample['data']
        idcs = np.arange(len(data))
        self.rs.shuffle(idcs)
        #         self.rs.shuffle(data)
        sample['data'] = data[idcs]

        return sample


class RandomChannelOff:
    def __init__(self, p, rs=np.random):
        self.rs = rs
        self.p = p

    def __call__(self, sample):
        sample['data'] = np.multiply(sample['data'], (self.rs.rand(8, 1) > self.p).astype(int))
        return sample


class ToTensor:
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    def __call__(self, sample):
        sample['data'] = torch.tensor(sample['data'], dtype=self.dtype, device=self.device)
        sample['y'] = torch.tensor(sample['y'], device=self.device)
        return sample

class PadToMinimum:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):

        if sample['data'].shape[1] < self.size:
            pad = int((self.size - sample['data'].shape[1]) // 2 + 1)
            sample['data'] = np.pad(sample['data'], ((0, 0), (pad, pad)), mode='constant', constant_values=0)
        return sample

def filter_signal(signal, dtype=np.float32):
    maxval = np.percentile(np.abs(signal), 99)
    return signal/maxval


class ECG:
    def __init__(self, fname):
        self.header = wfdb.rdheader(fname)
        self.record = wfdb.rdrecord(fname)

class ECGPN2017(ECG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = self.record.p_signal
        assert (data.shape[1] == 1 and len(data.shape) == 2)
        self.data = filter_signal(data.ravel())
        self.fs = self.record.fs


class ECGPN2020(ECG):
    labels = np.array(['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parse_comments()
        self.signal = self.record.p_signal

    def parse_comments(self):
        d = dict()
        for el in self.header.comments:
            k, v = el.split(':')
            d[k.strip()] = v.strip()

        self.age = d['Age']
        self.diagnosis = d['Dx']
        #         self.labels = np.array([labels[el.strip()] for el in d['Dx'].split(',')])
        self.label_mask = np.isin(self.labels, d['Dx'].split(','))
        self.prescription = d['Rx']
        self.history = d['Hx']
        self.surgery = d['Sx']

class PhysioNet2017Dataset(Dataset):
    def __init__(self, data_set, sample_size,
                 transform=None,
                 root_dir=None,
                 load_as_tensor=False,
                 binary=False,
                 debug=False):

        root_dir = Path(root_dir)

        ref = np.loadtxt(root_dir / 'REFERENCE.csv', delimiter=',', dtype=str)
        pids = ref[:, 0]
        if binary:
            labels = (ref[:, 1] != 'N').astype(int)
        else:
            labels = (ref[:, 1] == 'N').astype(int) * 0
            labels += (ref[:, 1] == 'A').astype(int) * 1
            labels += (ref[:, 1] == 'O').astype(int) * 2
            labels += (ref[:, 1] == '~').astype(int) * 3

        data = list()

        idcs = np.arange(len(pids))

        np.random.RandomState(808).shuffle(idcs)

        sep = int(len(idcs) * .9)
        if data_set == 'train':
            selected_idcs = idcs[:sep]
        elif data_set == 'validation':
            selected_idcs = idcs[sep:]
        else:
            raise Exception('"{}" is not a valid dataset. Please choose "train" or "validation".'.format(data_set))

        if debug:
            selected_idcs = selected_idcs[:100]

        samples = list()
        for pid, label in tqdm(zip(pids[selected_idcs], labels[selected_idcs]), desc='loading {} ecgs'.format(data_set)):
            fname = Path(root_dir) / '{}.hea'.format(pid)
            ecg = ECGPN2017(str(fname)[:-4])
            d = ecg.data[None].astype(np.float32)

            if len(d) < sample_size:
                print('skipping {}'.format(pid))
                continue
            if load_as_tensor:
                d = torch.from_numpy(d).cuda()

            samples.append({'data': d, 'pid': pid, 'y':int(label), 'fs': 300})

        self.transform = transform
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class PhysioNet2020Dataset(Dataset):
    def __init__(self, data_set,
                 root_dir,
                 transform=None,
                 self_distillation=None,
                 eight_channel_data=True,
                 debug=False):

        root_dir = Path(root_dir)
        fnames = list(root_dir.glob('*.hea'))

        np.random.RandomState(808).shuffle(fnames)

        sep = int(len(fnames) * .9)
        if data_set == 'all':
            pass
        elif data_set == 'train':
            fnames = fnames[:sep]#[:10]
        elif data_set == 'validation':
            fnames = fnames[sep:]#[:10]
        else:
            raise Exception('"{}" is not a valid dataset. Please choose "train" or "validation".'.format(data_set))

        if debug:
            fnames = fnames[:100]
        samples = list()
        for fname in tqdm(fnames):
            ecg = ECGPN2020(str(fname)[:-4])

            samples.append({'data': ecg.signal.T, 'pid': fname.stem, 'y': ecg.label_mask.astype(np.float32)})

        if self_distillation:
            probabilities = pickle.load(open(self_distillation, "rb"))
            for idx in tqdm(range(len(samples)), desc='Loading distilled probabilities.'):
                 samples[idx]['y'] = probabilities[samples[idx]['pid']]


        self.transform = transform
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

from scipy.io import loadmat

def load_challenge_data(filename):
    x = loadmat(filename.with_suffix('.mat'))
    data = np.asarray(x['val'], dtype=np.float64)
    header_file = filename.with_suffix('.hea')
    with open(header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

from collections import OrderedDict
def convert(data, header, channel_order):
    num_channels, frq, numsmps = (int(el) for el in header[0].split()[1:4])
    assert (numsmps == data.shape[1])

    if frq != 500:
        data = ndimage.zoom(data, (1, 500 / frq))

    channel_order = np.array(channel_order)

    channel_mapping = np.zeros((len(channel_order),), dtype=int)
    for idx in range(12):
        l = header[idx+1]
        vals = l.split()
        mv = float(vals[2].rstrip('/mV'))
        offset = float(vals[4])
        firstval = int(vals[5])
        checksum = int(vals[6])
        leadname = vals[8].strip()
        if leadname in channel_order:
            channel_mapping[int(np.where(leadname == channel_order)[0])] = idx

        data[idx] /=  mv

    data = data[channel_mapping]
    return data



import pandas as pd
import ast

class PTBXL:
    def __init__(self, dataset, transform=None, data_dir=Path(r'D:\data\ECG\ptb_xl'), hr=False, debug=False):
        classes = np.array(['NORM', 'MI', 'STTC', 'CD', 'HYP'])
        df_statements = pd.read_csv(data_dir / 'scp_statements.csv', index_col=0)
        df_statements = df_statements[df_statements.diagnostic == 1]
        df_statements['diagnosis_label'] = df_statements.diagnostic_class.apply(
            lambda x: np.squeeze(np.where(classes == x)))

        def aggregate_diagnostic_to_encoding(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in df_statements.index:
                    tmp.append(df_statements.loc[key].diagnosis_label)
            idcs = list(set(tmp))
            mask = np.zeros(len(classes), dtype=int)
            mask[idcs] = 1
            return mask

        df_ptbxl = pd.read_csv(data_dir / 'ptbxl_database.csv', index_col='ecg_id')
        if dataset == 'train':
            df_ptbxl = df_ptbxl[df_ptbxl.strat_fold < 9]
        elif dataset == 'validation':
            df_ptbxl = df_ptbxl[df_ptbxl.strat_fold == 9]
        elif dataset == 'test':
            df_ptbxl = df_ptbxl[df_ptbxl.strat_fold == 10]

        if debug:
            df_ptbxl = df_ptbxl.iloc[:100]

        df_ptbxl.scp_codes = df_ptbxl.scp_codes.apply(lambda x: ast.literal_eval(x))
        df_ptbxl['diagnostic_label_encoding'] = df_ptbxl.scp_codes.apply(aggregate_diagnostic_to_encoding)
        samples = list()
        for ecg_id, row in tqdm(df_ptbxl.iterrows(), total=len(df_ptbxl), desc=f'Loading {dataset} set'):
            if hr:
                fname = row.filename_hr
            else:
                fname = row.filename_lr
            data, meta = wfdb.rdsamp(str(data_dir / fname))
            samples.append(dict(data=data.T.astype(np.float32),
                                fs=meta['fs'],
                                pid=ecg_id,
                                y=row['diagnostic_label_encoding'].astype(np.float32)))

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample
