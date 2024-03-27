# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 17:25:20 2017

@author: BobD
"""

import os
from os import path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import interpolation
from scipy import ndimage
from tqdm import tqdm
import pandas as pd
import torch
from glob import glob, iglob
import torch.nn.functional as F
import cv2



def basename(arg):
    try:
        return os.path.splitext(os.path.basename(arg))[0] # is raising an exception faster than an if clause?
    except Exception as e:
        if isinstance(arg, list):
            return [basename(el) for el in arg]
        else:
            raise e

def saveImage(fname, arr, spacing):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, fname, False)


def circle_mask(imsize, radius=1.):
    #    imsize = 480
    xx, yy = np.mgrid[:imsize, :imsize]
    circle = (xx - imsize / 2) ** 2 + (yy - imsize / 2) ** 2
    return circle < (radius * imsize / 2) ** 2


def sitk_save(fname, arr, spacing=None, dtype=np.int16):
    if type(spacing) == type(None):
        spacing = np.ones((len(arr.shape),))
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, fname, True)

saveMHD = sitk_save

def sitk_read(fname):
    img = sitk.ReadImage(fname)
    spacing = img.GetSpacing()[::-1]
    im = sitk.GetArrayFromImage(img)
    return im, spacing

readMHD = sitk_read

def readMHDInfo(mhd_file):
    with open(mhd_file, 'r') as f:
        for l in f:
            if l.startswith('ElementSpacing'):
                spacing = np.array(l[17:].split()[::-1], np.float)
            if l.startswith('DimSize'):
                dims = np.array(l[10:].split()[::-1], np.uint)
    return dims, spacing


def extract_planes(data):
    shape = data.shape
    centers = np.divide(shape, 2).astype(int)
    return data[centers[0]], data[:, centers[1]], data[:, :, centers[2]]





def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    interpolator = sitk.sitkLanczosWindowedSinc
#     interpolator = sitk.sitkLinear
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def resize(img, vox_size):
    img.SetOrigin([0, 0, 0])
    spacing = img.GetSpacing()

    new_spacing = [vox_size] * 3
    x_scale, y_scale, z_scale = np.divide(vox_size, spacing)

    dimension = 3
    new_transform = sitk.AffineTransform(dimension)
    matrix = np.array(new_transform.GetMatrix()).reshape((dimension, dimension))
    matrix[0, 0] = x_scale
    matrix[1, 1] = y_scale
    matrix[2, 2] = z_scale
    new_transform.SetMatrix(matrix.ravel())
    resampled = resample(img, new_transform)
    arr = sitk.GetArrayFromImage(resampled)

    new_shape = np.divide(arr.shape, (z_scale, y_scale, x_scale)).astype(int)
    #     new_spacing = np.multiply(spacing[::-1], (z_scale, y_scale, x_scale))
    return arr[:new_shape[0], :new_shape[1], :new_shape[2]], new_spacing



def fast_resample_image(imdata, spacing, target_spacing, rs=np.random):
    downsample = np.round(np.divide(target_spacing, spacing)).astype(int)
    new_spacing = downsample * np.array(spacing)
    start_idx = np.floor(rs.rand(3) * downsample).astype(int)
    return imdata[start_idx[0]::downsample[0],
           start_idx[1]::downsample[1],
           start_idx[2]::downsample[2]], new_spacing


def makePatch(image, shape=(64, 64, 64), rs=np.random):
    '''
    get patch from image. No padding is applied. The patch is taken from
    within the image.
    '''

    data = image['data']
    extent = image['extent']
    spacing = image['spacing']
    origin = image['origin']
    small = image['small']

    random_pos = (rs.rand(len(extent)) * np.subtract(extent, shape)).astype(int)

    new_data = data[random_pos[0]:random_pos[0] + shape[0],
               random_pos[1]:random_pos[1] + shape[1],
               random_pos[2]:random_pos[2] + shape[2]]
    new_origin = random_pos * spacing + origin

    # pad:
    rest = (32 - np.mod(new_data.shape, 32))
    pad_l = (rs.rand(3) * rest).astype(int)
    pad_r = rest - pad_l
    pad = list(zip(pad_l, pad_r))
    new_data = np.pad(new_data, pad_width=pad, mode='constant')
    #    assert(new_data.shape == shape)
    new_extent = new_data.shape
    new_spacing = spacing

    return {'data': new_data, 'extent': new_extent,
            'spacing': new_spacing, 'origin': new_origin,
            'small': small}


import scipy

import SimpleITK as sitk
import nibabel as nib
def read_nifty(fname):
    img = sitk.ReadImage(fname)
    spacing = img.GetSpacing()[::-1]
    arr = sitk.GetArrayFromImage(img)
    return arr, spacing



class ACDCImage(object):
    def __init__(self, number, root_dir='B:/Data/ACDC/Training_correct', scale_intensities=True):
        self._number = number
        self._path = root_dir + '/patient{:03d}'.format(number)
        self._img_fname = self._path + '/patient{:03d}_4d.nii.gz'.format(self._number)
        self._img = nib.load(self._img_fname)
        self._scale_intensities = scale_intensities

    def ed(self):
        idx = int(self.info()['ED'])
        im, spacing = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number, idx)))
        gt, _ = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}_gt.nii.gz'.format(self._number, idx)))
        return im, spacing, gt

    def es(self):
        idx = int(self.info()['ES'])
        im, spacing = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number, idx)))
        gt, _ = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}_gt.nii.gz'.format(self._number, idx)))
        return im, spacing, gt

    def voxel_spacing(self):
        return self._img.header.get_zooms()[::-1]

    def image4d(self):
        return self._img.get_data().T

    def shape(self):
        return self._img.header.get_data_shape()[::-1]

    def info(self):
        try:
            self._info
        except AttributeError:
            self._info = dict()
            fname = self._path + '/Info.cfg'
            with open(fname, 'r') as f:
                for l in f:
                    k, v = l.split(':')
                    self._info[k.strip()] = v.strip()
        finally:
            return self._info


import scipy
def apply_2d_zoom(arr4d, spacing):
    vox_size = 1.4  # mm
    zoom = np.array(spacing, float)[1:] / vox_size
    for idx in range(arr4d.shape[0]):
        for jdx in range(arr4d.shape[1]):
            sigma = .25 / zoom
            arr4d[idx, jdx] = scipy.ndimage.gaussian_filter(arr4d[idx, jdx], sigma)
    return scipy.ndimage.interpolation.zoom(arr4d, (1, 1) + tuple(zoom), order=1), np.array(
        (spacing[0], vox_size, vox_size), np.float32)




def rescale_intensities(im, dtype=np.float32):
    min_val, max_val = np.percentile(im, (1, 99))
    im = ((im.astype(dtype) - min_val) / (max_val - min_val)).clip(0, 1)
    return im


from torch.nn.modules.utils import _pair

class RandomMirroring(object):
    def __init__(self, axis, p=0.5, rs=np.random):
        self.axis = axis
        self.p = p
        self.rs = rs

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        rs = self.rs
        if rs.rand() < 0.5:
            image = np.flip(image, axis=self.axis)
            reference = np.flip(reference, axis=self.axis)

        return {'image': image,
                'reference': reference,
                'spacing': spacing}


class RandomPerspective(object):
    def __init__(self, rs=np.random, max_scale=1, target_shape=None):
        self.rs = rs
        self.target_shape = target_shape

    #         self.max_scale = max_scale

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        if self.target_shape != None:
            target_shape = self.target_shape
        else:
            target_shape = image.shape

        #         scale = 1.5**self.rs.uniform(-self.max_scale, self.max_scale, 3)
        Td = np.identity(3, float)
        Td[:2, 2] -= np.divide(image.shape, 2)
        Tp = np.identity(3, float) + self.rs.uniform(-0.002, 0.002, (3, 3))
        Tdi = np.identity(3, float)
        Tdi[:2, 2] += np.divide(target_shape, 2)
        M = Tdi @ Tp @ Td
        if image.ndim == 3:
            image = cv2.warpPerspective(image.transpose(1, 2, 0), M, target_shape, flags=cv2.INTER_LINEAR).transpose(2,                                                                                              1)
        else:
            image = cv2.warpPerspective(image, M, target_shape, flags=cv2.INTER_LINEAR)
        reference = cv2.warpPerspective(reference, M, target_shape, flags=cv2.INTER_NEAREST)
        return {'image': image,
                'reference': reference,
                'spacing': spacing}


class RandomRotation(object):
    def __init__(self, axes, rs=np.random):
        self.axes = axes
        self.rs = rs

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        k = self.rs.randint(0, 4)
        image = np.rot90(image, k, self.axes)
        reference = np.rot90(reference, k, self.axes)

        return {'image': image,
                'reference': reference,
                'spacing': spacing}

class RandomIntensity(object):
    def __init__(self, rs=np.random):
        self.rs = rs
        # self.maximum_g = 1.25
        # self.maximum_gain = 10

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']

        gain = self.rs.uniform(2.5, 7.5)
        cutoff = self.rs.uniform(0.25, 0.75)
        image = (1 / (1 + np.exp(gain * (cutoff - image))))

        return {'image': image,
                'reference': reference,
                'spacing': spacing}


class RandomCropPad(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    TODO: Random sampler should be dependent on a specific random state. Same samples should be shown regardless of network used.
    """

    def __init__(self, output_size, input_padding=None, rs=np.random):
        assert isinstance(output_size, (int, tuple))
        self.rs = rs
        self.output_size = _pair(output_size)
        self.input_padding = input_padding

    def __call__(self, sample):
        rs = self.rs
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        h, w = reference.shape
        new_h, new_w = self.output_size
        out_img = np.zeros(self.output_size, dtype=image.dtype)
        out_ref = np.zeros(self.output_size, dtype=reference.dtype)

        top = 0
        left = 0
        top_pad = 0
        left_pad = 0

        diff_h = h - new_h
        diff_w = w - new_w
        if diff_h < 0:
            top_pad = rs.randint(0, -diff_h)
        else:
            top = rs.randint(0, diff_h)

        if diff_w < 0:
            left_pad = rs.randint(0, -diff_w)
        else:
            left = rs.randint(0, diff_w)

        out_img[top_pad:top_pad + new_h,
        left_pad:left_pad + new_w] = image[top:  top + new_h,
                                     left: left + new_w]
        reference[top_pad:top_pad + new_h,
        left_pad:left_pad + new_w] = reference[top:  top + new_h,
                                     left: left + new_w]

        return {'image': image,
                'reference': reference,
                'spacing': spacing}


class CenterCropPad(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    TODO: Random sampler should be dependent on a specific random state. Same samples should be shown regardless of network used.
    """

    def __init__(self, output_size, input_padding=None):
        assert isinstance(output_size, (int, tuple))
        self.output_size = _pair(output_size)
        self.input_padding = input_padding

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        h, w = reference.shape
        new_h, new_w = self.output_size
        out_img = np.zeros(self.output_size, dtype=image.dtype)
        out_ref = np.zeros(self.output_size, dtype=reference.dtype)

        top = 0
        left = 0
        top_pad = 0
        left_pad = 0

        diff_h = h - new_h
        diff_w = w - new_w
        if diff_h < 0:
            top_pad = -diff_h // 2
        else:
            top = diff_h // 2
            h = new_h

        if diff_w < 0:
            left_pad = -diff_w // 2
        else:
            left = diff_w // 2
            w = new_w

        #         print(top_pad, left_pad, new_h, new_w, top, left, h, w)
        out_img[top_pad:top_pad + h,
        left_pad:left_pad + w] = image[top:  top + h,
                                 left: left + w]
        out_ref[top_pad:top_pad + h,
        left_pad:left_pad + w] = reference[top:  top + h,
                                 left: left + w]

        return {'image': out_img,
                'reference': out_ref,
                'spacing': spacing}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    TODO: Random sampler should be dependent on a specific random state. Same samples should be shown regardless of network used.
    """

    def __init__(self, output_size, input_padding=None, rs=np.random):
        assert isinstance(output_size, (int, tuple))
        self.rs = rs
        self.output_size = _pair(output_size)
        self.input_padding = input_padding

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        h, w = reference.shape
        new_h, new_w = self.output_size

        rs = self.rs

        top = rs.randint(0, h - new_h)
        left = rs.randint(0, w - new_w)

        reference = reference[top:  top + new_h,
                              left: left + new_w]

        if self.input_padding:
            new_h += 2*self.input_padding
            new_w += 2*self.input_padding
        if image.ndim == 3:
            image = image[:, top:  top + new_h,
                              left: left + new_w]
        else:
            image = image[top:  top + new_h,
                          left: left + new_w]

        return {'image': image,
                'reference': reference,
                'spacing': spacing}

class PadInput(object):
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']

        if self.pad > 0:
            if image.ndim == 3:
                image = np.pad(image, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='edge')
            else:
                image = np.pad(image, ((self.pad, self.pad), (self.pad, self.pad)), mode='edge')
        return {'image': image,
                'reference': reference,
                'spacing': spacing}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):#, fixed_nograd=False):
        pass
        # if fixed_nograd:
        #     self.fixed_conversion = lambda arr: torch.tensor(arr, requires_grad=False)
        # else:

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']

        try:
            image = torch.from_numpy(image)
            reference = torch.from_numpy(reference)
        except ValueError:
            image = torch.from_numpy(np.ascontiguousarray(image))
            reference = torch.from_numpy(np.ascontiguousarray(reference))


        return {'image': image[None],
                'reference': reference.long(),
                'spacing': spacing}


from itertools import permutations
from torch.utils.data import Dataset, DataLoader

def acdc_validation_fold(fold, root_dir='/home/bob/data/ACDC/Training_correct'):
    rescale=True

    allpatnumbers = np.arange(1, 101)
    foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)  # ask Jelmer why this division and why not randomly selected.

    validation_nums = allpatnumbers[foldmask == fold]

    for patnum in validation_nums:
        img = ACDCImage(patnum, root_dir=root_dir)
        ed, sp, ed_gt = img.ed()
        es, _, es_gt = img.es()

        if sp[-1] < 1.:  # mm
            print(ed.shape, sp)
            ed = ed[:, ::2, ::2]
            es = es[:, ::2, ::2]
            ed_gt = ed_gt[:, ::2, ::2]
            es_gt = es_gt[:, ::2, ::2]
            sp = (sp[0], sp[1] * 2, sp[2] * 2)
            print(ed.shape, sp)
        if rescale:
            ed = rescale_intensities(ed).astype(np.float32)
            es = rescale_intensities(es).astype(np.float32)


        yield ed, sp, ed_gt, '{:03d}_{}'.format(patnum, 'ED')
        yield es, sp, es_gt, '{:03d}_{}'.format(patnum, 'ES')

def acdc_validation_fold_image4d(fold, root_dir='/home/bob/data/ACDC/Training_correct'):
    rescale=True

    allpatnumbers = np.arange(1, 101)
    foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)  # ask Jelmer why this division and why not randomly selected.

    validation_nums = allpatnumbers[foldmask == fold]

    for patnum in validation_nums:
        img = ACDCImage(patnum, root_dir=root_dir)
        arr = img.image4d()
        sp = img.voxel_spacing()
        if sp[-1] < 1.:  # mm
            print(arr.shape, sp)
            arr = arr[:, :, ::2, ::2]
            sp = (sp[0], sp[1], sp[2] * 2, sp[3] * 2)
            print(arr.shape, sp)
        if rescale:
            arr = rescale_intensities(arr).astype(np.float32)

        yield arr, sp, 'patient{:03d}'.format(patnum)


from scipy.ndimage import morphology
def dilate(reference, rs):#label, iterations=2):
    se = morphology.generate_binary_structure(2, 1)
    new_reference = list()
    for ref_slice in reference:
        iterations = 3
        if iterations > 0:
            # label = rs.choice(np.unique(ref_slice), 1)[0]
            label = 1
            s = ref_slice == label
            dilated = morphology.binary_dilation(s, se, iterations=iterations)
            ref_slice[dilated] = label
        new_reference.append(ref_slice)

    return np.array(new_reference)

def rand_dilate(reference, rs):
    se = morphology.generate_binary_structure(2, 1)

    new_reference = list()
    for ref_slice in reference:
        iterations = rs.randint(0, 4)
        if iterations > 0:
            label = rs.choice(np.unique(ref_slice), 1)[0]
            s = ref_slice == label
            dilated = morphology.binary_dilation(s, se, iterations=iterations)
            ref_slice[dilated] = label
        new_reference.append(ref_slice)

    return np.array(new_reference)

class AddLabelNoise:
    def __init__(self):
        pass

    def __call__(self, sample):
        if sample['augment']:
            sample['reference'] = (sample['reference'] >= 2).astype(np.int8)
        else:
            sample['reference'] = (sample['reference'] == 3).astype(np.int8)
        return sample

class ConvertLabel:
    def __init__(self, src, dst, randdst):
        self.src_label = src
        self.dst_label = dst
        self.randdst_label = randdst
        pass

    def __call__(self, sample):
        if sample['augment']:
            sample['reference'][sample['reference']>=self.src_label] = self.randdst_label
        else:
            sample['reference'][sample['reference']>=self.src_label] = self.dst_label

        return sample

class ClearLabel:
    def __init__(self):
        pass

    def __call__(self, sample):
        if sample['augment']:
            sample['reference'] = np.zeros_like(sample['reference'], dtype=np.int8)
        return sample

class ACDCDataset(Dataset):
    def __init__(self, dataset,
                 fold=0,
                 root_dir='/home/bob/data/ACDC/Training_correct',
                 preprocessing=None,
                 label=-1,
                 transform=None, only_ed_es=False,
                 rescale=True,
                 num_of_random_dilations=0,
                 num_of_random_switch_off=0):
        self._root_dir = root_dir
        self.transform = transform


        allpatnumbers = np.arange(1, 101)
        foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)  # this selections includes all diagnostic labels per fold.

        training_nums, validation_nums = allpatnumbers[foldmask != fold], allpatnumbers[foldmask == fold]
        if dataset == 'training':
            pat_nums = training_nums
        elif dataset == 'validation':
            pat_nums = validation_nums
        elif dataset == 'full':
            pat_nums = np.arange(1, 101)

        images = list()
        references = list()
        spacing = list()
        ids = list()
        allidcs = np.empty((0, 2), dtype=int)

        rs = np.random.RandomState(808)

        do_dilation = rs.choice(pat_nums, num_of_random_dilations, replace=False)
        do_switch_off = rs.choice(pat_nums, num_of_random_switch_off, replace=False)

        for idx, patnum in tqdm(enumerate(pat_nums), desc='Load {} set fold {}'.format(dataset, fold)):
            img = ACDCImage(patnum, root_dir=root_dir)
            ed, sp, ed_gt = img.ed()
            es, _,  es_gt = img.es()

            if label > 0:
                ed_gt = ed_gt == label
                es_gt = es_gt == label

            if sp[-1] < 1.:#mm
                print(ed.shape, sp)
                ed = ed[:, ::2, ::2]
                es = es[:, ::2, ::2]
                ed_gt = ed_gt[:, ::2, ::2]
                es_gt = es_gt[:, ::2, ::2]
                sp = (sp[0], sp[1]*2, sp[2]*2)
                print(ed.shape, sp)

            if rescale:
                ed = rescale_intensities(ed).astype(np.float32)
                es = rescale_intensities(es).astype(np.float32)



            zslices = np.where(ed_gt.any((1, 2)))[0][1:-2]
            ed = ed[zslices]
            ed_gt = ed_gt[zslices]

            zslices = np.where(es_gt.any((1, 2)))[0][1:-2]
            es = es[zslices]
            es_gt = es_gt[zslices]
            # zslices = np.where(ed.any((1, 2)))[0][1:-1]

            if patnum in do_dilation:
                print('dilate on pat', patnum)
                ed_gt = dilate(ed_gt, rs)
                es_gt = dilate(es_gt, rs)

            images.append(ed)
            images.append(es)

            references.append(ed_gt.astype(int))
            references.append(es_gt.astype(int))

            spacing.append(sp)
            spacing.append(sp)

            ids.append('{:03d} ED'.format(patnum))
            ids.append('{:03d} ES'.format(patnum))

            img_idx = (idx * 2)
            allidcs = np.vstack((allidcs, np.vstack((np.ones(len(ed)) * img_idx, np.arange(len(ed)))).T))
            img_idx = (idx * 2) + 1
            allidcs = np.vstack((allidcs, np.vstack((np.ones(len(es)) * img_idx, np.arange(len(es)))).T))

        self._idcs = allidcs.astype(int)
        self._images = images
        self._references = references
        self._spacings = spacing
        self._ids = ids
        self._augment = np.zeros(len(self._idcs), dtype=bool)
        percentage = int(0.4 * len(self._idcs))
        self._augment[:percentage] = True
        np.random.RandomState(808).shuffle(self._augment)

    def __len__(self):
        return len(self._idcs)

    def __getitem__(self, idx):

        img_idx, slice_idx = self._idcs[idx]

        sample = {'image': self._images[img_idx][slice_idx],
                  'reference': self._references[img_idx][slice_idx],
                  'spacing': self._spacings[img_idx],
                  'augment': self._augment[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import morphology
import pandas as pd


def load_sitk(fname):
    img = sitk.ReadImage(str(fname))
    spacing = img.GetSpacing()[::-1]
    nparr = sitk.GetArrayFromImage(img)
    return nparr, spacing


def load_reference(fname):
    ref, _ = load_sitk(fname)
    ref = ref > 0.0
    for idx in range(len(ref)):
        ref[idx] = morphology.binary_fill_holes(ref[idx])
    return ref


def load_sunnybrook(img_fname, ref_src):
    pid, tp = img_fname.with_suffix('').parts[-1].split('_')
    ocontour = load_reference(ref_src / f'{pid}_ocontour_{tp}.mhd')
    icontour = load_reference(ref_src / f'{pid}_icontour_{tp}.mhd')
    p1contour = load_reference(ref_src / f'{pid}_p1contour_{tp}.mhd')
    p2contour = load_reference(ref_src / f'{pid}_p2contour_{tp}.mhd')

    segmentation = ocontour.astype(np.int8) * 2
    segmentation[icontour] = 1

    segmentation[p1contour.astype(bool)] = 3
    segmentation[p2contour.astype(bool)] = 3

    image, spacing = load_sitk(img_fname)
    return image, segmentation, spacing

class SunnybrookDataset(Dataset):
    def __init__(self, dataset,
                 fold=0,
                 # root_dir='/home/bob/TorchIR/temp/ACDC/removed_highres',
                 root_dir=Path('../data/SunnyBrook_cleaned_bob_annotations'),
                 # root_dir=Path('D:/data/SunnyBrook_cleaned_bob_annotations/'),
                 debug=False,
                 preprocessing=None,
                 label=-1,
                 transform=None, only_ed_es=False,
                 rescale=True,
                 label_noise=0,
                 num_of_random_dilations=0,
                 num_of_random_switch_off=0):

        image_src = root_dir / 'image_mhd'
        reference_src = root_dir / 'contour_mhd'

        df = pd.read_excel(root_dir / 'SCD_PatientData_sets.xlsx')

        training_pids = df['OriginalID'][df['Training'].values == 1].values
        validation_pids = df['OriginalID'][df['Validation'].values == 1].values
        test_pids = df['OriginalID'][df['Test'].values == 1].values

        self.transform = transform
        if dataset == 'training':
            pids = list(training_pids) + list(validation_pids)
        elif dataset == 'validation':
            pids = list(test_pids)

        images = list()
        references = list()
        spacing = list()
        ids = list()
        allidcs = np.empty((0, 2), dtype=int)

        for idx, pid in enumerate(pids):
            fname = image_src / (pid + '_ED.mhd')
            ed, ed_gt, sp = load_sunnybrook(fname, reference_src)

            fname = image_src / (pid + '_ES.mhd')
            es, es_gt, sp = load_sunnybrook(fname, reference_src)
            #             print(np.unique(reference))

            if rescale:
                ed = rescale_intensities(ed).astype(np.float32)
                es = rescale_intensities(es).astype(np.float32)

            zslices = np.where((ed_gt==3).any((1, 2)))[0]
            for zs in zslices:
                im, re = ed[zs], ed_gt[zs]
                if not(re == 2).any():
                    print('missing myocardium')
                    continue
                images.append(im)
                references.append(re)
                ids.append(f'{pid}_ED_{zs}')
                spacing.append(sp)

            zslices = np.where((es_gt==3).any((1, 2)))[0]
            for zs in zslices:
                im, re = es[zs], es_gt[zs]
                if not(re == 2).any():
                    print('missing myocardium')
                    continue
                images.append(im)
                references.append(re)
                ids.append(f'{pid}_ES_{zs}')
                spacing.append(sp)


        rs = np.random.RandomState(808)
        self._images = images
        self._references = references
        self._spacings = spacing
        self._ids = ids
        self._augment = np.zeros(len(self._images), dtype=bool)
        num_with_noise = int(label_noise * len(self._images))
        self._augment[:num_with_noise] = True
        np.random.RandomState(808).shuffle(self._augment)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        sample = {'image': self._images[idx],
                  'reference': self._references[idx],
                  'spacing': self._spacings[idx],
                  'augment': self._augment[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

