import os, numpy as np
from time import time
import cv2, torch
from torch.utils.data import Dataset
from auxiliary.transforms import get_transform
from scipy.spatial.distance import cdist
from auxiliary.train_test_split import source_classes_


def get_ucf101(opt, dtype):

    source_classes = source_classes_[opt.sp]
    target_classes = []
    for i in range(101):
        if i not in source_classes:
            target_classes.append(i)

    folder = '/root/data1/sty/datasets/workplace/UCF101/videos/'
    train_fnames, train_labels = [], []
    test_fnames, test_labels = [], []
    all_class = sorted(os.listdir(str(folder)))

    source = np.array(all_class)[source_classes].tolist()
    target = np.array(all_class)[target_classes].tolist()

    if dtype == 'source':
        labels = source
    if dtype == 'target':
        labels = target
    if dtype == 'all':
        labels = sorted(source + target)

    for label in labels:
        file = os.listdir(os.path.join(str(folder), label))
        num = int(len(file)*0.8)
        train_file = file[:num]
        test_file = file[num:]
        for fname in train_file:
            train_fnames.append(os.path.join(str(folder), label, fname))
            train_labels.append(label)

        for fname in test_file:
            test_fnames.append(os.path.join(str(folder), label, fname))
            test_labels.append(label)

    return train_fnames, train_labels, test_fnames, test_labels, all_class


def get_hmdb(opt, dtype):
    
    source_classes = source_classes_[opt.sp]
    target_classes = []
    for i in range(101):
        if i not in source_classes:
            target_classes.append(i)

    folder = '/root/data1/sty/datasets/workplace/HMDB51/videos/'
    train_fnames, train_labels = [], []
    test_fnames, test_labels = [], []

    classes = sorted(os.listdir(str(folder)))
    all_class = []
    for label in classes:
        all_class.append(label.replace('_', ' '))

    source_index = [index for index in source_classes if index<51]
    target_index = [index for index in target_classes if index<51]

    source = np.array(all_class)[source_index].tolist()
    target = np.array(all_class)[target_index].tolist()

    if dtype == 'source':
        labels = source
    if dtype == 'target':
        labels = target
    if dtype == 'all':
        labels = sorted(source + target)

    for label in labels:
        dir = os.path.join(str(folder), label.replace(' ', '_'))
        if not os.path.isdir(dir): 
            print(dir,' not exists')
        file = sorted(os.listdir(dir))
        num = int(len(file)*0.8)
        train_file = file[:num]
        test_file = file[num:]

        for fname in train_file:
            if fname[-4:] != '.avi':
                continue
            train_fnames.append(os.path.join(str(folder), label.replace(' ', '_'), fname))
            train_labels.append(label)

        for fname in test_file:
            if fname[-4:] != '.avi':
                continue
            test_fnames.append(os.path.join(str(folder), label.replace(' ', '_'), fname))
            test_labels.append(label)

    return train_fnames, train_labels, test_fnames, test_labels, all_class


def load_clips_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    if not os.path.exists(fname):
        print('Missing: '+fname)
        return []
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count #min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0
    while count < selection[-1]+clip_len:
        retained, frame = capture.read()
        if count not in selection:
            count += 1
            continue
        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    capture.release()
    frames = np.stack(frames)
    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames


class VideoDataset(Dataset):

    def __init__(self, fnames, labels, class_embed, classes, load_clips=load_clips_tsn,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False, evaluation_only=False):
        self.data = fnames
        self.labels = labels
        self.class_embed = class_embed
        self.class_name = classes

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size  # 112
        self.is_validation = is_validation

        self.transform = get_transform(self.is_validation, crop_size)
        self.loadvideo = load_clips
        self.classname2index = {name: index for index, name in enumerate(self.class_name)}


    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        classid = self.classname2index[label]

        buffer = self.loadvideo(sample, self.clip_len, self.n_clips, self.is_validation)
        if len(buffer) == 0:
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], -1
        s = buffer.shape
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], 0)
        buffer = self.transform(buffer)
        buffer = buffer.reshape(3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)
        return buffer, classid, self.class_embed[classid], sample

    def __len__(self):
        return len(self.data)

