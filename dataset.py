import copy
import os
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np


def load_feature_from_index(video_feat_path, frame_index):
    feat_path = os.path.join(video_feat_path, 'image_{:06d}.npy'.format(frame_index))

    if os.path.exists(feat_path):
        return np.load(feat_path)
    else:
        return None


def load_features_from_index(video_feat_path, frame_indices):
    """
        loads and returns given frame indices as a list of image objects.
    """
    video = []

    for frame_index in frame_indices:
        video.append(load_feature_from_index(video_feat_path, frame_index))

    stack = np.row_stack(video)

    return stack


class FeatureDataset(data.Dataset):
    """
        Given a video, stack its i3d feature vectors in columns.

        Attributes:
            annotation_video_mapping (dict): Dictionary mapping each annotation file path to a video folder path. Both
            paths must be absolute.
            class2idx (dict): mapping from str class names to integer class indices.
            sampling_step (int): Number of steps to sample a frame from given frames.
            num_classes (int): Number of class labels.
            temporal_transform (torchvision.transforms.Compose): Compose object which contains several transforms
            to be applied sequentially. Please check src.transforms.temporal_transforms.py.
    """

    def __init__(self, annotation_video_mapping, class2idx, sampling_step, num_classes, temporal_transform=None):
        """
            The constructor for I3dFeatureDataset class.

            :param dict annotation_video_mapping: Dictionary mapping each annotation file path to a video folder path.
            Both paths should be absolute.
            :param dict class2idx: mapping from str class names to integer class indices.
            :param int sampling_step: Number of steps to sample a frame from given frames.
            :param int num_classes: Number of class labels.
            :param torchvision.transforms.Compose temporal_transform: Compose object which contains several transforms
            to be applied sequentially. Please check src.transforms.temporal_transforms.py.
        """
        # some consistency checks
        if not isinstance(annotation_video_mapping, dict):
            raise ValueError('annotation_video_mapping {} should be a dict value'.format(annotation_video_mapping))

        self.annotation_video_mapping = annotation_video_mapping

        if not isinstance(class2idx, dict):
            raise ValueError('class2idx should be a dict of str to int values')

        self.class2idx = class2idx

        self.sampling_step = sampling_step

        self.temporal_transform = temporal_transform

        self.num_classes = num_classes

        self.data = self._make_dataset()

    def gen_data(self):

        for annot_filepath, video_dirpath in tqdm(self.annotation_video_mapping.items()):

            # read annotation file as a dataframe
            df = pd.read_csv(annot_filepath)

            # optionally downsample video along time axis
            if self.sampling_step > 1:
                df = df.iloc[0:-1:self.sampling_step, :]

            # filenames are in image_{:06d}.npy format
            # extract existing frame indices from file names
            frame_indices = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(video_dirpath) if
                             f.endswith('.npy')]

            yield df.loc[df.frame.isin(frame_indices) & df.phase.isin(self.class2idx.keys())], video_dirpath

    def _make_dataset(self):
        """
            Construct dataset of samples.

            The dictionary "annotation_video_mapping" maps each full path to an annotation file to its respective
            video directory path. Both paths are supposed to be absolute paths. In each video directory path, we expect
            video frames should have been extracted and present.

            Each annotation file is supposed to be in a csv format which contains a frame index denoted by "Frame"
            column name and a class label (phase) denoted by "Phase" column name at each of its row.

            :return: list of samples (dataset)
        """
        dataset = []

        for df, video_dirpath in self.gen_data():
            sample_init = {
                'video': video_dirpath
            }

            labels = [self.class2idx[label] for label in df.phase]

            sample = copy.deepcopy(sample_init)

            sample['frame_indices'] = list(df.frame)

            sample['label'] = labels

            dataset.append(sample)

        return dataset

    def __getitem__(self, index):
        """
            Loads frames using the subclip indices and their respective target, and returns them as a tuple.

            :param int index: index of the returned sample
            :return: tuple (sample, target)
        """
        video_path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        # feature vectors stacked in rows -> [L, 1024]
        clip = load_features_from_index(video_path, frame_indices)

        clip = torch.tensor(clip)

        target = self.data[index]['label']

        target = torch.LongTensor(target)

        return clip, target, self.num_classes

    def __len__(self):
        return len(self.data)


class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        labels = [x[1] for x in sorted_batch]

        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

        # the third element in the batch tuple is "num_classes"
        num_classes = batch[0][2]

        # here we define a mask to mark the padded entries as "zero"
        mask = torch.zeros(len(batch), 1, labels_padded.shape[1], dtype=torch.float)

        mask[:, 0, :] = (labels_padded != -1)

        mask = mask.repeat(1, num_classes, 1)

        # feature vectors stacked in columns -> [batch_size, n_channels, L]
        sequences_padded = sequences_padded.permute(0, 2, 1)

        return sequences_padded, labels_padded, mask
