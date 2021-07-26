"""
Resamples a torch dataset.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from copy import copy

def get_labels_and_class_counts(labels_list):
    '''
    Calculates the counts of all unique classes.
    '''
    labels = np.array(copy(labels_list))
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts

class ImbalancedDataset(Dataset):
    """
    Implements a torchvision dataset, possibly imbalanced.
    """
    def __init__(self, dataset_class, classes, transform, size=None, props=None,
                train=True, download=True, root='./data'):
        """
        Load the dataset and subsample it so that only contains samples from the
        desired classes and whose distribution is given by "props".

        Args:
            dataset_class (torchvision dataset): the base dataset.
            classes (list): labels of the classes to consider.
            transform (torchvision.transforms): to apply to the images.
            props (list of floats): probability distribution of samples
                across the provided digits.
            train (bool, optional): Whether to use the training or the
                testing portion of the dataset. Defaults to True.
            download (bool, optional): whether to download the dataset.
                Defaults to True.
            root (str, optional): where to store the dataset.
                Defaults to './data'.
        """
        if props is None:
            # perfectly balanced
            self.props = [1/len(classes) for _ in range(len(classes))]
        else:
            self.props = props
        assert np.round(sum(self.props),2) == 1.0, "Proportions do not sum to one"

        self.classes = classes
        self.train = train

        self.dataset = dataset_class(
            root=root,
            train=train,
            download=download,
            transform=transform,
            )
        self.size = size if size is not None and size < len(self.dataset) else len(self.dataset)

        # Resample the dataset
        self.dataset = self.imbal_resample()

    def imbal_resample(self):
        """
        Subsample the indices to create an artificially imbalanced dataset.

        Returns:
            torch Dataset: a subsampled dataset.
        """

        aux_labels = self.dataset.train_labels if self.train else self.dataset.test_labels
        labels, cl_counts = get_labels_and_class_counts(aux_labels)
        # Ignore counts for elements which are not in classes.
        cl_counts = cl_counts[self.classes]

        # All samples from the class with the largest prop are used
        largest_cl_prop = max(self.props)
        largest_cl_idx = [p == largest_cl_prop for p in self.props]
        # In case various classes have the largest prop, choose min of their counts.
        largest_cl_count = min(cl_counts[largest_cl_idx])
        # Update size in case props and size are not consistent
        self.size = min(self.size, int(largest_cl_count / largest_cl_prop)) # rule of three

        self.imbal_cl_counts = [int(p * self.size) for p in self.props]

        # Subsample within each class
        class_idx = [np.where(labels == d)[0] for d in self.classes]
        imbal_idx_list = []
        for idx,count in zip(class_idx, self.imbal_cl_counts):
            # sample randomly
            imbal_idx = np.random.choice(idx, size = count, replace = False)
            imbal_idx_list.append(imbal_idx)

        imbal_idx_list = np.hstack(imbal_idx_list)

        return torch.utils.data.Subset(self.dataset, imbal_idx_list)

    def __getitem__(self, index):
        item, labels = copy(self.dataset[index])
        return item, labels

    def __len__(self):
        return len(self.dataset)