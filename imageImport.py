import os
import torch
from kagglehub import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import ConcatDataset, DataLoader, Subset

# one label space shared by every dataset: 0-9 then A-Z (36 classes)
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

trainingStandard = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# same as trainingStandard without the random rotation, for validation/testing
evalStandard = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class ASLFolder(datasets.ImageFolder):
    # ImageFolder numbers classes alphabetically per dataset, so "A" would get a
    # different index in each one. This pins every class to the shared mapping
    # and skips folders that aren't 0-9/A-Z (del, space, nothing, ...)
    def find_classes(self, directory):
        names = sorted(d.name for d in os.scandir(directory)
                       if d.is_dir() and d.name.upper() in CLASS_TO_IDX)
        return names, {n: CLASS_TO_IDX[n.upper()] for n in names}


def classRoot(root):
    # kaggle downloads often nest the class folders, find the folder holding them
    for dirpath, dirnames, _ in os.walk(root):
        if sum(d.upper() in CLASS_TO_IDX for d in dirnames) >= 10:
            return dirpath
    raise RuntimeError(f"no class folders found under {root}")


def load_data(batch_size=64):
    # downloads happen here, not on import
    #training set, 3 k-folds
    all = kagglehub.dataset_download("prathumarikeri/american-sign-language-09az")
    alphabet = kagglehub.dataset_download("piotrpopis/asl-hands")
    numbers = kagglehub.dataset_download("lexset/synthetic-asl-numbers")

    #validation set
    valid = kagglehub.dataset_download("ayuraj/asl-dataset")
    #testing set
    test = kagglehub.dataset_download("dorukdemirci/asl-alphabet-dataset")

    trainingDataAll = ASLFolder(classRoot(all), transform=trainingStandard)
    trainingDataAlphabet = ASLFolder(classRoot(alphabet), transform=trainingStandard)
    trainingDataNums = ASLFolder(classRoot(numbers), transform=trainingStandard)
    validationSet = ASLFolder(classRoot(valid), transform=evalStandard)
    testSet = ASLFolder(classRoot(test), transform=evalStandard)

    # combine the three training sets and batch everything for the model
    trainingSet = ConcatDataset([trainingDataAll, trainingDataAlphabet, trainingDataNums])
    return (
        DataLoader(trainingSet, batch_size=batch_size, shuffle=True),
        DataLoader(validationSet, batch_size=batch_size),
        DataLoader(testSet, batch_size=batch_size),
    )


def kfold_indices(n, k=3, seed=0):
    # shuffle once (fixed seed so folds are reproducible), deal into k folds,
    # then yield (train indices, held-out indices) with each fold held out once
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    folds = [perm[i::k] for i in range(k)]
    for held in range(k):
        yield [j for f in range(k) if f != held for j in folds[f]], folds[held]


def kfold_loaders(k=3, batch_size=64, seed=0):
    # real k-fold over the combined training data, yields (trainLoader, foldLoader).
    # two copies of the data, same file order: the training folds get the random
    # rotation, the held-out fold doesn't
    paths = [
        kagglehub.dataset_download("prathumarikeri/american-sign-language-09az"),
        kagglehub.dataset_download("piotrpopis/asl-hands"),
        kagglehub.dataset_download("lexset/synthetic-asl-numbers"),
    ]
    augmented = ConcatDataset([ASLFolder(classRoot(p), transform=trainingStandard) for p in paths])
    plain = ConcatDataset([ASLFolder(classRoot(p), transform=evalStandard) for p in paths])

    for trainIdx, foldIdx in kfold_indices(len(augmented), k, seed):
        yield (
            DataLoader(Subset(augmented, trainIdx), batch_size=batch_size, shuffle=True),
            DataLoader(Subset(plain, foldIdx), batch_size=batch_size),
        )


if __name__ == "__main__":
    # downloads everything, run directly to sanity check the datasets
    trainLoader, validLoader, testLoader = load_data()
    print(trainLoader.dataset)
    for name, loader in (("train", trainLoader), ("valid", validLoader), ("test", testLoader)):
        images, labels = next(iter(loader))
        print(name, len(loader.dataset), "images", tuple(images.shape), labels[:8].tolist())
