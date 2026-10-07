import os
import torch
from kagglehub import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import ConcatDataset, DataLoader, Subset

# one label set for all datasets: 0-9 then A-Z
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# asl-hands names letters 0-25 instead of A-Z
HANDS_TO_IDX = {str(i): CLASS_TO_IDX[chr(ord("A") + i)] for i in range(26)}

trainingStandard = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# same as trainingStandard but without rotation
evalStandard = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class ASLFolder(datasets.ImageFolder):
    # gives each class the same index in every dataset, skipping unknown folders
    def __init__(self, root, transform=None, names=None):
        self.names = names or CLASS_TO_IDX
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        found = {}
        for d in os.scandir(directory):
            key = d.name.upper().removesuffix("-SAMPLES")
            if d.is_dir() and key in self.names:
                found[d.name] = self.names[key]
        return sorted(found), found


def classRoot(root, sub):
    # class folders are inside a subfolder of each download
    path = os.path.join(root, sub)
    if not os.path.isdir(path):
        raise RuntimeError(f"{sub} not found under {root}")
    return path


def handsFolders(root, transform):
    # asl-hands is images/<person>/<letter>, so one dataset per person
    images = classRoot(root, "images")
    people = sorted(d.path for d in os.scandir(images) if d.is_dir())
    return ConcatDataset([ASLFolder(p, transform=transform, names=HANDS_TO_IDX) for p in people])


def makeLoader(dataset, batch_size, shuffle, num_workers):
    # worker processes speed up image loading
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, persistent_workers=num_workers > 0)


def load_data(batch_size=64, num_workers=8):
    # downloads happen here, not on import
    #training set, 3 k-folds
    all = kagglehub.dataset_download("prathumarikeri/american-sign-language-09az")
    alphabet = kagglehub.dataset_download("piotrpopis/asl-hands")
    numbers = kagglehub.dataset_download("lexset/synthetic-asl-numbers")

    #validation set
    valid = kagglehub.dataset_download("ayuraj/asl-dataset")
    #testing set
    test = kagglehub.dataset_download("dorukdemirci/asl-alphabet-dataset")

    trainingDataAll = ASLFolder(classRoot(all, "American"), transform=trainingStandard)
    trainingDataAlphabet = handsFolders(alphabet, trainingStandard)
    trainingDataNums = ASLFolder(classRoot(numbers, "Train_Nums"), transform=trainingStandard)
    validationSet = ASLFolder(classRoot(valid, "asl_dataset"), transform=evalStandard)
    testSet = ASLFolder(classRoot(test, "dataset"), transform=evalStandard)

    # combine the training sets and batch everything
    trainingSet = ConcatDataset([trainingDataAll, trainingDataAlphabet, trainingDataNums])
    return (
        makeLoader(trainingSet, batch_size, True, num_workers),
        # small sets, so workers aren't worth it
        makeLoader(validationSet, batch_size, False, 0),
        makeLoader(testSet, batch_size, False, 0),
    )


def trainingSets(transform):
    # the three training datasets
    return ConcatDataset([
        ASLFolder(classRoot(kagglehub.dataset_download("prathumarikeri/american-sign-language-09az"), "American"), transform=transform),
        handsFolders(kagglehub.dataset_download("piotrpopis/asl-hands"), transform),
        ASLFolder(classRoot(kagglehub.dataset_download("lexset/synthetic-asl-numbers"), "Train_Nums"), transform=transform),
    ])


def kfold_indices(n, k=3, seed=0):
    # splits indices into k folds, yielding (train, held-out) for each
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    folds = [perm[i::k] for i in range(k)]
    for held in range(k):
        yield [j for f in range(k) if f != held for j in folds[f]], folds[held]


def kfold_loaders(k=3, batch_size=64, seed=0, num_workers=8):
    # k-fold loaders, the training folds get rotation and the held-out fold doesn't
    augmented = trainingSets(trainingStandard)
    plain = trainingSets(evalStandard)

    for trainIdx, foldIdx in kfold_indices(len(augmented), k, seed):
        yield (
            makeLoader(Subset(augmented, trainIdx), batch_size, True, num_workers),
            makeLoader(Subset(plain, foldIdx), batch_size, False, num_workers),
        )


if __name__ == "__main__":
    # run directly to check the datasets
    trainLoader, validLoader, testLoader = load_data()
    print(trainLoader.dataset)
    for name, loader in (("train", trainLoader), ("valid", validLoader), ("test", testLoader)):
        images, labels = next(iter(loader))
        print(name, len(loader.dataset), "images", tuple(images.shape), labels[:8].tolist())
