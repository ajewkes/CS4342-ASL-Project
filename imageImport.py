from kagglehub import *
from torchvision import transforms
from torchvision import datasets

#training set, 3 k-folds
all = kagglehub.dataset_download("prathumarikeri/american-sign-language-09az")
alphabet = kagglehub.dataset_download("piotrpopis/asl-hands")
numbers = kagglehub.dataset_download("lexset/synthetic-asl-numbers")

#validation set
valid = kagglehub.dataset_download("ayuraj/asl-dataset")
#testing set
test = kagglehub.dataset_download("dorukdemirci/asl-alphabet-dataset")

trainingStandard = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

trainingDataAll = datasets.ImageFolder(all, transform=trainingStandard)
trainingDataAlphabet = datasets.ImageFolder(alphabet, transform=trainingStandard)
trainingDataNums = datasets.ImageFolder(numbers, transform=trainingStandard)
validationSet = datasets.ImageFolder(valid, transform=trainingStandard)
testSet = datasets.ImageFolder(test, transform=trainingStandard)

print(trainingDataAll)