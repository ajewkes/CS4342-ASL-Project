import torch
from torch import nn

NUM_CLASSES = 36  # 0-9 then A-Z, matches imageImport.CLASSES


class CNN(nn.Module):
    # [N, 1, 28, 28] -> [N, 36] raw scores (no softmax, CrossEntropyLoss applies it)
    def __init__(self, numClasses=NUM_CLASSES, dropout=0.4):
        super().__init__()

        def block(inCh, outCh):
            return nn.Sequential(
                nn.Conv2d(inCh, outCh, kernel_size=3, padding=1),
                nn.BatchNorm2d(outCh),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),     # 28 -> 14
            block(32, 64),    # 14 -> 7
            block(64, 128),   # 7  -> 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, numClasses),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = CNN()
    out = model(torch.zeros(64, 1, 28, 28))
    print("output", tuple(out.shape), "parameters", sum(p.numel() for p in model.parameters()))
