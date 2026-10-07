import torch
from torch import nn

NUM_CLASSES = 36  # 0-9 then A-Z
# input images are IMAGE_SIZE x IMAGE_SIZE grayscale, changing it needs a retrain
IMAGE_SIZE = 64


class CNN(nn.Module):
    # [N, 1, 64, 64] in, [N, 36] scores out
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
            block(1, 32),     # 64 -> 32
            block(32, 64),    # 32 -> 16
            block(64, 128),   # 16 -> 8
            block(128, 128),  # 8  -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * (IMAGE_SIZE // 16) ** 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, numClasses),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = CNN()
    out = model(torch.zeros(64, 1, IMAGE_SIZE, IMAGE_SIZE))
    print("output", tuple(out.shape), "parameters", sum(p.numel() for p in model.parameters()))
