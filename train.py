import time
import torch
from torch import nn
from PIL import Image

import imageImport
from model import CNN, NUM_CLASSES

CHECKPOINT = "model.pt"


def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, dev):
    # returns (loss, accuracy, per-class accuracy as a list with None for classes
    # not in the data, confusion matrix [true class, predicted class])
    model.eval()
    lossFn = nn.CrossEntropyLoss(reduction="sum")
    loss, total, correct = 0.0, 0, 0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            pred = out.argmax(1)
            loss += lossFn(out, y).item()
            total += len(y)
            correct += (pred == y).sum().item()
            both = (y * NUM_CLASSES + pred).cpu()
            confusion += torch.bincount(both, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    seen = confusion.sum(dim=1)
    perClass = [confusion[i, i].item() / seen[i].item() if seen[i] > 0 else None for i in range(NUM_CLASSES)]
    return loss / total, correct / total, perClass, confusion


def top_confusions(confusion, n=5):
    # most common mistakes as (true char, predicted char, count)
    wrong = confusion.clone()
    wrong.fill_diagonal_(0)
    counts, flat = wrong.flatten().topk(n)
    return [(imageImport.CLASSES[i // NUM_CLASSES], imageImport.CLASSES[i % NUM_CLASSES], c)
            for c, i in zip(counts.tolist(), flat.tolist()) if c > 0]


def save(state, path, **meta):
    torch.save({"model": state, "classes": imageImport.CLASSES, **meta}, path)


def load(path=CHECKPOINT, dev=None):
    dev = dev or getDevice()
    checkpoint = torch.load(path, map_location=dev)
    model = CNN().to(dev)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def train(trainLoader, validLoader, lr=0.001, minutes=10, evalEvery=500, seed=0, path=CHECKPOINT):
    # Adam, runs until the time budget is up (not a fixed number of epochs),
    # checking validation every evalEvery steps and keeping the best weights.
    # path=None skips saving (used by k-fold). Returns the history, one entry
    # per validation check: {step, epoch, seconds, trainLoss, valLoss, valAcc}
    torch.manual_seed(seed)
    dev = getDevice()
    model = CNN().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossFn = nn.CrossEntropyLoss()

    batch = trainLoader.batch_size
    step, best, bestState = 0, -1.0, None
    history = []
    start = deadline = None
    print(f"Training on {dev} for up to {minutes} minutes (Adam, lr={lr})")
    while True:
        for x, y in trainLoader:
            if start is None:
                # the clock starts once the first batch is ready, loader worker
                # start-up (tens of seconds on Windows) isn't training time
                start = time.time()
                deadline = start + minutes * 60
            model.train()
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = lossFn(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1

            timeUp = time.time() >= deadline
            if step % evalEvery == 0 or timeUp:
                vLoss, vAcc, _, _ = evaluate(model, validLoader, dev)
                epochs = step * batch / len(trainLoader.dataset)
                elapsed = time.time() - start
                print(f"step {step:5d} epoch {epochs:4.2f} {elapsed:5.0f}s  "
                      f"train loss {loss.item():.3f}  val loss {vLoss:.3f}  val acc {vAcc:.3f}")
                history.append({"step": step, "epoch": epochs, "seconds": elapsed,
                                "trainLoss": loss.item(), "valLoss": vLoss, "valAcc": vAcc})
                if vAcc > best:
                    best = vAcc
                    bestState = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if path:
                        save(bestState, path, valAcc=vAcc, step=step, lr=lr, history=history)
            if timeUp:
                if path:
                    # rewrite so the saved history covers the whole run, not just up to the best step
                    save(bestState, path, valAcc=best, step=history[-1]["step"], lr=lr, history=history)
                    print(f"Time budget reached. Best validation accuracy {best:.3f}, saved to {path}")
                else:
                    print(f"Time budget reached. Final validation accuracy {history[-1]['valAcc']:.3f}")
                return history


def kfold(k=3, lr=0.001, minutes=10):
    # k-fold cross-validation over the combined training data: a fresh model per
    # fold, scored on its held-out fold once at the end. This estimates how well
    # the setup generalises, nothing is saved. minutes is per fold
    scores = []
    for i, (trainLoader, foldLoader) in enumerate(imageImport.kfold_loaders(k)):
        print(f"Fold {i + 1}/{k}")
        # the held-out fold is large, so only check it once when time is up
        history = train(trainLoader, foldLoader, lr=lr, minutes=minutes, evalEvery=10 ** 9, path=None)
        scores.append(history[-1]["valAcc"])
    mean = sum(scores) / k
    spread = (sum((s - mean) ** 2 for s in scores) / k) ** 0.5
    print("fold accuracies: " + ", ".join(f"{s:.3f}" for s in scores))
    print(f"mean {mean:.3f} +/- {spread:.3f}")
    return scores


def plot_history(path=CHECKPOINT, out="history.png"):
    # learning curves from the history stored in the checkpoint
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = torch.load(path, map_location="cpu").get("history")
    if not history:
        raise RuntimeError(f"{path} has no training history, train again with -tr")
    steps = [h["step"] for h in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(steps, [h["trainLoss"] for h in history], label="train (last batch)")
    left.plot(steps, [h["valLoss"] for h in history], label="validation")
    left.set_xlabel("step")
    left.set_ylabel("loss")
    left.legend()
    right.plot(steps, [h["valAcc"] for h in history])
    right.set_xlabel("step")
    right.set_ylabel("validation accuracy")
    fig.tight_layout()
    fig.savefig(out)
    return out


def predict(model, imagePath, dev=None):
    # one image -> (character, confidence)
    dev = dev or getDevice()
    image = Image.open(imagePath).convert("RGB")
    x = imageImport.evalStandard(image).unsqueeze(0).to(dev)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = probs.argmax().item()
    return imageImport.CLASSES[idx], probs[idx].item()
