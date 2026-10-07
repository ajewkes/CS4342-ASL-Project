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
    # returns loss, accuracy, per-class accuracy and a confusion matrix
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
    # most common mistakes as (true, predicted, count)
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


def evalEstimate(loader):
    # rough seconds for one validation pass, used until a real one has been timed
    workers = loader.num_workers
    return len(loader.dataset) * (0.0005 if workers else 0.002) + 5 * workers


def train(trainLoader, validLoader, lr=0.001, epoch=10, evalEvery=500, seed=0, path=CHECKPOINT,
          log=print, stop=None, onEval=None, startTime=None):
    # trains with Adam until time runs out or stop is set, keeps the best weights (path=None skips saving)
    # the whole run, loader start-up and final validation included, fits in the time budget
    torch.manual_seed(seed)
    dev = getDevice()
    model = CNN().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossFn = nn.CrossEntropyLoss()

    batch = trainLoader.batch_size
    step, best, bestState = 0, -1.0, None
    history = []
    begin = time.time()
    log(f"Training on {dev} for {epoch} epochs in total (Adam, lr={lr})")
    for epoch in range:
        for x, y in trainLoader:
            model.train()
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = lossFn(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1

    if step % evalEvery == 0:
        vLoss, vAcc, _, _ = evaluate(model, validLoader, dev)
        elapsed = time.time() - begin
        log(f"step {step:5d} epoch {epoch:4.2f} {elapsed:5.0f}s  "
            f"train loss {loss.item():.3f}  val loss {vLoss:.3f}  val acc {vAcc:.3f}")
        history.append({"step": step, "epoch": epoch, "seconds": elapsed,
                        "trainLoss": loss.item(), "valLoss": vLoss, "valAcc": vAcc})
        if onEval:
            onEval(history[-1])
        if vAcc > best:
            best = vAcc
            bestState = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if path:
                save(bestState, path, valAcc=vAcc, step=step, lr=lr, history=history)
    return history


def kfold(k=3, lr=0.001, epoch=10, batchSize=64, datasetPower=imageImport.DATASET_POWER,
          workers=8, log=print, stop=None, onEval=None):
    # k-fold cross-validation with a fresh model per fold, nothing is saved
    scores = []
    folds = imageImport.kfold_loaders(k, batchSize, num_workers=workers, datasetPower=datasetPower)
    for i, (trainLoader, foldLoader) in enumerate(folds):
        if stop is not None and stop.is_set():
            break
        log(f"Fold {i + 1}/{k}")
        # the held-out fold is big, so only check it at the end
        history = train(trainLoader, foldLoader, lr=lr, epoch=epoch, evalEvery=10 ** 9, path=None,
                        log=log, stop=stop, onEval=(lambda e, i=i: onEval({**e, "fold": i})) if onEval else None)
        if stop is not None and stop.is_set():
            log("Stopped, this fold was cut short so its score isn't counted")
            break
        scores.append(history[-1]["valAcc"])
    if not scores:
        return scores
    mean = sum(scores) / len(scores)
    spread = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    log("fold accuracies: " + ", ".join(f"{s:.3f}" for s in scores))
    log(f"mean {mean:.3f} +/- {spread:.3f}")
    return scores


def plot_history(path=CHECKPOINT, out="history.png"):
    # plots the training history saved in the checkpoint
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
