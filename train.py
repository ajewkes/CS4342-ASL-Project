import time
import torch
from torch import nn
from PIL import Image

import imageImport
from model import CNN, NUM_CLASSES, IMAGE_SIZE

CHECKPOINT = "model.pt"
EPOCHS = 5

# a saved model has to beat twice the chance rate on validation before it is used for predictions
MIN_VAL_ACC = 2 / NUM_CLASSES


def getDevice():
    # an NVIDIA GPU, else an Apple-silicon GPU, else the CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    torch.save({"model": state, "classes": imageImport.CLASSES, "imageSize": IMAGE_SIZE, **meta}, path)


def readCheckpoint(path=CHECKPOINT):
    return torch.load(path, map_location="cpu")


def wrongSize(saved):
    # checkpoints from before the size was stored are 28x28
    size = saved.get("imageSize", 28)
    if size != IMAGE_SIZE:
        return f"The saved model was trained on {size}x{size} images but the model now uses {IMAGE_SIZE}x{IMAGE_SIZE}. Train it again."
    return None


def notUsable(saved):
    # why a checkpoint can't be used for predictions yet, or None when it can
    # the last save of a run marks it finished, older checkpoints have no mark and are trusted
    if wrongSize(saved):
        return wrongSize(saved)
    if not saved.get("finished", True):
        return ("The saved model is from a training run that hasn't finished (it is still running or was cut off). "
                "Predictions open once a run finishes.")
    valAcc = saved.get("valAcc")
    if valAcc is not None and valAcc < MIN_VAL_ACC:
        return (f"The saved model scored {valAcc:.1%} on validation, no better than guessing "
                f"({1 / NUM_CLASSES:.1%} for {NUM_CLASSES} classes). Train it for longer.")
    return None


def fromCheckpoint(saved, dev=None):
    if wrongSize(saved):
        raise RuntimeError(wrongSize(saved))
    dev = dev or getDevice()
    model = CNN().to(dev)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def load(path=CHECKPOINT, dev=None):
    return fromCheckpoint(readCheckpoint(path), dev)


def train(trainLoader, validLoader, lr=0.001, epochs=EPOCHS, evalEvery=500, seed=0, path=CHECKPOINT,
          log=print, stop=None, onEval=None, onProgress=None):
    # trains with Adam for a number of epochs or until stop is set, keeps the best weights (path=None skips saving)
    # validates every evalEvery steps and once at the end, onProgress(step, total) reports how far along it is
    torch.manual_seed(seed)
    dev = getDevice()
    model = CNN().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossFn = nn.CrossEntropyLoss()

    perEpoch = len(trainLoader)
    total = epochs * perEpoch
    step, best, bestState = 0, -1.0, None
    history = []
    begin = time.time()
    log(f"Training on {dev} for {epochs} epochs of {perEpoch} steps (Adam, lr={lr})")
    if onProgress:
        onProgress(0, total)
    stopped = False
    while not stopped and step < total:
        for x, y in trainLoader:
            model.train()
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = lossFn(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1

            stopped = stop is not None and stop.is_set()
            last = stopped or step == total
            if onProgress and (step % 10 == 0 or last):
                onProgress(step, total)
            if step % evalEvery == 0 or last:
                vLoss, vAcc, _, _ = evaluate(model, validLoader, dev)
                elapsed = time.time() - begin
                log(f"step {step:5d} epoch {step / perEpoch:4.2f} {elapsed:5.0f}s  "
                    f"train loss {loss.item():.3f}  val loss {vLoss:.3f}  val acc {vAcc:.3f}")
                history.append({"step": step, "epoch": step / perEpoch, "seconds": elapsed,
                                "trainLoss": loss.item(), "valLoss": vLoss, "valAcc": vAcc})
                if onEval:
                    onEval(history[-1])
                if vAcc > best:
                    best = vAcc
                    bestState = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if path:
                        save(bestState, path, valAcc=vAcc, step=step, lr=lr, epochs=epochs, history=history, finished=False)
            if last:
                break

    reason = "Stopped" if stopped else f"Finished {epochs} epochs"
    if path:
        # save again so the history covers the whole run, and mark the model ready for predictions
        save(bestState, path, valAcc=best, step=history[-1]["step"], lr=lr, epochs=epochs, history=history, finished=True)
        log(f"{reason}. Best validation accuracy {best:.3f}, saved to {path}")
    else:
        log(f"{reason}. Final validation accuracy {history[-1]['valAcc']:.3f}")
    return history


def kfold(k=3, lr=0.001, epochs=EPOCHS, batchSize=64, datasetPower=imageImport.DATASET_POWER,
          workers=8, log=print, stop=None, onEval=None, onProgress=None):
    # k-fold cross-validation with a fresh model per fold, nothing is saved
    scores = []
    folds = imageImport.kfold_loaders(k, batchSize, num_workers=workers, datasetPower=datasetPower)
    for i, (trainLoader, foldLoader) in enumerate(folds):
        if stop is not None and stop.is_set():
            break
        log(f"Fold {i + 1}/{k}")
        # the held-out fold is big, so only check it at the end
        history = train(trainLoader, foldLoader, lr=lr, epochs=epochs, evalEvery=10 ** 9, path=None, log=log, stop=stop,
                        onEval=(lambda e, i=i: onEval({**e, "fold": i})) if onEval else None,
                        onProgress=(lambda s, t, i=i: onProgress(s, t, i)) if onProgress else None)
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


def flatten(image):
    # RGB image, with any transparent background filled white rather than black
    image = image.convert("RGBA") if image.mode in ("P", "LA") or "transparency" in image.info else image
    if image.mode == "RGBA":
        white = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(white, image).convert("RGB")
    return image.convert("RGB")


def predictImage(model, image, dev=None, top=3):
    # PIL image -> [(character, confidence), ...] best first, look-alike pairs are joined as "O / 0"
    dev = dev or next(model.parameters()).device
    x = imageImport.evalStandard(flatten(image)).unsqueeze(0).to(dev)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    scores = dict(zip(imageImport.CLASSES, probs))
    for a, b in imageImport.LOOKALIKES:
        scores[f"{a} / {b}"] = scores.pop(a) + scores.pop(b)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top]


def lookalikeAccuracy(confusion):
    # accuracy when mixing up a look-alike pair counts as correct
    right = confusion.diag().sum().item()
    for a, b in imageImport.LOOKALIKES:
        i, j = imageImport.CLASS_TO_IDX[a], imageImport.CLASS_TO_IDX[b]
        right += confusion[i, j].item() + confusion[j, i].item()
    return right / confusion.sum().item()


def predict(model, imagePath, dev=None):
    # one image file -> (character, confidence)
    return predictImage(model, Image.open(imagePath), dev, top=1)[0]
