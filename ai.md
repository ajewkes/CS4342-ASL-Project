# ai.md

Context for LLMs working in this repository. `README.md` is the short human version; this file holds the details and the non-obvious behaviour.

## What this is

A CS4342 (machine learning) course project. It classifies images of American Sign Language hand signs into 36 classes (`0-9`, `A-Z`) with a small PyTorch CNN. There are two front ends over the same training code: a CLI (`main.py`) and a localhost web UI (`server.py` + `ui/index.html`).

No tests, no linter config, no build step, no package structure. All Python modules sit flat in the repo root and import each other by name, so everything must be run with the repo root as the working directory.

## Layout

```
imageImport.py    data: label mapping, transforms, datasets, samplers, loaders, k-fold splits
model.py          CNN definition, NUM_CLASSES
train.py          train loop, evaluate, kfold, checkpoint save/load, plot_history, predict
main.py           CLI (argv flags), prints results
server.py         stdlib HTTP server, runs one training job in a background thread
ui/index.html     single-file UI (inline CSS and JS, no dependencies, no build)
datasets.md       two dataset links, incomplete (lists 2 of the 5 datasets)
requirements.txt  torch, torchvision, kagglehub, pillow, matplotlib (unpinned)
```

Git-ignored outputs: `model.pt` (checkpoint), `history.png` (plot), `__pycache__/`.

Import graph: `main.py` and `server.py` -> `train.py` -> `imageImport.py`, `model.py`. `imageImport.py` and `model.py` do not import each other.

## Running

```
pip install -r requirements.txt
python main.py -tr | -m | -v | -t | -k | -p | -r <image>
python server.py [port]        # default 8000, then open http://localhost:8000
python model.py                # sanity check: output shape and parameter count
python imageImport.py          # sanity check: downloads data, prints one batch per split
```

Python 3.10+ is required (`main.py` uses `match`). The development machine is Windows 11 with Python 3.13 and a CPU-only torch build. `train.getDevice()` picks CUDA when it is available.

## Data pipeline (`imageImport.py`)

### Labels

`CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"`. A class index is the position in that string: digits are 0-9, `A` is 10, `Z` is 35. `CLASS_TO_IDX` is the reverse map. `model.NUM_CLASSES = 36` is defined separately and must stay in sync with `len(CLASSES)`.

`HANDS_TO_IDX` exists because the `asl-hands` dataset names its letter folders `"0"`..`"25"` instead of `A`..`Z`. It maps `"0"` -> index of `A`, and so on. Do not read those folders with the default mapping; they would be mislabelled as digits.

### Datasets

All come from `kagglehub.dataset_download`, which caches in the user's kagglehub cache directory, so only the first call downloads.

| Kaggle id | Role | Class folders under | Notes |
| --- | --- | --- | --- |
| `prathumarikeri/american-sign-language-09az` | train | `American/` | digits and letters |
| `piotrpopis/asl-hands` | train | `images/<person>/<0..25>/` | letters only, one `ASLFolder` per person, uses `HANDS_TO_IDX` |
| `lexset/synthetic-asl-numbers` | train | `Train_Nums/` | digits only, synthetic |
| `ayuraj/asl-dataset` | validation | `asl_dataset/` | |
| `dorukdemirci/asl-alphabet-dataset` | test | `dataset/` | |

`classRoot(root, sub)` joins the download path and that subfolder and raises `RuntimeError` when it is missing.

`ASLFolder` subclasses `torchvision.datasets.ImageFolder` and overrides `find_classes` so that every dataset shares one label space. A folder name is upper-cased and a trailing `-SAMPLES` is stripped before lookup. Folders whose name is not a known class (for example `del`, `space`, `nothing`) are silently skipped. A dataset can therefore cover only some of the 36 classes.

`imageImport.py` does `from kagglehub import *` and then calls `kagglehub.dataset_download`. This works because the star import happens to bind the name `kagglehub`. It is fragile; `import kagglehub` would be the robust form.

### Transforms

Both produce a `[1, 28, 28]` float tensor normalised to roughly `[-1, 1]`:

- `trainingStandard`: Grayscale(1) -> Resize(28) -> CenterCrop(28) -> RandomRotation(10) -> ToTensor -> Normalize(0.5, 0.5)
- `evalStandard`: the same without the rotation

`Resize(28)` scales the shorter side to 28 and `CenterCrop(28)` cuts the middle square, so non-square images lose their edges. `train.predict` uses `evalStandard`. Any change to the input size or channels has to be made in both transforms and in `model.py` (the first conv's input channels and the `128 * 3 * 3` flatten size).

### Dataset weighting

The three training datasets differ a lot in size. `datasetWeights(trainingSet, power)` gives each dataset a share of the drawn samples proportional to `size ** power`, spread evenly over its samples. `power = 1` is natural sampling, `power = 0` gives each dataset an equal share, and the default `DATASET_POWER = 0.5` is in between. The result feeds a `WeightedRandomSampler(replacement=True)` with `num_samples = len(trainingSet)`.

Consequences:

- An "epoch" is `len(trainingSet)` draws with replacement, not one pass over every image.
- Weighting is per dataset, not per class. Classes are not balanced.
- `datasetWeights` reads `trainingSet.datasets`, so it needs a `ConcatDataset` whose top-level children are the three datasets. The `asl-hands` child is itself a `ConcatDataset` of people and counts as one dataset.

### Loaders

`load_data(batch_size=64, num_workers=8, datasetPower=0.5)` returns `(trainLoader, validLoader, testLoader)`. The train loader uses the weighted sampler and worker processes. The validation and test loaders always use `num_workers=0` and no shuffling. Every call downloads or resolves all five datasets and scans their folders, even when the caller wants only one split.

`makeLoader` sets `persistent_workers` whenever there are workers and turns `shuffle` off when a sampler is given (PyTorch forbids both).

`trainingSets(transform)` builds the same three-dataset `ConcatDataset` as `load_data`. The dataset ids and subfolders are duplicated between the two functions; change both.

### K-fold

`kfold_indices(n, k, seed)` makes a seeded permutation and splits it into `k` interleaved folds, yielding `(trainIdx, heldIdx)` per fold. The split is over individual images of the combined training data. It is not grouped by person or by dataset, so images of one person can land on both sides.

`kfold_loaders` is a generator. It builds the training data twice, once with `trainingStandard` and once with `evalStandard`, so the same index gives an augmented image for training and a plain one for the held-out fold. Training folds get the weighted sampler (`weights[trainIdx]`); the held-out fold does not.

## Model (`model.py`)

`CNN(numClasses=36, dropout=0.4)`, about 245k parameters. Input `[N, 1, 28, 28]`, output `[N, 36]` raw scores (no softmax).

```
features:   3 x [Conv2d 3x3 pad 1 -> BatchNorm2d -> ReLU -> MaxPool2d(2)]
            channels 1 -> 32 -> 64 -> 128, spatial 28 -> 14 -> 7 -> 3
classifier: Flatten -> Dropout -> Linear(1152, 128) -> ReLU -> Dropout -> Linear(128, 36)
```

## Training (`train.py`)

### `train(trainLoader, validLoader, lr, minutes, evalEvery, seed, path, log, stop, onEval, startTime)`

Adam plus cross-entropy, a fresh `CNN` every call. There is no resume, no LR schedule, and no weight decay.

- **Time budget, not epochs.** The loop runs until `begin + minutes * 60`. `begin` is `startTime` when given, otherwise the moment `train` is called. The server passes the job start time, so dataset loading counts against the budget in UI runs.
- **Room for the final check.** The loop stops at `deadline - evalCost` so that the last validation pass still fits. `evalCost` starts as a guess from `evalEstimate(validLoader)` and is replaced by the measured duration of each real pass.
- **Validation** runs every `evalEvery` steps and once more at the end. Each pass appends `{"step", "epoch", "seconds", "trainLoss", "valLoss", "valAcc"}` to `history` and calls `onEval(entry)`.
- **`trainLoss` is the loss of the last batch only**, not an average. It is noisy by design.
- **Best weights.** When `valAcc` improves, the state dict is copied to CPU and saved to `path`. At the end it is saved again so the stored `history` covers the whole run. The weights in the file are the best ones, not the last ones.
- **`path=None`** disables saving (used by k-fold).
- **`stop`** is a `threading.Event`. When set, the loop does one last validation pass and returns as if time had run out.
- **`log`** is a callable taking a string. It defaults to `print`; the server passes its own buffer writer.
- Returns `history`.

The seed is fixed (`torch.manual_seed(0)`), but worker processes and the sampler make runs only roughly repeatable.

### Checkpoint format

`model.pt` is `torch.save` of a dict:

```
{"model": state_dict, "classes": CLASSES, "valAcc": float, "step": int, "lr": float, "history": [entry, ...]}
```

`CHECKPOINT = "model.pt"` is a relative path, resolved against the current working directory. `load(path, dev)` rebuilds `CNN()` with default arguments, loads the state dict, and puts the model in eval mode. The stored `classes` string is not checked on load. Changing the architecture or the class list invalidates existing checkpoints.

### Other functions

- `evaluate(model, loader, dev)` -> `(meanLoss, accuracy, perClass, confusion)`. `confusion[true, pred]` is a 36x36 count matrix. `perClass[i]` is `None` when class `i` never appears in the loader, and callers must handle that. It divides by the sample count, so an empty loader raises.
- `top_confusions(confusion, n=5)` -> list of `(trueChar, predictedChar, count)` for the largest off-diagonal cells.
- `kfold(k, lr, minutes, batchSize, datasetPower, workers, log, stop, onEval)` trains a fresh model per fold with `evalEvery=10**9`, so each fold is validated exactly once, at the end. **`minutes` is per fold**; a whole run takes about `k * minutes`. Nothing is saved. It returns the per-fold accuracies and logs their mean and population standard deviation. A fold cut short by `stop` is not counted. Each `onEval` entry gets a `"fold"` key (0-based).
- `plot_history(path, out="history.png")` reads `history` from the checkpoint and plots loss and validation accuracy against step. matplotlib is imported inside the function with the `Agg` backend, so nothing else needs it.
- `predict(model, imagePath, dev)` -> `(character, confidence)`. Opens the image as RGB, applies `evalStandard`, and returns the top softmax class and its probability.

## CLI (`main.py`)

`python main.py <flag> [path]`. The dispatcher is `start_learning(cmd, filepath)`.

| Flag | Behaviour |
| --- | --- |
| `-tr` | `load_data()` then `train.train` with `lr=0.001`, `minutes=10` |
| `-m` | prompts on stdin for `L` or `M`, then for a number, overrides that one value, then trains |
| `-v` / `-t` | loads `model.pt`, evaluates on validation / test, prints loss, accuracy, per-class accuracy, top confusions |
| `-k` | `train.kfold(k=3)` with the default lr and minutes |
| `-p` | `train.plot_history` -> `history.png` |
| `-r <image>` | checks the file exists and is an image (`PIL.Image.verify`), then `train.predict` |

Details:

- Hyperparameters are local variables inside `start_learning`. They are not persisted, and `-m` can change only one of them per run.
- Batch size, dataset weighting, eval interval, fold count, and worker count can be set only through the UI. The CLI always uses the defaults.
- The flag is lower-cased for dispatch, but `__main__()` compares `sys.argv[1] == "-r"` case-sensitively when deciding whether to pass the file path. `-R <image>` therefore fails with "no picture given".
- `safe_read(n)` truncates stdin input to `n` characters and lower-cases it. Numbers longer than 8 characters are cut.
- Errors print a message and call `exit(1)`.
- The entry function is literally named `__main__()`.

## Web UI (`server.py`, `ui/index.html`)

### Server

Standard library only: `ThreadingHTTPServer` bound to `127.0.0.1`. There is no framework, and none should be added without a reason.

State is module-level and guarded by one `RLock`:

- `job`: `state`, `params`, `log`, `history`, `error`, `started`, `finished`
- `result`: `scores` (k-fold fold accuracies)
- `stopEvent`: the `threading.Event` handed to `train`

Job states: `idle` -> `running` -> (`stopping` ->) `done` | `stopped` | `error`. One job runs at a time, in a daemon thread started by `start(params)`. State lives in memory only and is lost when the server restarts.

| Method and path | Response |
| --- | --- |
| `GET /` or `/index.html` | the UI page, read from disk on every request |
| `GET /api/config` | `{"defaults": DEFAULTS, "limits": LIMITS}` |
| `GET /api/status` | `state`, `params`, `error`, `elapsed`, `budgetSeconds`, `log` (last 200 lines), `history`, `scores` |
| `GET /api/checkpoint` | `{"exists": false}` or `exists`, `valAcc`, `step`, `lr`, `modified` |
| `POST /api/train` | body is the params object; 202 with status, 400 on bad input, 409 when a job is already running |
| `POST /api/stop` | 202 with status, 409 when nothing is running |

Parameters (`DEFAULTS`, `LIMITS`, `WHOLE` at the top of `server.py`):

| Key | Default | Range | Whole number |
| --- | --- | --- | --- |
| `mode` | `"train"` | `"train"` or `"kfold"` | |
| `lr` | 0.001 | 1e-6 to 1 | |
| `minutes` | 10 | 0.1 to 600 | |
| `batchSize` | 64 | 8 to 512 | yes |
| `datasetPower` | 0.5 | 0 to 1 | |
| `evalEvery` | 500 | 10 to 5000 | yes |
| `k` | 3 | 2 to 10 | yes |
| `workers` | 8 | 0 to 16 | yes |

`parse(body)` validates the body and fills in defaults. Unknown keys are ignored. `evalEvery` is ignored in k-fold mode and `k` is ignored in train mode. `budgetSeconds` is `minutes * 60`, times `k` in k-fold mode.

Request guards, all deliberate:

- The `Host` header must be `localhost` or `127.0.0.1`, otherwise 403. This blocks DNS rebinding.
- POST bodies must be `application/json`, otherwise 415. This forces a CORS preflight, so other websites cannot start a job from the user's browser.
- At most 10000 bytes of body are read.

`log()` keeps the last 500 lines. `checkpointInfo()` tolerates a half-written `model.pt` by returning only `{"exists": true}`. An exception in the job thread becomes `state="error"` with `error` set to `"Type: message"`, and is also appended to the log.

### Page

One HTML file with inline CSS and vanilla JS. No frameworks, no CDN, no bundler. Dark blue theme driven by CSS variables on `:root`.

- On load it fetches `/api/config` to fill the form and set the input min and max, then polls `/api/status` every 1.5 s. `render(s)` redraws everything from that one status object.
- The chart is hand-drawn on a `<canvas>` in `drawChart(history)`: validation accuracy on the left axis (0 to 100%), validation loss on the right axis, step on x. It needs at least two points.
- In k-fold mode the chart is hidden and one chip per finished fold is shown. The "best" tile becomes the mean fold accuracy.
- `syncMode()` swaps the labels and hints and shows `k` in place of `evalEvery`.
- Reloading the page mid-run restores the form from `status.params`.
- Each setting's help text lives in the HTML as a `.hint` span. A new setting needs an entry in `DEFAULTS` and `LIMITS` (and `WHOLE` when it is an integer) in `server.py`, a use in `run()`, an id in `FIELDS` in the page script, and a `.field` block with a hint.

The UI trains and cross-validates only. Test-set evaluation, plotting, and single-image prediction are CLI-only.

## Things that will bite

- **Windows multiprocessing.** DataLoader workers are spawned, not forked. Each worker re-imports the modules, and start-up costs roughly 5 seconds per worker (`evalEstimate` and the UI hint both assume this). Keep module top levels free of side effects: downloads live inside functions for this reason, and every script has an `if __name__ == "__main__":` guard.
- **Working directory.** `model.pt` and `history.png` use relative paths. `server.py` resolves the UI page by absolute path but still reads and writes `model.pt` relative to the cwd.
- **Three places define the hyperparameter defaults**: the `train()` / `kfold()` / `load_data()` signatures, `main.start_learning`, and `server.DEFAULTS`. They currently agree. Keep them in agreement.
- **`NUM_CLASSES` and `CLASSES`** are defined in different files and must match.
- **Validation and test sets are different datasets from the training data**, with different backgrounds and hands. Low validation accuracy next to high held-out-fold accuracy points to this domain gap, not to a bug.
- **The k-fold score is optimistic** compared with validation: the folds come from the training datasets, and the split is per image.
- **The test set is for final reporting only.** Nothing in training reads it; keep it that way.
- **`torch.load` is called without `weights_only`.** Checkpoints hold only tensors, strings, numbers, lists, and dicts, so they load under either setting. Storing other object types in a checkpoint would break that.

## Code style

Match what is there:

- Comments are short, lower-case, one line, placed above the code or as the first line inside a function. No docstrings, no type hints.
- Names are mostly camelCase (`trainLoader`, `evalEvery`, `getDevice`, `datasetWeights`), with snake_case for several public functions (`load_data`, `kfold_loaders`, `kfold_indices`, `top_confusions`, `plot_history`). Follow the style of the neighbouring code and do not rename existing functions.
- Small flat modules and plain functions. No classes beyond `CNN`, `ASLFolder`, and the HTTP `Handler`. No new dependencies unless they are needed.
- Lines run long (about 120 characters).

## Git

Work happens on feature branches (currently `ajewkes-26`); `main` is the PR target. Ask the user before any `git push`.
