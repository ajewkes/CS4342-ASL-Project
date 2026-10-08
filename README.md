# ASL Sign Classifier

A small neural network that looks at a picture of a hand and says which American Sign Language character it is: a digit `0-9` or a letter `A-Z` (36 classes).

## How it works

1. **Load images.** Five ASL image datasets are downloaded from Kaggle. Every image is turned into a 64x64 grayscale picture.
2. **Train.** A neural network learns from three of the datasets. Training runs for a set number of epochs (passes over the training data), 5 by default.
3. **Check.** Every so often the model is scored on a separate validation dataset. The best version so far is saved to `model.pt`.
4. **Use.** Once a training run has finished, the saved model can be scored on a test dataset it has never seen, or asked to read a single image.

| Dataset (Kaggle) | Used for |
| --- | --- |
| `prathumarikeri/american-sign-language-09az` | training |
| `piotrpopis/asl-hands` | training, currently switched off (see below) |
| `lexset/synthetic-asl-numbers` | training |
| `ayuraj/asl-dataset` | validation |
| `dorukdemirci/asl-alphabet-dataset` | testing |

## The model

A convolutional neural network (CNN) with about 508,000 parameters, defined in `model.py`. Training uses an NVIDIA GPU (CUDA) or an Apple-silicon GPU (MPS) when there is one, and the CPU otherwise.

```
input: 1 x 64 x 64 grayscale image

conv block 1:   1 ->  32 channels, 64x64 -> 32x32
conv block 2:  32 ->  64 channels, 32x32 -> 16x16
conv block 3:  64 -> 128 channels, 16x16 -> 8x8
conv block 4: 128 -> 128 channels,  8x8  -> 4x4

flatten -> dropout -> 128 hidden units -> ReLU -> dropout -> 36 scores
```

Each conv block is a 3x3 convolution, batch normalization, ReLU, and 2x2 max pooling. The conv blocks find shapes in the image (edges first, then finger and hand outlines). The two fully connected layers at the end turn those shapes into one score per character, and the highest score is the prediction.

It is trained with the Adam optimizer and cross-entropy loss, at a learning rate of 0.001 by default.

## Design decisions

- **64x64 grayscale images.** Hand shape matters more than color. The model first used 28x28 images, but at that size the thumb position that separates M, N, S and T is only a pixel or two. 64x64 keeps that detail while still training in a few minutes per epoch on a laptop GPU. The largest training dataset is only 50x50, so going much bigger adds little. The size is `IMAGE_SIZE` in `model.py`. Models saved at another size are refused with a message to retrain.
- **One label set for every dataset.** The datasets name their folders differently (one uses `0-25` for the letters). `imageImport.py` maps them all onto the same 36 labels and skips folders that are not a digit or letter.
- **Separate datasets for training, validation, and testing.** The model is scored on hands and backgrounds it did not train on, which is a more honest measure than holding back part of the training data.
- **Dataset weighting.** The three training datasets are very different sizes. Without weighting, the biggest one would dominate. Each dataset's share of the training samples is its size to the power 0.5, so smaller datasets are seen more often than their size alone would give them.
- **Light augmentation.** Training images are rotated randomly by up to 10 degrees, and their brightness and contrast are varied, so the model copes with tilted hands and different lighting. Validation and test images are left alone. Stronger augmentation was tested over 2-epoch runs and rejected:
  - Mirroring made the model handle the other hand, but cut validation accuracy from 68% to 43%.
  - Mirroring plus random zoom and shift cut it to 29%.
- **`asl-hands` is switched off.** In most of its photos the hand is tiny or outside the centre square, so the model sees mostly background. Turning it off raised validation accuracy from 54% to 68% at 28x28, and from 51% to 67% at 64x64. It is the only training set of letters photographed in ordinary rooms, though, so it may matter for everyday photos even though validation doesn't reward it. `DATASET_SCALE` in `imageImport.py` turns each training dataset up, down or off.
- **Look-alike signs are reported together.** O and 0, W and 6, V and 2, and F and 9 use the same handshape in ASL, so a single image can't tell them apart. Predictions show them as one answer, such as `O / 0`, and `-v` and `-t` also print the accuracy with these pairs counted as one.
- **Dropout (0.4) and batch normalization.** Dropout reduces overfitting to the training datasets. Batch normalization makes training more stable.
- **Epochs, not a time limit.** You choose how many epochs to train. An earlier version trained for a set number of minutes, but a slow start-up (such as the first dataset download) could use up the whole limit before training began, leaving an untrained model. With epochs the amount of training is always the same, and the UI shows progress and an estimate of the time left. One epoch is as many samples as there are training images, drawn by the weighted sampler, so some images come up twice and others not at all. One epoch takes about 2.5 minutes on an Apple-silicon laptop GPU. Most of that is loading images, so a CPU-only machine is slower (roughly 5 minutes).
- **Keep the best model, not the last.** The model is scored on the validation set every 500 steps, and `model.pt` always holds the version with the highest validation accuracy.
- **Predictions wait for a finished run.** While a run is going, `model.pt` is marked unfinished. Only the run's last save marks it finished, so a run that crashes or is killed stays locked. The UI and `-r` only predict with a finished model. Pressing **Stop** in the UI still counts as finishing. A model must also score at least twice the chance rate on validation (5.6%), so a run stopped almost at once can't be used.
- **K-fold as a separate check.** K-fold cross-validation trains a fresh model on each fold of the training data to show how stable a setup is. It does not produce a saved model.

## The parts

| File | What it does |
| --- | --- |
| `imageImport.py` | Downloads the datasets, resizes the images, and hands them out in batches. |
| `model.py` | The CNN: three convolution blocks, then two fully connected layers. |
| `train.py` | Training, evaluation, k-fold cross-validation, plotting, and single-image prediction. |
| `main.py` | Command line interface. |
| `server.py` | Local web server behind the UI: training jobs and predictions. |
| `ui/index.html` | The UI page: settings form, live chart, log, and the "Try the model" card. |

## Setup

Needs Python 3.10 or newer.

```
pip install -r requirements.txt
```

The datasets download automatically the first time you run anything. This takes a while and may need a [Kaggle login](https://github.com/Kaggle/kagglehub#authenticate). After that they are cached.

Run every command from the project folder.

## Running it

### Option 1: web UI

```
python server.py
```

Open http://localhost:8000, pick your settings, and press **Start training**. You get a live accuracy chart and a log. Each setting is explained on the page. To use another port: `python server.py 8080`.

Once a training run finishes, the **Try the model** card unlocks. Choose or drop an image and the model shows its top three guesses. The **Flip** box mirrors the image first, which can help with a left-handed sign. The `samples/` folder has one clear example per sign to try. Transparent images are put on a white background first. Only the centre square of the picture is used, so keep your hand inside it. A plain background and good light help a lot. The card stays locked while training runs, when no finished model exists, and when the saved model is no better than guessing.

Test-set scoring and learning-curve plots are command-line only.

### Option 2: command line

```
python main.py <command>
```

| Command | What it does |
| --- | --- |
| `-tr` | Train for 5 epochs and save the best model to `model.pt`. |
| `-m` | Same as `-tr`, but asks you for a learning rate or a number of epochs first. |
| `-v` | Score the saved model on the validation set. |
| `-t` | Score the saved model on the test set. |
| `-k` | 3-fold cross-validation on the training data. Saves nothing. |
| `-p` | Save the learning curves from the last training run to `history.png`. |
| `-r <image>` | Read one image and print the character it shows. Needs a finished training run that beats guessing. |

`-v` and `-t` also print the accuracy for each character and the most common mix-ups.

A typical first run:

```
python main.py -tr
python main.py -t
python main.py -r my_hand.jpg
```

Every command except `-tr`, `-m`, and `-k` needs `model.pt`. The repo includes a trained one (83.3% validation accuracy), so you can try predictions straight away. Training replaces it. Git will then show `model.pt` as changed. Commit it only if you mean to share your new model; otherwise `git restore model.pt` brings the shared one back.
