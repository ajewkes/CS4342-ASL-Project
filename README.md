# ASL Sign Classifier

A small neural network that looks at a picture of a hand and says which American Sign Language character it is: a digit `0-9` or a letter `A-Z` (36 classes).

## How it works

1. **Load images.** Five ASL image datasets are downloaded from Kaggle. Every image is turned into a 28x28 grayscale picture.
2. **Train.** A neural network learns from three of the datasets. Training runs for a set number of minutes, not a set number of epochs.
3. **Check.** Every so often the model is scored on a separate validation dataset. The best version so far is saved to `model.pt`.
4. **Use.** The saved model can be scored on a test dataset it has never seen, or asked to read a single image.

| Dataset (Kaggle) | Used for |
| --- | --- |
| `prathumarikeri/american-sign-language-09az` | training |
| `piotrpopis/asl-hands` | training |
| `lexset/synthetic-asl-numbers` | training |
| `ayuraj/asl-dataset` | validation |
| `dorukdemirci/asl-alphabet-dataset` | testing |

## The model

A convolutional neural network (CNN) with about 245,000 parameters, defined in `model.py`. It is small enough to train on a laptop CPU.

```
input: 1 x 28 x 28 grayscale image

conv block 1:  1 ->  32 channels, 28x28 -> 14x14
conv block 2: 32 ->  64 channels, 14x14 -> 7x7
conv block 3: 64 -> 128 channels,  7x7  -> 3x3

flatten -> dropout -> 128 hidden units -> ReLU -> dropout -> 36 scores
```

Each conv block is a 3x3 convolution, batch normalization, ReLU, and 2x2 max pooling. The conv blocks find shapes in the image (edges first, then finger and hand outlines). The two fully connected layers at the end turn those shapes into one score per character, and the highest score is the prediction.

It is trained with the Adam optimizer and cross-entropy loss, at a learning rate of 0.001 by default.

## Design decisions

- **Small grayscale images (28x28).** Hand shape matters more than color, and small inputs keep training fast. The cost is that fine detail is lost, so signs that look alike are harder to tell apart.
- **One label set for every dataset.** The datasets name their folders differently (one uses `0-25` for the letters). `imageImport.py` maps them all onto the same 36 labels and skips folders that are not a digit or letter.
- **Separate datasets for training, validation, and testing.** The model is scored on hands and backgrounds it did not train on, which is a more honest measure than holding back part of the training data.
- **Dataset weighting.** The three training datasets are very different sizes. Without weighting, the biggest one would dominate. Each dataset's share of the training samples is its size to the power 0.5, so smaller datasets are seen more often than their size alone would give them.
- **Light augmentation.** Training images are rotated randomly by up to 10 degrees so the model copes with tilted hands. Validation and test images are left alone.
- **Dropout (0.4) and batch normalization.** Dropout reduces overfitting to the training datasets. Batch normalization makes training more stable.
- **Time limit instead of epochs.** You choose how many minutes to train. This makes runs predictable on any machine, and the limit covers the whole run, including start-up and the final check.
- **Keep the best model, not the last.** The model is scored on the validation set every 500 steps, and `model.pt` always holds the version with the highest validation accuracy.
- **K-fold as a separate check.** K-fold cross-validation trains a fresh model on each fold of the training data to show how stable a setup is. It does not produce a saved model.

## The parts

| File | What it does |
| --- | --- |
| `imageImport.py` | Downloads the datasets, resizes the images, and hands them out in batches. |
| `model.py` | The CNN: three convolution blocks, then two fully connected layers. |
| `train.py` | Training, evaluation, k-fold cross-validation, plotting, and single-image prediction. |
| `main.py` | Command line interface. |
| `server.py` | Local web server behind the training UI. |
| `ui/index.html` | The training UI page: settings form, live chart, and log. |

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

The UI only trains. Use the command line to test the model or read an image.

### Option 2: command line

```
python main.py <command>
```

| Command | What it does |
| --- | --- |
| `-tr` | Train for 10 minutes and save the best model to `model.pt`. |
| `-m` | Same as `-tr`, but asks you for a learning rate or a number of minutes first. |
| `-v` | Score the saved model on the validation set. |
| `-t` | Score the saved model on the test set. |
| `-k` | 3-fold cross-validation on the training data. Saves nothing. |
| `-p` | Save the learning curves from the last training run to `history.png`. |
| `-r <image>` | Read one image and print the character it shows. |

`-v` and `-t` also print the accuracy for each character and the most common mix-ups.

A typical first run:

```
python main.py -tr
python main.py -t
python main.py -r my_hand.jpg
```

Train first. Every command except `-tr`, `-m`, and `-k` needs `model.pt` to exist.
