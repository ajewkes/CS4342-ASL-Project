import sys
from os import path
from PIL import Image
import imageImport
import train

def safe_read(numbytes):
    value = input()
    return value[:numbytes].lower()

def modify_param():
    print("What would you like to change the value to?")
    x = safe_read(8)
    try:
        x = float(x)
    except ValueError:
        print("Value is not a number, exiting")
        exit(1)
    print("Value successfully changed. New value: ", x)
    return x

def modify_epochs():
    # epochs have to be a whole number of at least 1
    x = modify_param()
    if x != int(x) or x < 1:
        print("Epochs must be a whole number of at least 1, exiting")
        exit(1)
    return int(x)

def load_model():
    if not path.isfile(train.CHECKPOINT):
        print("No trained model found, run -tr first")
        exit(1)
    saved = train.readCheckpoint(train.CHECKPOINT)
    if train.wrongSize(saved):
        print(train.wrongSize(saved))
        exit(1)
    return train.fromCheckpoint(saved)

def load_finished_model():
    # predictions only use a model whose training run has finished and that beats guessing
    if not path.isfile(train.CHECKPOINT):
        print("No trained model found, run -tr first")
        exit(1)
    saved = train.readCheckpoint(train.CHECKPOINT)
    problem = train.notUsable(saved)
    if problem:
        print(problem)
        exit(1)
    return train.fromCheckpoint(saved)

def run_training(lr, epochs):
    print("Training...")
    trainLoader, validLoader, _ = imageImport.load_data()
    train.train(trainLoader, validLoader, lr=lr, epochs=epochs)
    print(f"parameters were:\nLearning Rate: {lr}\nEpochs: {epochs}\nOptimizer: Adam")

def run_evaluation(split):
    model = load_model()
    # workers are only needed for training
    _, validLoader, testLoader = imageImport.load_data(num_workers=0)
    loader = validLoader if split == "validation" else testLoader
    loss, acc, perClass, confusion = train.evaluate(model, loader, train.getDevice())
    print(f"{split} loss: {loss:.3f}  accuracy: {acc:.3f}  "
          f"(look-alike pairs {', '.join(x + '/' + y for x, y in imageImport.LOOKALIKES)} counted as one: "
          f"{train.lookalikeAccuracy(confusion):.3f})")
    print("per-class accuracy:")
    print("  ".join(f"{imageImport.CLASSES[i]}:{a:.2f}" for i, a in enumerate(perClass) if a is not None))
    print("most common mistakes (true -> predicted):")
    print("  ".join(f"{t}->{p}:{c}" for t, p, c in train.top_confusions(confusion)))

def start_learning(cmd, filepath = "N/A"):
    # hyperparameters here for some reason, can/will change
    lr = 0.001
    epochs = train.EPOCHS
    match cmd.lower():
        case "-m":
            print("What hyperparameter are you trying to modify?\n [L]earing Rate or [E]pochs of training")
            x = safe_read(1)
            match x:
                case 'l':
                    lr = modify_param()
                case 'e':
                    epochs = modify_epochs()
                case _:
                    print("Incorrect value given, exiting")
                    exit(1)
            run_training(lr, epochs)
        case "-t":
            print("Testing...")
            run_evaluation("test")
        case "-tr":
            run_training(lr, epochs)
        case "-v":
            print("Validating...")
            run_evaluation("validation")
        case "-k":
            print("K-fold cross-validation (3 folds)...")
            train.kfold(k=3, lr=lr, epochs=epochs)
        case "-p":
            if not path.isfile(train.CHECKPOINT):
                print("No trained model found, run -tr first")
                exit(1)
            print("Learning curves saved to", train.plot_history(train.CHECKPOINT))
        case "-r":
            if filepath == "N/A":
                print("no picture given, exiting.")
                exit(1)
            elif path.isfile(filepath) is False:
                print("Path given is not a file, exiting")
                exit(1)
            try:
                Image.open(filepath).verify()
            except Exception:
                print("Not an image, exiting")
                exit(1)
            print("Determining Input...")

            character, confidence = train.predict(load_finished_model(), filepath)

            print(f"Character associated with value given is {character} ({confidence:.0%} confident)")
        case _:
            print("Unknown Command, exiting")
            exit(1)

def __main__():
    if len(sys.argv) < 2:
        print("No command giving, exiting")
        exit(1)
    if sys.argv[1].lower() == "-r" and len(sys.argv) < 3:
        print("no file path given, exiting")
        exit(1)
    if len(sys.argv) > 2 and not(sys.argv[1] == "-r" and len(sys.argv) == 3):
        print("More arguments provided than is necessary, ignoring excess arguments")
    if sys.argv[1] == "-r":
        start_learning(sys.argv[1], sys.argv[2])
    else:
        start_learning(sys.argv[1])

if __name__ == "__main__":
    __main__()
