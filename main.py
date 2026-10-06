import sys
from os import path
import random
import filetype
import imageImport

def lookup(x):
    if x < 1 or x > 35:
        print("Invalid input")
        return -1
    if x > 26 and x <= 35:
        return str(x - 26)
    else:
        return chr(ord('@')+x)

def safe_read(numbytes):
    value = input()
    return value[:numbytes].lower()

def modify_param():
    print("What would you like to change the value to?")
    x = safe_read(8)
    print("Value successfully changed. New value: ", x)
    return x

def start_learning(cmd, filepath = "N/A"):
    # hyperparameters here for some reason, can/will change
    lr = 0.01
    epoch = 10
    momentum = 0.5
    match cmd.lower():
        case "-m":
            print("What hyperparameter are you trying to modify?\n [L]earing Rate, [E]poch, or [M]omentum")
            x = safe_read(1)
            match x:
                case 'l':
                    lr = modify_param()
                case 'e':
                    epoch = modify_param()
                case 'm':
                    momentum = modify_param()
                case _:
                    print("Incorrect value given, exiting")
                    exit(1)
        case "-t":
            print("Testing...")

            #call cnn with testing data

            exit(1)
        case "-tr":
            print("Training...")

            #call cnn with training data

            exit(1)
        case "-v":
            print("Validating...")

            # call cnn with validation data

            print(f"parameters were:\nLearning Rate: {lr}\nEpoch: {epoch}\nMomentum: {momentum}")
            exit(1)
        case "-r":
            if filepath == "N/A":
                print("no picture given, exiting.")
                exit(1)
            elif path.isfile(filepath) is False:
                print("Path given is not a file, exiting")
                exit(1)
            elif filetype.is_image(filepath) is False:
                print("Not an image, exiting")
                exit(1)
            print("Determining Input...")
            
            #call cnn with path provided
            
            #temp to compile
            a = lookup(int(random.random() * 34) + 1)

            print("Character associated with value given is", a)
        case _:
            print("Unknown Command, exiting")
            exit(1)
    exit(1)

def __main__():
    if sys.argv[1] is None:
        print("No command giving, exiting")
        exit(1)
    if sys.argv[1].lower() == "-r" and sys.argv[2] is None:
        print("no file path given, exiting")
        exit(1)
    if len(sys.argv) >= 2 and not(sys.argv[1] == "-r" and len(sys.argv) >= 3):
        print("More arguments provided than is necessary, ignoring excess arguments")
    if sys.argv[1] == "-r":
        start_learning(sys.argv[1], sys.argv[2])
    else:
        start_learning(sys.argv[1])
    exit(1)

if __name__ == "__main__":
    __main__()