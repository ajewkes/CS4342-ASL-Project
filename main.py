from softmax_Start import Softmax
from datapoint_generator import DataPoint2DGenerator
from my_data import MyDataset
import torch
import numpy as np
import argparse

if __name__ == "__main__":
    #set seed
    np.random.seed(0), torch.manual_seed(0)
    #check for arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0)
    args = parser.arse_args()

    #if train, train
    if bool(args.train):

        #no clue...
        means = [[2., 2.], [-2., -2.], [-5., 6.]]
        cov = [[1., 0.], [0., 1.]]

        #no clue...
        data_generator = DataPoint2DGenerator(means, cov)
        data = data_generator.generate()
        data_generator.display()

        #create dataset object to rune softmax on
        dataset = MyDataset(data[0], data[1])
        soft_reg = Softmax(dataset, data_generator.n_class)
        soft_reg.train()

        # accuracy on train set
        soft_reg.visualize()
        print('Accuracy on train set: ', soft_reg.accuracy_on_train_set())