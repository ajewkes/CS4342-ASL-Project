import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader

'''how are we doing data?
    like obviously 35 by (dataset size) array
    but how do we import and get our model to read pixel data'''

#I ripped all this off the internet, this all looks pretty straightforward 
#but I could 100% be misunderstanding this
class Softmax:

    def __init__(self, dataset, n_class, batching = 32, workers = 2, learnRate = 0.001, moment = 0.75):
        #passing data to be referenced during training
        self.data_loader = Dataloader(dataset=dataset, batch_size = batching, shuffle = True, num_workers = workers)
        #
        self.n_class = n_class
        #trying to use GPU if available
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        #empty weight matrix
        self.weights = nn.Linear(dataset[0][0].shape[0], self.n_class, bias=True)
        #define the model type
        self.model = nn.Sequential(self.weights, nn.Softmax().to(self.device))
        #define loss
        self.criterion = nn.CrossEntropyLoss()
        #define parameters
        self.optimizer = optim.SGD(self.model.parameters(), lr = learnRate, momentum = moment)
        #define where model data is saved to
        #self.model_path = ''

    def train(self, iterations = 100, checkpoints = 10):
        loss = None
        for epoch in range (iterations):
            for _, (inputs, labels) in enumerate(self.data_loader, 0):
                #pull input/label at this value
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                #model the values after applying regression
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                #calculate loss
                loss = self.criterion(outputs, labels)
                loss.backwards()
                #increment loss
                self.optimizer.step()

            #after a number of iterations, print current value
            if epoch % checkpoints == 0:
                print('Loss at %d epoch: %5f', epoch, loss.item())

        #print end of module stats
        print('Loss at last epoch %5f', loss.item())
        print('Saving the model')
        #save model
        torch.save(self.model.state_dict(), self.model_path)

    def accuracy_on_train(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in self.data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                predict = torch.max(outputs, 1)

                correct += (predict == labels).sum().item()
                total += labels.shape[0]

        incorrect = total - correct
        print("model predicted %d correct out of %d inputs and had %d incorrect prediction", correct, total, incorrect)

        return correct/total

            
