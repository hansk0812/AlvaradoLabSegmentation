from .dataset.fish import fish_train_dataset, fish_val_dataset, fish_test_dataset
from .model import vgg_unet

import get_deepfish_dataset

import torch
from torch import nn 
import torch.optim as optim
from torch.utils.data import DataLoader

def train(net, dataset, num_epochs, optimizer, log_every=100):

    train_loader = DataLoader(dataset, batch_size=4, num_workers=4)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, segments = data
            images, segments = images.cuda(), segments.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, segments)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_every == log_every-1:    # print every 2000 mini-batches
                print("Epoch: %d ; Loss: %.7f" % (epoch+1, running_loss / float(log_every)))
                running_loss = 0.0

if __name__ == "__main__":
    
    #TODO Discretized image sizes to closest multiple of 8
    
    vgg_unet = vgg_unet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg_unet.parameters(), lr=0.001, momentum=0.9)

    train(net=vgg_unet, dataset=fish_train_dataset, num_epochs=500, optimizer=optimizer)  
