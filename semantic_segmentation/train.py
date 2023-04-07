import os
import glob
import traceback

from .dataset.fish import fish_train_dataset, fish_val_dataset, fish_test_dataset
from .model import vgg_unet
from .loss_f import cross_entropy_loss, focal_loss, classification_dice_loss

#import get_deepfish_dataset

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

def train(net, traindataloader, valdataloader, losses_fn, optimizer, save_dir, start_epoch, num_epochs=1000, log_every=100):
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=60, verbose=True) 
    for epoch in range(start_epoch+1, num_epochs):  # loop over the dataset multiple times

        running_loss, ce_t, fl_t, dice_t = 0.0, 0.0, 0.0, [0.0, 0.0, 0.0]
        for i, data in enumerate(traindataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, fname = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            ce_l, fl_l, dice, twersky_dice, focal_dice = losses_fn(outputs, labels)
            dice_l = [dice, twersky_dice, focal_dice]
            loss = F.binary_cross_entropy(torch.sigmoid(outputs), labels)  #ce_l + fl_l + sum(dice_l)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            ce_t+= ce_l.item()
            fl_t+= fl_l.item()
            
            dice_t = [x.item() + y for (x,y) in zip(dice_l, dice_t)]
            if i % log_every == log_every-1:    # print every log_every mini-batches
                
                if epoch % 5 == 0:
                    torch.save(net.state_dict(), "%s/%s_epoch%d.pt" % (save_dir, "densenet", epoch))

                print("Epoch: %d ; Batch: %d/%d : Training Loss: %.8f" % (epoch+1, i+1, len(traindataloader), running_loss / log_every))
                print ("\t Cross-Entropy: %0.8f; Focal Loss: %0.8f; Dice Loss: %0.8f [D: %.8f, TwD: %.8f, FocD: %.8f]" % (
                            ce_t/float(log_every), fl_t/float(log_every), sum([x/float(log_every) for x in dice_t]), 
                            dice_t[0]/float(log_every), dice_t[1]/float(log_every), dice_t[2]/float(log_every)))
                
                running_loss, ce_t, fl_t, dice_t = 0.0, 0.0, 0.0, [0.0, 0.0, 0.0]
        
        with torch.no_grad():
            val_running_loss, ce_t, fl_t, dice_t = 0.0, 0.0, 0.0, [0.0, 0.0, 0.0]
            for j, val_data in enumerate(valdataloader, 0):
                val_inputs, val_labels, _ = val_data
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

                val_outputs = net(val_inputs)
                ce_l, fl_l, dice, twersky_dice, focal_dice = losses_fn(val_outputs, val_labels)
                dice_l = [dice, twersky_dice, focal_dice]
                val_loss = F.binary_cross_entropy(torch.sigmoid(val_outputs), val_labels) #ce_l + fl_l + sum(dice_l)
                val_running_loss += val_loss.item()
                ce_t += ce_l.item()
                fl_t += fl_l.item()
                dice_t = [x.item() + y for (x,y) in zip(dice_l, dice_t)]
            
            num_avg = float(len(valdataloader)*val_inputs.shape[0])
            val_running_loss /= float(num_avg)

        scheduler.step(val_running_loss)

        print("\nVal Loss: %.8f!" % val_running_loss)
        print ("\t Cross-Entropy: %0.8f; Focal Loss: %0.8f; Dice Loss: %0.8f [D: %.8f, TwD: %.8f, FocD: %.8f]" % (
                    ce_t/float(num_avg), fl_t/float(num_avg), sum([x/float(num_avg) for x in dice_t]), 
                    dice_t[0]/float(num_avg), dice_t[1]/float(num_avg), dice_t[2]/float(num_avg)))

    print('finished training')

def losses_fn(x, g): 
    ce_loss, fl_loss = cross_entropy_loss(x, g), focal_loss(x, g, factor=1e-5)
    dice, twersky_dice, focal_dice = classification_dice_loss(x, g, factor=10)
    return ce_loss, fl_loss, dice, twersky_dice, focal_dice

def load_recent_model(saved_dir, net):
    
    try:
        model_file = sorted(glob.glob(os.path.join(saved_dir, "*")), \
                            key=lambda x: int(x.split("epoch")[-1].split('.')[0]))[-1]

        start_epoch = int(model_file.split("epoch")[-1].split('.')[0])
        load_state = torch.load(model_file)
        print ("Used latest model file: %s" % model_file)
        net.load_state_dict(load_state)
        return start_epoch
    except Exception:
        traceback.print_exc()
        print ("Pretrained checkpoint unavailable! Starting from scratch!")
        return -1

if __name__ == "__main__":
    
    #TODO Discretized image sizes to closest multiple of 8
   
   # Training script

    train_dataloader = DataLoader(fish_train_dataset, shuffle=True, batch_size=7, num_workers=3)
    val_dataloader = DataLoader(fish_val_dataset, shuffle=False, batch_size=1, num_workers=1)
    
    optimizer = optim.SGD(vgg_unet.parameters(), lr=0.005, momentum=0.9)
    #optimizer = optim.Adam(vgg_unet.parameters(), lr=0.001)

    model_dir = "vgg/"
    save_dir = "models/"+model_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    start_epoch = load_recent_model(save_dir, vgg_unet)

    vgg_unet = vgg_unet.cuda()

    train(vgg_unet, train_dataloader, val_dataloader, losses_fn, optimizer, save_dir=save_dir, start_epoch=start_epoch, log_every=100)

