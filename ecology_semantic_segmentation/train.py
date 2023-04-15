import argparse

import os
import glob
import traceback

from . import fish_train_dataset, fish_val_dataset, fish_test_dataset
from . import vgg_unet
from .loss_functions import cross_entropy_loss, focal_loss, classification_dice_loss

#import get_deepfish_dataset
import numpy as np
import cv2

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

torchcpu_to_opencv = lambda img: (img.numpy().transpose((1,2,0))*255).astype(np.uint8)

def train(net, traindataloader, valdataloader, losses_fn, optimizer, save_dir, start_epoch, num_epochs=2000, log_every=100):
    
    if not os.path.isdir("val_images"):
        os.mkdir("val_images")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=30, verbose=True) 
    
    [dataset.dataset.set_augment_flag(True) for dataset in traindataloader.dataset.datasets]
    for epoch in range(start_epoch+1, num_epochs):  # loop over the dataset multiple times

        running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
        for i, data in enumerate(traindataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, fname = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = torch.softmax(net(inputs), dim=1)
            
            ce_l, bce_l, fl_l, dice, generalized_dice, twersky_dice, focal_dice = losses_fn(outputs, labels)
            dice_l = [dice, generalized_dice, twersky_dice, focal_dice]
            
            loss = generalized_dice + dice + twersky_dice #bce_l #ce_l + fl_l + sum(dice_l)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            ce_t+= ce_l.item()
            bce_t+= bce_l.item()
            fl_t+= fl_l.item()
            
            dice_t = [x.item() + y for (x,y) in zip(dice_l, dice_t)]

            if log_every == 0:
                log_every = 1
            if len(traindataloader) < log_every or \
               i % log_every == log_every-1:    # print every log_every mini-batches
                
                if epoch % 10 == 0:
                    torch.save(net.state_dict(), "%s/%s_epoch%d.pt" % (save_dir, "densenet", epoch))

                print("Epoch: %d ; Batch: %d/%d : Training Loss: %.8f" % (epoch+1, i+1, len(traindataloader), running_loss / log_every))
                print ("\t Cross-Entropy: %0.8f; BCE: %.8f; Focal Loss: %0.8f; Dice Loss: %0.8f [D: %.8f, GD: %.8f, TwD: %.8f, FocD: %.8f]" % (
                            ce_t/float(log_every), bce_t/float(log_every), fl_t/float(log_every), sum([x/float(log_every) for x in dice_t]), 
                            dice_t[0]/float(log_every), dice_t[1]/float(log_every), dice_t[2]/float(log_every), dice_t[3]/float(log_every)))
                
                running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
        
        [dataset.dataset.set_augment_flag(False) for dataset in valdataloader.dataset.datasets]
        with torch.no_grad():
            val_running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
            for j, val_data in enumerate(valdataloader, 0):
                val_inputs, val_labels, _ = val_data
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

                val_outputs = net(val_inputs)
                ce_l, bce_l, fl_l, dice, generalized_dice, twersky_dice, focal_dice = losses_fn(val_outputs, val_labels)
                dice_l = [dice, generalized_dice, twersky_dice, focal_dice]
                val_loss = generalized_dice #ce_l + fl_l + sum(dice_l)
                val_running_loss += val_loss.item()
                ce_t += ce_l.item()
                bce_t += bce_l.item()
                fl_t += fl_l.item()
                dice_t = [x.item() + y for (x,y) in zip(dice_l, dice_t)]
                
                # save 5 images per epoch for testing
                if j < 5:
                    
                    if not os.path.isdir(os.path.join("val_images", str(epoch))):
                        os.mkdir(os.path.join("val_images", str(epoch)))
                    
                    if torch.cuda.is_available():
                        val_inputs = val_inputs.cpu()
                        val_outputs = val_outputs.cpu()
                        val_labels = val_labels.cpu()

                    img = torchcpu_to_opencv(val_inputs[0])
                    gt = torchcpu_to_opencv(val_labels[0])
                    out = torchcpu_to_opencv(val_outputs[0])
                    
                    imgpath = os.path.join("val_images", str(epoch), str(j)) 

                    cv2.imwrite(imgpath+"_img.png", img)
                    cv2.imwrite(imgpath+"_gt.png", gt)
                    cv2.imwrite(imgpath+"_pred.png", out)

            num_avg = float(len(valdataloader)*val_inputs.shape[0])
            val_running_loss /= float(num_avg)

        scheduler.step(val_running_loss)

        print("\nVal Loss: %.8f!" % val_running_loss)
        print ("\t Cross-Entropy: %0.8f; BCE: %.8f; Focal Loss: %0.8f; Dice Loss: %0.8f [D: %.8f, GD: %.8f, TwD: %.8f, FocD: %.8f]" % (
                    ce_t/float(num_avg), bce_t/float(num_avg), fl_t/float(num_avg), sum([x/float(num_avg) for x in dice_t]), 
                    dice_t[0]/float(num_avg), dice_t[1]/float(num_avg), dice_t[2]/float(num_avg), dice_t[3]/float(num_avg)))

    print('finished training')

def losses_fn(x, g): 
    bce_loss = cross_entropy_loss(x, g, bce=True)
    ce_loss, fl_loss = cross_entropy_loss(x, g), focal_loss(x, g, factor=1e-5)
    dice, generalized_dice, twersky_dice, focal_dice = classification_dice_loss(x, g, factor=10)
    return ce_loss, bce_loss, fl_loss, dice, generalized_dice, twersky_dice, focal_dice

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
   
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", default=7, type=int, help="imgsize -> batchsize: 256 -> 7; 128 -> 20;")
    args = ap.parse_args()

   # Training script

    train_dataloader = DataLoader(fish_train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=3)
    val_dataloader = DataLoader(fish_val_dataset, shuffle=False, batch_size=1, num_workers=1)
    
    #optimizer = optim.SGD(vgg_unet.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(vgg_unet.parameters(), lr=0.01)

    model_dir = "vgg/"
    save_dir = "models/"+model_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    start_epoch = load_recent_model(save_dir, vgg_unet)

    vgg_unet = vgg_unet.cuda()
    
    train(vgg_unet, train_dataloader, val_dataloader, losses_fn, optimizer, save_dir=save_dir, start_epoch=start_epoch, 
            log_every = len(train_dataloader) // 5)

