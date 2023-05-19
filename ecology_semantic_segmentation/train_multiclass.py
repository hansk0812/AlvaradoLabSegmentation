import os
import argparse

import glob
import traceback

from . import fish_train_dataset, fish_val_dataset, fish_test_dataset

from .dataset.fish import ORGANS, IMGSIZE, MAXCHANNELS, get_env_variable
EXPTNAME = get_env_variable("EXPTNAME", default_value="deeplabv3p")

from . import unet_model
from .loss_functions import cross_entropy_loss, focal_loss, classification_dice_loss
from .loss_functions import cross_entropy_list, binary_cross_entropy_list, focal_list, classification_dice_list

from .loss_functions import dice_loss

#import get_deepfish_dataset
import random
import numpy as np
import cv2

import torch

from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

torchcpu_to_opencv = lambda img: (img.numpy().transpose((1,2,0))*255).astype(np.uint8)

def train(net, traindataloader, valdataloader, losses_fn, optimizer, save_dir, start_epoch, num_epochs=5000, log_every=100):
    
    background_keys = [0, int(1.6 * num_epochs//5), int(1.8 * num_epochs//5)]
    background_weight = {0: 0, num_epochs//5: 0.3, int(1.6 * num_epochs//5): 0.5, int(1.8 * num_epochs//5): 0.7}
    # sine bg
    binary_flag = False
    for epoch_cycle in range(2*num_epochs//5, num_epochs, 100):
        if binary_flag:
            # without the 0.75 factor, the parameter is too large for subsets to benefit from custom loss
            background_weight[epoch_cycle] = 0.3 + (0.2*np.random.rand()) #0.75*(1 - np.random.rand())
        else:
            # without the 0.5 factor, the parameter is too large for subsets to benefit from custom loss
            background_weight[epoch_cycle] = 0.7 - (0.3*np.random.rand()) #0.75*(1 + 0.5*np.random.rand())
        background_keys.append(epoch_cycle)
        binary_flag = not binary_flag
    
    def find_background_weight(x):
        for idx, b in enumerate(background_keys): 
            if x == 0:
                return 0
            if b>x: 
                bg_w = background_weight[background_keys[idx-1]]
                if x % 99 == 0:
                    print ("."*50, "\n\tUsing background weight: %0.3f\n" % bg_w, '.'*50+'\n')
                return bg_w

    net = net.train()
    if not os.path.isdir("val_images"):
        os.mkdir("val_images")
    if not os.path.isdir(os.path.join(save_dir, "channels%d" % MAXCHANNELS, "img%d" % IMGSIZE)):
        os.makedirs(os.path.join(save_dir, "channels%d" % MAXCHANNELS, "img%d" % IMGSIZE))

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=30, verbose=True) 
    
    for epoch in range(start_epoch+1, num_epochs):  # loop over the dataset multiple times

        [dataset.dataset.set_augment_flag(True) for dataset in traindataloader.dataset.datasets]
        bg_weight = find_background_weight(epoch+1)
        
        running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
        for i, data in enumerate(traindataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, fname = data
            
            """
             #print (inputs.min(), inputs.max(), labels.min(), labels.max())
            img = torchcpu_to_opencv(inputs[0])
            seg = torchcpu_to_opencv(labels[0])
            cv2.imwrite('train.png', img); cv2.imwrite('seg.png', seg)
            """
            
            if torch.cuda.is_available():
                if isinstance(labels, list):
                    inputs, labels = inputs.cuda(), [x.cuda() for x in labels]
                else:
                    inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            # Found ValueError: Expected more than 1 value per channel when training 
            assert inputs.shape[0] != 1, "Found last batch with 1 example only, change batch size multiplier!"

            outputs = net(inputs)
            outputs = F.sigmoid(outputs)

            if isinstance(outputs, tuple):
                outputs = [outputs[0]] + outputs[1]
            
            ce_l, bce_l, fl_l, dice, generalized_dice, twersky_dice, focal_dice = \
                    losses_fn(outputs, labels, composite_set_theory=True, background_weight=bg_weight)
            dice_l = [dice, generalized_dice, twersky_dice, focal_dice]
            
            # focal_dice works great with DeepLabv3 but doesn't as much with resnet34 or resnet50

            #loss =  dice + generalized_dice + twersky_dice + focal_dice
            #loss = dice + generalized_dice + twersky_dice + bce_l
            loss = generalized_dice + focal_dice + bce_l #focal_dice + 
            # Chose generalized_dice with k for correctness' sake when focal_dice_bg was giving good variation 
            # + k * (bg_focal_dice + bg_generalized_dice)
            
            # focal_dice #ce_l + fl_l + sum(dice_l)
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
                    torch.save(net.state_dict(), os.path.join(save_dir, "channels%d" % MAXCHANNELS, 
                                "img%d" % IMGSIZE,"%s_epoch%d.pt" % (EXPTNAME, epoch)))

                print("Epoch: %d ; Batch: %d/%d : Training Loss: %.8f" % (epoch+1, i+1, len(traindataloader), running_loss / log_every))
                print ("\t Cross-Entropy: %0.8f; BCE: %.8f; Focal Loss: %0.8f; Dice Loss: %0.8f [D: %.8f, GD: %.8f, TwD: %.8f, FocD: %.8f]" % (
                            ce_t/float(log_every), bce_t/float(log_every), fl_t/float(log_every), sum([x/float(log_every) for x in dice_t]), 
                            dice_t[0]/float(log_every), dice_t[1]/float(log_every), dice_t[2]/float(log_every), dice_t[3]/float(log_every)))
                
                running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
        
        [dataset.dataset.set_augment_flag(False) for dataset in valdataloader.dataset.datasets]
        with torch.no_grad():
            net = net.eval()
            val_running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
            for j, val_data in enumerate(valdataloader, 0):
                val_inputs, val_labels, _ = val_data
                
                if torch.cuda.is_available():
                    if isinstance(val_labels, list):
                        val_inputs, val_labels = val_inputs.cuda(), [x.cuda() for x in val_labels]
                    else:
                        val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                
                val_outputs = net(val_inputs)
                val_outputs = F.sigmoid(val_outputs)
                
                if isinstance(val_outputs, tuple):
                    val_outputs = [val_outputs[0]] + val_outputs[1]
                
                if isinstance(val_outputs, list):
                    bce_l = binary_cross_entropy_list(val_labels, val_outputs)
                else:
                    bce_l = cross_entropy_loss(val_labels, val_outputs, bce=True)
                #ce_l, bce_l, fl_l, dice, generalized_dice, twersky_dice, focal_dice = losses_fn(val_outputs, val_labels)
                
                #dice_l = [dice, generalized_dice, twersky_dice, focal_dice]
                val_loss = bce_l + dice #generalized_dice #ce_l + fl_l + sum(dice_l)
                val_running_loss += val_loss.item()
                #ce_t += ce_l.item()
                bce_t += bce_l.item()
                #fl_t += fl_l.item()
                #dice_t = [x.item() + y for (x,y) in zip(dice_l, dice_t)]
                
                # save 5 images per epoch for testing
                if j < 10:
                    
                    if not os.path.isdir(os.path.join("val_images", str(epoch))):
                        os.mkdir(os.path.join("val_images", str(epoch)))
                    
                    if torch.cuda.is_available():
                        val_inputs = val_inputs.cpu()
                        if isinstance(val_labels, list):
                            val_outputs = [x.cpu() for x in val_outputs]
                            val_labels = [x.cpu() for x in val_labels]
                        else:
                            val_outputs = val_outputs.cpu()
                            val_labels = val_labels.cpu()
                    
                    img = torchcpu_to_opencv(val_inputs[0])

                    for idx in range(len(ORGANS)):
                        if isinstance(val_labels, list):
                            gt = torchcpu_to_opencv(val_labels[0][idx])
                            out = torchcpu_to_opencv(val_outputs[0][idx])
                        else:
                            gt = torchcpu_to_opencv(val_labels[0][idx:idx+1])
                            out = torchcpu_to_opencv(val_outputs[0][idx:idx+1])

                        imgpath = os.path.join("val_images", str(epoch), str(j)) 

                        cv2.imwrite(imgpath+"_img.png", img)
                        cv2.imwrite(imgpath+"_gt_organ%d.png" % idx, gt)
                        cv2.imwrite(imgpath+"_pred_organ%d.png" % idx, out)

            num_avg = float(len(valdataloader)*val_inputs.shape[0])
            val_running_loss /= float(num_avg)

        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch + 1)
        else:
            scheduler.step(val_running_loss)

        print("\nVal Loss: %.8f!" % val_running_loss)
#        print ("\t Cross-Entropy: %0.8f; BCE: %.8f; Focal Loss: %0.8f; Dice Loss: %0.8f [D: %.8f, GD: %.8f, TwD: %.8f, FocD: %.8f]" % (
#                    ce_t/float(num_avg), bce_t/float(num_avg), fl_t/float(num_avg), sum([x/float(num_avg) for x in dice_t]), 
#                    dice_t[0]/float(num_avg), dice_t[1]/float(num_avg), dice_t[2]/float(num_avg), dice_t[3]/float(num_avg)))

    print('finished training')

def losses_fn(x, g, composite_set_theory=False, background_weight=0):
    
    # Hardcoded subset membership loss for each composite set of organs
    # [whole_body, ventral_side, dorsal_side] = [1.         0.20878016 0.22319692] 
    # Use dataset.get_relative_ratios for every organ subset

    CLASS_INDEX = 1
    if g.shape[CLASS_INDEX] > 1:
        losses = [losses_fn(g[:,idx:idx+1,:,:], x[:,idx:idx+1,:,:]) for idx in range(g.shape[CLASS_INDEX])]
        return [sum(i) for i in zip(*losses)] # /float(g.shape[CLASS_INDEX]) Using sum loss for now
    
    if isinstance(x, list):
        bce_loss = binary_cross_entropy_list(x, g)
        ce_loss, fl_loss = cross_entropy_list(x, g), focal_list(x, g, factor=1e-5)
        dice, generalized_dice, twersky_dice, focal_dice = classification_dice_list(x, g, factor=10)
    else: 
        bce_loss = cross_entropy_loss(x, g, bce=True, background_weight=background_weight)
        ce_loss, fl_loss = cross_entropy_loss(x, g, bce=False, background_weight=background_weight), focal_loss(x, g, factor=1, background_weight=background_weight)

        dice, generalized_dice, twersky_dice, focal_dice = classification_dice_loss(x, g, factor=10, background_weight=background_weight)
    
    return_losses = [ce_loss, bce_loss, fl_loss, dice, generalized_dice, twersky_dice, focal_dice]
    
    # ignore subset losses 20% of the training to prevent subsets dominating superset shapes
    if composite_set_theory and np.random.rand() < 0.2:
        
        whole_body_g, whole_body_p = g[:,0:1,...], x[:,0:1,...]
        ventral_side_g, ventral_side_p = g[:,1:2,...], x[:,1:2,...]
        dorsal_side_g, dorsal_side_p = g[:,2:3,...], x[:,2:3,...]

        ventral_side_w = 4.789727146487483
        dorsal_side_w = 4.480348563949717
        
        ventral_side_negative_loss = sum(list(losses_fn(ventral_side_g, whole_body_p * ventral_side_p)))
        dorsal_side_negative_loss = sum(list(losses_fn(dorsal_side_g, whole_body_p * dorsal_side_p)))
        
#        # Ignoring positive loss for further ablations
#        ventral_side_positive_loss = sum(list(losses_fn(whole_body_g, \
#                                        (whole_body_p * (1 - ventral_side_p) + (whole_body_p * ventral_side_p + ventral_side_p)*0.5))))
#        dorsal_side_positive_loss = sum(list(losses_fn(whole_body_g, \
#                                        (whole_body_p * (1 - dorsal_side_p) + (whole_body_p * dorsal_side_p + dorsal_side_p)*0.5))))

        return_losses1 = [x + ventral_side_w * y \
                for x,y in zip(return_losses, ventral_side_negative_loss)] 
        # x + 4.789727146487483 * y Subsets creating gaps in whole_body segment
        return_losses2 = [x + dorsal_side_w * y \
                for x,y in zip(return_losses, dorsal_side_negative_loss)]
        # x + 4.480348563949717 * y Subsets creating gaps in whole_body segment
        
        return_losses = [x + y \
                for x,y in zip(return_losses1, return_losses2)]

    return return_losses

def load_recent_model(saved_dir, net, epoch=None):
    # Load model from a particular epoch and train like the rest of the epochs are relevant anyway
    #TODO: Delete all models from epoch to latest_epoch to enable checkpoint dir consistency

    try:
        gl = glob.glob(os.path.join(saved_dir, "channels%d" % MAXCHANNELS, 
                            "img%d" % IMGSIZE, "%s*"%EXPTNAME))
        
        epochs_list = [int(x.split("epoch")[-1].split('.')[0]) for x in gl] 
        latest_index = np.argmax(epochs_list)
        if epoch is None:
            index = latest_index
        else:
            index = epochs_list.index(epoch)

        model_file = gl[index]

        start_epoch = int(model_file.split("epoch")[-1].split('.')[0])
        if not torch.cuda.is_available():
            load_state = torch.load(model_file, map_location=torch.device('cpu'))
        else:    
            load_state = torch.load(model_file)
        print ("Used latest model file: %s" % model_file)
        net.load_state_dict(load_state)
        
        return start_epoch

    except Exception:
        traceback.print_exc()
        return -1

import segmentation_models_pytorch as smp
#unet_model = smp.Unet(
#            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#            classes=1,                      # model output channels (number of classes in your dataset)
#            #activation="silu"              ReLU makes sigmoid more stable # changed from default relu to silu for some resnet50 tests
#        )

#TODO: Layer normalization
unet_model = smp.DeepLabV3Plus(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(ORGANS),                      # model output channels (number of classes in your dataset)
            #activation="prelu"
        )

if __name__ == "__main__":
    
    #TODO Discretized image sizes to closest multiple of 8
   
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", default=7, type=int, help="Multiples of 9 give the best GPU utilization (~2023)")
    ap.add_argument("--start_epoch", default=0, type=int, help="Start training from a known model for a conceptual optimization landscape")
    ap.add_argument("--lr", default=0.0003, type=float, help="Start training based on amount of predictions>0")
    args = ap.parse_args()

   # Training script

    def worker_init_fn(worker_id):
        torch_seed = torch.initial_seed()
        random.seed(torch_seed + worker_id)
        if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
            torch_seed = torch_seed % 2**30
        np.random.seed(torch_seed + worker_id)

    train_dataloader = DataLoader(fish_train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=3, \
                                    worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(fish_val_dataset, shuffle=False, batch_size=1, num_workers=1)
    
    saved_dir = os.path.join("models", EXPTNAME)
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    start_epoch = load_recent_model(saved_dir, unet_model, epoch=None if args.start_epoch==0 else args.start_epoch)
    
    if torch.cuda.is_available():
        unet_model = unet_model.cuda()
    
    optimizer = optim.Adam(unet_model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(unet_model.parameters(), lr=0.001, momentum=0.9)
    
    train(unet_model, train_dataloader, val_dataloader, losses_fn, optimizer, save_dir=saved_dir, start_epoch=start_epoch, 
            log_every = len(train_dataloader) // 5)

