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
#
from .loss_functions import dice_loss

#from .loss_composite import losses_fn

from .utils.subsets_union import return_union_sets_descending_order

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

import segmentation_models_pytorch as smp
from .model import DeepLabV3PlusDepthwise

# #TODO: Idea: Impose GT on prediction and compute loss without GT of subset for superset learning
def train(net, traindataloader, valdataloader, losses_fn, optimizer, save_dir, start_epoch, num_epochs=11000, log_every=100, early_stop_epoch=400):
    
    #TODO: KL Divergence as loss and predictions as probability distributions

    # Initial test for the elusive composite_set_theory flag
    composite_flag = (len(ORGANS) > 1)
    print ("Using composite losses: ", composite_flag)

    background_keys = [0, int(1.6 * num_epochs//5), int(1.8 * num_epochs//5)]
    background_weight = {0: 0, num_epochs//10: 0.3, int(1.6 * num_epochs//5): 0.5, int(1.8 * num_epochs//5): 0.7}
    # sine bg
    binary_flag = False
    for epoch_cycle in range(2*num_epochs//5, num_epochs, 100):
        # Increasing background weight init values to account for edges removal in non-superset segment
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
    
    # #TODO: cosine annealing needs more analysis into the epoch parameter 
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=50, verbose=True) 
    
    for epoch in range(start_epoch+1, num_epochs):  # loop over the dataset multiple times

        print ("Epoch %d" % epoch)

        [dataset.dataset.set_augment_flag(True) for dataset in traindataloader.dataset.datasets]
        bg_weight = find_background_weight(epoch+1)
        
        # loss curriculum
        # DICE score gives automatic edge detection with superset based predictions to help improve performance of whole_body beyond best multiclass model
        #TODO: loss curriculum heuristics wrt early_stop_epoch
        # loss curriculum gave minimal changes in performance when regularized losses were used!
        generalized_dice_w = int(epoch<1000) + int(epoch<2500 and epoch>1500)
        generalized_dice_w = int(generalized_dice_w>0)

        focal_dice_w = int(epoch>2000) + int(generalized_dice_w!=1 or (epoch>2000 and epoch<2500))
        focal_dice_w = int(focal_dice_w>0) * int(epoch%5==0) #unpredictable small object outliers if too frequent
        
        # Increasing BCE weight frequency to prevent dice loss from creating edge artifacts using the g * p numerator
        # Increasing focal loss frequency causes failure in edge detection
        bce_l_w = int(epoch<2000) or int(epoch % 8 == 0)
        fl_l_w = int(epoch>1200 and epoch<2000) + int(epoch%9==0)

        random_multiclass_weight_bool = epoch>early_stop_epoch
       
        running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
        for i, data in enumerate(traindataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, fname = data
            
            # strongest in the pack segmentation only!
            labels = return_union_sets_descending_order(labels)

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
                    losses_fn(outputs, labels, 
                            composite_set_theory=composite_flag, 
                            background_weight=bg_weight, 
                            early_stopped=random_multiclass_weight_bool)
            dice_l = [dice, generalized_dice, twersky_dice, focal_dice]
            
            #TODO: FOCAL LOSS IS WRONG! 
            # focal_dice works great with DeepLabv3 but doesn't as much with resnet34 or resnet50
            loss = bce_l * bce_l_w + focal_dice_w * focal_dice + generalized_dice_w * (generalized_dice + twersky_dice)
            
            loss.backward()
            optimizer.step()
 
            if epoch % 5 == 0:
                torch.save(net.state_dict(), os.path.join(save_dir, "channels%d" % MAXCHANNELS, 
                            "img%d" % IMGSIZE,"%s_epoch%d.pt" % (EXPTNAME, epoch)))
           
            """ # Print when debugging only for faster training!
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

                print("Epoch: %d ; Batch: %d/%d : Training Loss: %.8f" % (epoch+1, i+1, len(traindataloader), running_loss / log_every))
                print ("\t Cross-Entropy: %0.8f; BCE: %.8f; Focal Loss: %0.8f; Dice Loss: %0.8f [D: %.8f, GD: %.8f, TwD: %.8f, FocD: %.8f]" % (
                            ce_t/float(log_every), bce_t/float(log_every), fl_t/float(log_every), sum([x/float(log_every) for x in dice_t]), 
                            dice_t[0]/float(log_every), dice_t[1]/float(log_every), dice_t[2]/float(log_every), dice_t[3]/float(log_every)))
                
                running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
            """

        [dataset.dataset.set_augment_flag(False) for dataset in valdataloader.dataset.datasets]
        with torch.no_grad():
            net = net.eval()
            val_running_loss, ce_t, bce_t, fl_t, dice_t = 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]
            for j, val_data in enumerate(valdataloader, 0):
                val_inputs, val_labels, _ = val_data
                
                val_labels = return_union_sets_descending_order(val_labels)
                
                if torch.cuda.is_available():
                    if isinstance(val_labels, list):
                        val_inputs, val_labels = val_inputs.cuda(), [x.cuda() for x in val_labels]
                    else:
                        val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                
                val_outputs = net(val_inputs)
                val_outputs = F.sigmoid(val_outputs)
                
                """ # Ignoring logging for faster training
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
                """

                # save 5 images per 10 epochs for testing
                if j < 10 and epoch % 10 == 0:
                    
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
                        
                        assert out.sum() > 0, "gradient descent gave no positives! aborting"

                        cv2.imwrite(imgpath+"_img.png", img)
                        cv2.imwrite(imgpath+"_gt_organ%d.png" % idx, gt)
                        cv2.imwrite(imgpath+"_pred_organ%d.png" % idx, out)
            
            """
            num_avg = float(len(valdataloader)*val_inputs.shape[0])
            val_running_loss /= float(num_avg)
            """

        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch + 1)
        else:
            scheduler.step(val_running_loss)
        
        """
        print("\nVal Loss: %.8f!" % val_running_loss)
        print ("\t Cross-Entropy: %0.8f; BCE: %.8f; Focal Loss: %0.8f; Dice Loss: %0.8f [D: %.8f, GD: %.8f, TwD: %.8f, FocD: %.8f]" % (
                    ce_t/float(num_avg), bce_t/float(num_avg), fl_t/float(num_avg), sum([x/float(num_avg) for x in dice_t]), 
                    dice_t[0]/float(num_avg), dice_t[1]/float(num_avg), dice_t[2]/float(num_avg), dice_t[3]/float(num_avg)))
        """

    print('finished training')

#"""
def losses_fn(x, g, composite_set_theory=False, background_weight=0, early_stopped=False):
    
    # Hardcoded subset membership loss for each composite set of organs
    # [whole_body, ventral_side, dorsal_side] = [1.         0.20878016 0.22319692] 
    # Use dataset.get_relative_ratios for every organ subset


    CLASS_INDEX = 1
    if g.shape[CLASS_INDEX] > 1:
        losses = [losses_fn(g[:,idx:idx+1,:,:], x[:,idx:idx+1,:,:]) for idx in range(g.shape[CLASS_INDEX])]

        # direct optimization on target objective
        # superset index 1 depending on subset 2 to get set 1 
        losses[1] = [x + y for x, y in zip(losses_fn(g[:,1:2,:,:] - g[:,2:3,:,:], torch.abs(x[:,1:2,:,:] - x[:,2:3,:,:])), losses[1])]
        
        # Adding composite weights to prevent over-prioritization on supersets during non-superset prediction
        # Increasing loss weight by superset size creates small non-object artefacts in the background
        
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
    
    if composite_set_theory:
        
        whole_body_g, whole_body_p = g[:,0:1,...], x[:,0:1,...]
        ventral_union_g, ventral_union_p = g[:,1:2,...], x[:,1:2,...]
        dorsal_side_g, dorsal_side_p = g[:,2:3,...], x[:,2:3,...]
        
        #TODO: Flip the terms and check for subset-superset dependence in terms of errors
        ventral_side_g, ventral_side_p = torch.abs(ventral_union_g - dorsal_side_g), torch.abs(ventral_union_p - dorsal_side_p) 
        
        # re-weighting directly optimized ventral_side subset causes poor superset prediction
        # poor superset prediction generalizes to subsets in terms of decrease in performance if high lr is used along with epoch alternating focal losses
        # added a factor 2 to compensate for loss in performance - NO EFFECT
        ventral_union_w = 2.4376792669332903 * (1 - int(early_stopped) * np.random.choice([0,1]) * np.random.rand()) #*2

        dorsal_side_w = 4.480348563949717 * (1 - int(early_stopped) * np.random.choice([0,1]) * np.random.rand())
        
        ventral_union_negative_loss = sum(list(losses_fn(ventral_union_g, whole_body_p * ventral_union_p)))
        ventral_side_negative_loss = sum(list(losses_fn(ventral_side_g, whole_body_p * ventral_side_p)))
        ventral_russel_negative_loss = sum(list(losses_fn(ventral_side_g, ventral_union_p * ventral_side_p)))
        dorsal_side_negative_loss = sum(list(losses_fn(dorsal_side_g, whole_body_p * dorsal_side_p)))

        # densenet-like loss introduction
        dorsal_side_union_negative_loss = sum(list(losses_fn(dorsal_side_g, ventral_union_p * dorsal_side_p)))
        
        # set overlap as 0.5 based regularizer: possible generalization to factorized pixel map (1/k for k>2)
        # Overlap contribution set as 0.5 based on 2 classes assumption: possible generalization might involve factors based on size of segment
        # SUSPECTED: ventral_union weight too low for focus on subsets outside whole_body - ventral_union
        ventral_union_positive_loss = sum(list(losses_fn(whole_body_g, \
                                        (whole_body_p * (1 - ventral_union_p) + (whole_body_p * ventral_union_p + ventral_union_p)*0.5))))
        ventral_side_positive_loss = sum(list(losses_fn(whole_body_g, \
                                        (whole_body_p * (1 - ventral_side_p) + (whole_body_p * ventral_side_p + ventral_side_p)*0.5))))
        ventral_russel_positive_loss = sum(list(losses_fn(ventral_side_g, \
                                        (ventral_union_p * (1 - ventral_side_p) + (ventral_union_p * ventral_side_p + ventral_side_p)*0.5))))
        dorsal_side_positive_loss = sum(list(losses_fn(whole_body_g, \
                                        (whole_body_p * (1 - dorsal_side_p) + (whole_body_p * dorsal_side_p + dorsal_side_p)*0.5))))

        # densenet-like loss introduction
        dorsal_side_union_positive_loss = sum(list(losses_fn(ventral_union_g, \
                                        (ventral_union_p * (1 - dorsal_side_p) + (ventral_union_p * dorsal_side_p + dorsal_side_p)*0.5))))
        
        # dorsal_side vs dorsal_ventral_union loss assumed to have union weight (2 vs 4)
        return_losses1 = [l + ventral_side_w * (x+y) + ventral_union_w * (w+z) + 4*r \
                for l,w,x,r,y,z in zip(return_losses, 
                                    ventral_union_negative_loss, ventral_side_negative_loss, ventral_russel_negative_loss,
                                    dorsal_side_negative_loss, dorsal_side_union_negative_loss)] 
        # x + 4.789727146487483 * y Subsets creating gaps in whole_body segment

        # hypothesis: loss enforcing disparate segments hurts performance - looks to be true from visual inspection
        # dorsal_side vs dorsal_ventral_union loss assumed to have union weight (2 vs 4)
        return_losses2 = [l + dorsal_side_w * (2*y+x) + w + 4*ventral_union_w * z + 4*r \
                for l,w,x,r,y,z in zip(return_losses, 
                                        ventral_union_positive_loss, ventral_side_positive_loss, ventral_russel_positive_loss,
                                        dorsal_side_positive_loss, dorsal_side_union_positive_loss)]
        # x + 4.480348563949717 * y Subsets creating gaps in whole_body segment
        
        return_losses = [x + y \
                for x,y in zip(return_losses1, return_losses2)]

    return return_losses
#"""

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
        print ('Loading from ', model_file)
        start_epoch = int(model_file.split("epoch")[-1].split('.')[0])
        if not torch.cuda.is_available():
            load_state = torch.load(model_file, map_location=torch.device('cpu'))
        else:    
            load_state = torch.load(model_file)
        net.load_state_dict(load_state)
        print ("Used latest model file: %s" % model_file)
        
        return start_epoch

    except Exception as e:
         
        if isinstance(e, RuntimeError):
            if isinstance(net, DeepLabV3PlusDepthwise):
                
                from collections import OrderedDict
                try:
                    if not load_state is None:
                        model_layer_dict_with_new_layers = OrderedDict()
                        for key in load_state.keys():
                            if "segmentation_head" in key:
                                break
                            model_layer_dict_with_new_layers["smp_deeplab_model."+key] = load_state[key]
                        load_state = model_layer_dict_with_new_layers
                        
                        net.load_state_dict(load_state, strict=False)
                        return start_epoch
                except Exception:
                    traceback.print_exc()
                    pass
            else:
                print("Did you want to use the --depthwise flag instead?")
                exit()

#        if len(gl) > 0:
#            os.remove(gl[latest_index])
#            print ("Removed incomplete model file: ", gl[latest_index])
#            load_recent_model(saved_dir, net, epoch)

        traceback.print_exc()

        return -1

#unet_model = smp.Unet(
#            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#            classes=1,                      # model output channels (number of classes in your dataset)
#            #activation="silu"              ReLU makes sigmoid more stable # changed from default relu to silu for some resnet50 tests
#        )

#TODO: Layer normalization
def unet_model(depthwise):

    if depthwise: 
        return DeepLabV3PlusDepthwise(  
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(ORGANS),                      # model output channels (number of classes in your dataset)
            #activation="prelu"
            )
    else:
        return smp.DeepLabV3Plus( 
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
    ap.add_argument("--lr", default=0.001, type=float, help="Start training based on amount of predictions>0")
    ap.add_argument("--depthwiseconv", action="store_true", help="Flag to enable depthwise convolution enabling learning multiclass segments separately")
    args = ap.parse_args()
  
    unet_model = unet_model(args.depthwiseconv) 

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
    #optimizer = optim.SGD(unet_model.parameters(), lr=0.01, momentum=0.9)
    
    train(unet_model, train_dataloader, val_dataloader, losses_fn, optimizer, save_dir=saved_dir, start_epoch=start_epoch, 
            log_every = len(train_dataloader) // 5)

