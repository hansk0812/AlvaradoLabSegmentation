# Testing
import glob
import os
import cv2

import numpy as np

import torch
from torch.nn import functional as F

from torch.utils.data import DataLoader

from .dataset.fish import fish_test_dataset, ORGANS
from .dataset.visualize_composite_labels import display_composite_annotations

from .train_multiclass import vgg_unet
from .train_multiclass import load_recent_model

from .loss_functions import cross_entropy_loss, dice_loss

def tensor_to_cv2(img_batch):
    img_batch = img_batch.numpy().transpose((0,2,3,1))
    img_batch = img_batch[0]
    
    img_batch = cv2.cvtColor(img_batch, cv2.COLOR_RGB2BGR)

    return img_batch

def test(net, dataloader, models_dir="models/vgg", results_dir="test_results/", batch_size=1, saved_epoch=-1):
    
    test_dice = [0, 0]
    label_dirs = ORGANS
    
    dir_name = os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), ",".join(label_dirs))
    try:
        os.makedirs(dir_name)
    except Exception:
        if os.path.isdir(dir_name):
            print ("Skipping epoch %d! Test already done!" % saved_epoch)
            return None
    
    # deeplabv3plus GPU hotfix 
    net = net.eval()

    with torch.no_grad():
        for j, test_images in enumerate(dataloader, 0):
            
            # Hard-to-read log file
            #print ("Predictions on batch: %d/%d" % (j+1, len(dataloader)), end='\r')
            
            test_images, test_labels, image_ids = test_images

            if torch.cuda.is_available():
                test_images = test_images.cuda()
                test_labels = test_labels.cuda()
            
            test_outputs = F.sigmoid(net(test_images))

            CLASS_INDEX = 1
            if test_labels.shape[CLASS_INDEX] > 1:
                loss = sum([dice_loss(test_outputs[:,idx:idx+1,:,:], test_labels[:,idx:idx+1,:,:]) \
                                for idx in range(test_labels.shape[CLASS_INDEX])]) / float(test_labels.shape[CLASS_INDEX])
                
                test_dice = [test_dice[0] - loss, test_dice[1]+1]

            if torch.cuda.is_available():
                test_images = test_images.cpu()
                test_labels = test_labels.cpu()
                test_outputs = test_outputs.cpu()

            test_images = (test_images.numpy() * 255).astype(np.uint8)
            test_labels = (test_labels.numpy() * 255).astype(np.uint8)
            test_outputs = (test_outputs.numpy() * 255).astype(np.uint8)

            preds = display_composite_annotations(test_images[0], test_outputs[0], ORGANS, return_image=True, verbose=False)
            gts = display_composite_annotations(test_images[0], test_labels[0], ORGANS, return_image=True, verbose=False)
            
            img_keys = [list(x.keys())[0] for x in gts]

            for idx, key in enumerate(img_keys):

                cv2.imwrite(os.path.join(dir_name, key+"_%d_gt.png" % j), gts[idx][key])
                cv2.imwrite(os.path.join(dir_name, key+"_%d_pred.png" % j), preds[idx][key])
        
        dice_loss_val = test_dice[0] / float(test_dice[1])
        print ("Epoch %d: \n\t Test Dice Score: %.5f" % (
            saved_epoch, dice_loss_val))
        print('Finished Testing')

        return dice_loss_val

if __name__ == "__main__":
    
    batch_size = 1

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_model", type=int, help="Epoch number for model selection vs testing entire test set", default=None)
    ap.add_argument("--models_dir", default="models/vgg", help="Flag for model selection vs testing entire test set")
    args = ap.parse_args()
    
    [x.dataset.set_augment_flag(False) for x in fish_test_dataset.datasets]
    test_dataloader = DataLoader(fish_test_dataset, shuffle=False, batch_size=batch_size, num_workers=0)
 
    if torch.cuda.is_available():
        net = vgg_unet.cuda()
    else:
        net = vgg_unet

    print ("Using batch size: %d" % batch_size)
    models_dir = args.models_dir
    channels=256
    img=256

    test_losses = []
    test_model_files = glob.glob(
        os.path.join(models_dir, "channels%d" % channels, "img%d" % img,'*'))
    if not args.single_model is None:
        load_recent_model(models_dir, net, args.single_model)
        saved_epoch = args.single_model
        test_model_files = [x for x in test_model_files if "epoch%d.pt"%saved_epoch in x]

    for model_file in test_model_files:
        try:
            saved_epoch = int(model_file.split('epoch')[-1].split('.pt')[0])
        except Exception:
            continue
        
        try:
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(model_file))
            else:
                net.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
        except Exception:
            print ("Skipped epoch %d because of model file incompatibility!" % saved_epoch)
            continue

        with torch.no_grad():
            dice_loss_val = test(net, test_dataloader, models_dir=models_dir, batch_size=batch_size, saved_epoch=saved_epoch)
            if dice_loss_val is None:
                continue

            test_losses.append([saved_epoch, dice_loss_val])

    for loss in sorted(test_losses, key = lambda x: x[1]):
        print ("Epoch %d : DICE Score %.10f" % (loss[0], loss[1]))
