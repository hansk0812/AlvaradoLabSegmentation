# Testing
import glob
import os
import cv2

import numpy as np

import torch
from torch.nn import functional as F

from torch.utils.data import DataLoader

from .dataset.fish import fish_test_dataset
from .train import vgg_unet

from .train import load_recent_model
from .loss_functions import cross_entropy_loss, dice_loss

def tensor_to_cv2(img_batch):
    img_batch = img_batch.numpy().transpose((0,2,3,1))
    img_batch = img_batch[0]
    
    img_batch = cv2.cvtColor(img_batch, cv2.COLOR_RGB2BGR)

    return img_batch

def test(net, dataloader, num_epochs=100, log_every=100, batch_size=8, models_dir="models/vgg", results_dir="test_results/", saved_epoch=-1):
    
    test_dice = [0, 0]
    label_dirs = ["whole_body"]
    try:
        for label_dir in label_dirs:
            os.makedirs(os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), label_dir))
    except Exception:
        pass
    
    with torch.no_grad():
        for j, test_images in enumerate(dataloader, 0):
            
            # Hard-to-read log file
            #print ("Predictions on batch: %d/%d" % (j+1, len(dataloader)), end='\r')
            
            test_images, test_labels, image_ids = test_images

            if torch.cuda.is_available():
                test_images = test_images.cuda()
                test_labels = test_labels.cuda()
            
            test_outputs = F.sigmoid(net(test_images))
            
            test_dice = [test_dice[0] - dice_loss(test_outputs, test_labels), test_dice[1]+1]

            if torch.cuda.is_available():
                test_images = test_images.cpu()
                test_labels = test_labels.cpu()
                test_outputs = test_outputs.cpu()
            
            test_outputs = test_outputs[0][0].round().detach().numpy()
            test_labels = test_labels[0][0].detach().numpy()
            test_images = [tensor_to_cv2(test_images)] #.detach())
            
            for idx, (image, label, pred) in enumerate(zip(test_images, test_labels, test_outputs)):
                
                label_dir = label_dirs[0]
                
                image = (image*255).astype(np.uint8)
                test_labels = (test_labels*255).astype(np.uint8)
                test_outputs = (test_outputs*255).astype(np.uint8)

                img_file = os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), label_dir, "%s.png"% str(j*batch_size + idx).zfill(5))
                
                test_outputs = np.dstack((np.zeros_like(test_outputs), np.zeros_like(test_outputs), test_outputs))
                image1 = cv2.addWeighted(image, 0.6, test_outputs, 0.4, 0)

                test_labels = np.dstack((np.zeros_like(test_labels), test_labels, np.zeros_like(test_labels)))
                image2 = cv2.addWeighted(image, 0.6, test_labels, 0.4, 0)

                image3 = cv2.addWeighted(image, 0.2, image2, 0.8, 0)
                image3 = cv2.addWeighted(image3, 0.2, image1, 0.8, 0)
                
                image1 = cv2.putText(image1, 'Predictions', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                image2 = cv2.putText(image2, 'Ground Truth', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                
                image = np.concatenate((image1, image2, image3), axis=0)
                cv2.imwrite(img_file, image)
        
        dice_loss_val = test_dice[0] / float(test_dice[1])
        print ("Epoch %d: \n\t Test Dice Score: %.5f" % (
            saved_epoch, dice_loss_val))
        print('Finished Testing')

        return dice_loss_val

if __name__ == "__main__":
    
    batch_size = 1

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_model", action="store_true", help="Flag for model selection vs testing entire test set")
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
    if args.single_model:
        saved_epoch = load_recent_model(models_dir, net)
        test_model_files = [x for x in test_model_files if "epoch%d.pt"%saved_epoch in x]

    for model_file in test_model_files:
        try:
            saved_epoch = int(model_file.split('epoch')[-1].split('.pt')[0])
        except Exception:
            continue

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_file))
        else:
            net.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

        with torch.no_grad():
            dice_loss_val = test(net, test_dataloader, models_dir=models_dir, batch_size=batch_size, saved_epoch=saved_epoch)
            test_losses.append([saved_epoch, dice_loss_val])

    for loss in sorted(test_losses, key = lambda x: x[1]):
        print ("Epoch %d : DICE Score %.10f" % (loss[0], loss[1]))
