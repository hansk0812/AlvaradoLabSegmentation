# Testing
import glob
import os
import cv2

import numpy as np

import torch
from torch.nn import functional as F

from torch.utils.data import DataLoader

from .dataset.fish import fish_test_dataset
from .model import vgg_unet

from .train import load_recent_model
from .loss_functions import cross_entropy_loss, dice_loss

def tensor_to_cv2(img_batch):
    img_batch = img_batch.numpy().transpose((0,2,3,1))

    for idx in range(img_batch.shape[0]):
        img_batch[idx] = cv2.cvtColor(img_batch[idx], cv2.COLOR_RGB2BGR)

    return img_batch

def test(net, dataloader, num_epochs=100, log_every=100, batch_size=8, models_dir="models/vgg", results_dir="test_results/", saved_epoch=-1):
    
    test_loss = {"bce": [0, 0], "dice": [0, 0]}

    label_dirs = ["whole_body"]
    try:
        for label_dir in label_dirs:
            os.makedirs(os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), label_dir))
    except Exception:
        pass
    
    with torch.no_grad():
        for j, test_images in enumerate(dataloader, 0):

            print ("Predictions on batch: %d/%d" % (j+1, len(dataloader)), end='\r')
            
            test_images, test_labels, image_ids = test_images

            if torch.cuda.is_available():
                test_images = test_images.cuda()
            
            test_outputs = F.sigmoid(net(test_images))
            
            test_loss["bce"] = [cross_entropy_loss(test_outputs, test_labels, bce=True), test_loss["bce"][1]+1]

            test_loss["dice"] = [dice_loss(test_outputs, test_labels), test_loss["dice"][1]+1]

            if torch.cuda.is_available():
                test_images = test_images.cpu()
                test_outputs = test_outputs.cpu()
            
            test_outputs = test_outputs[0][0].round().detach().numpy()
            test_labels = test_labels[0][0].detach().numpy()
            test_images = tensor_to_cv2(test_images) #.detach())
            
            for idx, (image, label, pred) in enumerate(zip(test_images, test_labels, test_outputs)):
                
                label_dir = label_dirs[0]
                
                image = (image*255).astype(np.uint8)
                test_labels = (test_labels*255).astype(np.uint8)
                test_outputs = (test_outputs*255).astype(np.uint8)

                img_file = os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), label_dir, "%s_img.png"% str(j*batch_size + idx).zfill(5))
                gt_file = os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), label_dir, "%s_gt.png"% str(j*batch_size + idx).zfill(5))
                pred_file = os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), label_dir, "%s_pred.png"% str(j*batch_size + idx).zfill(5))
                cv2.imwrite(img_file, image)
                cv2.imwrite(gt_file, test_labels)
                cv2.imwrite(pred_file, test_outputs)
        
        bce_loss_val = test_loss["bce"][0] / float(test_loss["bce"][1])
        dice_loss_val = test_loss["dice"][0] / float(test_loss["dice"][1])
        print ("Epoch %d Test Loss: \n\tBCE Loss: %.5f; Dice Loss: %.5f" % (
            saved_epoch, bce_loss_val, dice_loss_val))
        print('Finished Testing')

        return bce_loss_val, dice_loss_val

if __name__ == "__main__":

    batch_size = 1
    [x.dataset.set_augment_flag(False) for x in fish_test_dataset.datasets]
    test_dataloader = DataLoader(fish_test_dataset, shuffle=False, batch_size=batch_size, num_workers=0)
 
    import segmentation_models_pytorch as smp
    vgg_unet = smp.Unet(
                encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )


    if torch.cuda.is_available():
        net = vgg_unet.cuda()
    else:
        net = vgg_unet

    #saved_epoch = load_recent_model(models_dir, net)
    print ("Using batch size: %d" % batch_size)
    models_dir = "models/vgg"
    channels=256
    img=256

    test_losses = []
    for model_file in glob.glob(
            os.path.join(models_dir, "channels%d" % channels, "img%d" % img,'*')):
        saved_epoch = int(model_file.split('epoch')[-1].split('.pt')[0])

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_file))
        else:
            net.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

        with torch.no_grad():
            bce_loss_val, dice_loss_val = test(net, test_dataloader, models_dir=models_dir, batch_size=batch_size, saved_epoch=saved_epoch)
            test_losses.append([saved_epoch, bce_loss_val])

    for loss in sorted(test_losses, key = lambda x: x[1]):
        print ("Epoch %d : Loss %.10f" % (loss[0], loss[1]))
