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

from .train_multiclass import unet_model 
from .train_multiclass import load_recent_model

from .train_multiclass_sequential_densenetloss import return_union_sets_descending_order

from .loss_functions import cross_entropy_loss, dice_loss

def tensor_to_cv2(img_batch):
    img_batch = img_batch.numpy().transpose((0,2,3,1))
    img_batch = img_batch[0]
    
    img_batch = cv2.cvtColor(img_batch, cv2.COLOR_RGB2BGR)

    return img_batch

def test(net, dataloader, models_dir="models/vgg", results_dir="test_results/", batch_size=1, saved_epoch=-1, single_model=False):
    
    test_dice = [[0 for _ in range(len(ORGANS))], 0]
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
            
            # idx 0: superset - whole body
            # idx -1: easiest organ to segment
            test_outputs = return_union_sets_descending_order(test_outputs, reverse=True)

            #TODO: Beam search to determine over-expression: per-model hyperparameter vs accuracy!
            # Pending
            # Adding union and intersection losses created the need for this test! 
            # Ignored for other models before ablation and deleted because of lack of space
            # Will not be included for performance benchmarks but a similarity in the pattern can be expected
            BEAM_SEARCH_THRESHOLDS = [] #np.arange(0.8, 0.99, step=0.01)
            if args.single_model:
                avg_losses = []
                for threshold in BEAM_SEARCH_THRESHOLDS:
                    test_outputs[test_outputs > BEAM_SEARCH_THRESHOLD] = 1
                    test_outputs[test_outputs!=1] = 0

                    CLASS_INDEX = 1
                    loss = [dice_loss(test_outputs[:,idx:idx+1,:,:], test_labels[:,idx:idx+1,:,:], background_weight=0) \
                                    for idx in range(test_labels.shape[CLASS_INDEX])]
                    avg_losses.append(loss)
                
                if len(BEAM_SEARCH_THRESHOLDS) > 0:
                    best_idx = np.argmin(np.mean(avg_losses, axis=0))
                    print ("Best performance using threshold: %.3f" % BEAM_SEARCH_THRESHOLDS[best_idx])
                    print ("Accuracy:", avg_losses[best_idx])
                
            CLASS_INDEX = 1
            loss = [dice_loss(test_outputs[:,idx:idx+1,:,:], test_labels[:,idx:idx+1,:,:], background_weight=0) \
                            for idx in range(test_labels.shape[CLASS_INDEX])]
            test_dice = [[x - l for x, l in zip(test_dice[0], loss)], test_dice[1]+1]
            
            if args.single_model:
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
        
        dice_loss_val = torch.tensor(test_dice[0]) / float(test_dice[1])
        print ("Epoch %d: \n\t Test Dice Score: " % saved_epoch, dice_loss_val)
        print('Finished Testing')

        return dice_loss_val

if __name__ == "__main__":
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_model", type=int, help="Epoch number for model selection vs testing entire test set", default=None)
    ap.add_argument("--models_dir", default="models/vgg", help="Flag for model selection vs testing entire test set")
    args = ap.parse_args()
    
    batch_size = 1 if not torch.cuda.is_available() or args.single_model else 45

    [x.dataset.set_augment_flag(False) for x in fish_test_dataset.datasets]
    test_dataloader = DataLoader(fish_test_dataset, shuffle=False, batch_size=batch_size, num_workers=6)
 
    if torch.cuda.is_available():
        net = unet_model.cuda()
    else:
        net = unet_model

    print ("Using batch size: %d" % batch_size)

    try:
        exp_models = os.environ["EXPTNAME"]
        models_dir = os.path.join("models", exp_models)
    except Exception:
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
            dice_loss_val = test(net, test_dataloader, models_dir=models_dir, batch_size=batch_size, \
                                saved_epoch=saved_epoch, single_model=bool(args.single_model))
            if dice_loss_val is None:
                continue

            test_losses.append([saved_epoch, dice_loss_val])
    
    for organ_idx in range(len(ORGANS)):
        for loss in sorted(test_losses, key = lambda x: x[1][organ_idx]):
            print ("Epoch %d : Organ : %s DICE Score " % (loss[0], ORGANS[organ_idx]), loss[1][organ_idx])
