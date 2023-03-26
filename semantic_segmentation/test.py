# Testing

from .dataset.fish import fish_test_dataset
from .model import vgg_unet

import torch

from torch.utils.data import DataLoader

from .train import load_recent_model

import csv

import os
import cv2

import numpy as np

def tensor_to_cv2(img_batch):
    img_batch = img_batch.numpy().transpose((0,2,3,1))

    for idx in range(img_batch.shape[0]):
        img_batch[idx] = cv2.cvtColor(img_batch[idx], cv2.COLOR_RGB2BGR)

    return img_batch

def test(net, dataloader, num_epochs=100, log_every=100, batch_size=8, models_dir="models/scratch", results_dir="test_results/"):

    saved_epoch = load_recent_model(models_dir, net)
 
    label_dirs = ["whole_body"]
    try:
        for label_dir in label_dirs:
            os.makedirs(os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), label_dir))
    except Exception:
        pass
    
    for j, test_images in enumerate(dataloader, 0):

        print ("Predictions on image batch: %d/%d" % (j+1, len(dataloader)), end='\r')
        
        test_images, test_labels, image_ids = test_images

        test_images = test_images.cuda()
        test_outputs = net(test_images)
        
        test_outputs = test_outputs.numpy()
        test_labels = list(np.argmax(test_outputs, axis=-1)[:,0]) #.detach().numpy())
        test_images = tensor_to_cv2(test_images) #.detach())
        
        for idx, (image, label, img_id) in enumerate(zip(test_images, test_labels, image_ids)):
            
            label_prob = abs((1-label) - test_outputs[idx, 0, label])
            results_csv.append([img_id, label_prob])
            label_dir = label_dirs[label]
            image = (image*255).astype(np.uint8)
            
            img_file = os.path.join(results_dir, "%s"%str(saved_epoch).zfill(4), label_dir, "%s.png"% str(j*batch_size + idx).zfill(5)) 
            cv2.imwrite(img_file, image)
    
    print ()
    with open(os.path.join(results_dir, "%s.csv"%str(saved_epoch).zfill(4)), 'w', newline='') as f:
        W = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for result in results_csv:
            W.writerow(result)

    print('Finished Testing')

if __name__ == "__main__":

    batch_size = 8
    test_dataloader = DataLoader(fish_test_dataset, shuffle=False, batch_size=batch_size, num_workers=0)

    net = vgg_unet.cuda()

    print ("Using batch size: %d" % batch_size)

    with torch.no_grad():
        test(net, test_dataloader, models_dir="models/scratch", batch_size=batch_size)
