import cv2
import numpy as np

from torch.utils.data import Dataset

class SegmentColorDistribution(Dataset):

    COLOR_PALETTE_IMG = "ecology_semantic_segmentation/dataset/resources/palette.png" 
    COLOR_PALETTE_RANGES = "ecology_semantic_segmentation/dataset/resources/color_palette.txt"

    def __init__(self, 
                 dataset, 
                 img_shape=256):

        self.dataset = dataset

        self.palette_image = cv2.imread(self.COLOR_PALETTE_IMG)
        self.palette_image = cv2.resize(self.palette_image, (img_shape, img_shape))

        # Use standard color palette for CNN experiments with color
        with open(self.COLOR_PALETTE_RANGES, 'r') as f:
            color_ranges = [x for x in f.readlines() if "(" in x]
            
            self.color_palette = []
            for x in color_ranges:

                color_name = x.split('(')[1].split(',')[0].replace(")\n", "")
                color_range = ','.join(x.split(',')[1:])[:-2]
                
                #numpy fromstring compatible
                color_range = color_range.replace('(','').replace(')','')
                
                if color_range == "":
                    color_range = None
                else:
                    color_range = np.array([int(x) for x in color_range.split(',')]).reshape((2,3))
                self.color_palette.append({"color_name": color_name, "color_range": color_range})
            
            self.color_palette = sorted(self.color_palette, key = lambda x: x["color_name"])
            self.colors = [x["color_name"] for x in self.color_palette]

            self.show_colors(self.colors)

    def show_colors(self, colors):

        assert all([c in self.colors for c in colors]), "Trying to remove unavailable color!"
        
        return_image = cv2.cvtColor(self.palette_image, cv2.COLOR_BGR2HSV)

        for idx, color in enumerate(colors):
            if self.color_palette[idx]["color_range"] is None:
                continue
            color_mask = cv2.inRange(return_image, *self.color_palette[idx]["color_range"]) 
            color_mask[color_mask>0] = 1
            black_image = np.ones_like(self.palette_image) * 255
            black_image[:,:,0] = color_mask * self.palette_image[:,:,0]
            black_image[:,:,1] = color_mask * self.palette_image[:,:,1]
            black_image[:,:,2] = color_mask * self.palette_image[:,:,2]
            

            cv2.imshow('f', self.palette_image)
            cv2.imshow('g', black_image)
            cv2.waitKey()

if __name__ == "__main__":

    from .fish import fish_train_dataset
    obj = SegmentColorDistribution(fish_train_dataset)

