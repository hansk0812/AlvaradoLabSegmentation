import cv2
import numpy as np

from torch.utils.data import Dataset

from .fish import fish_train_dataset

class SegmentColorDistribution(Dataset):
    
    BACKGROUND_CLASS = "background"
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
                    color_range = np.array([int(x) for x in color_range.split(',')]).reshape((-1,3))
                    color_range = [color_range[idx:idx+2] for idx in range(0, len(color_range), 2)]
                self.color_palette.append({"color_name": color_name, "color_range": color_range})
            
            self.color_palette = sorted(self.color_palette, key = lambda x: x["color_name"])

            self.colors = [x["color_name"] for x in self.color_palette if x["color_name"] != self.BACKGROUND_CLASS]
            index = 0
            for idx, color_idx in enumerate(range(6, 25, 6)):
                self.show_colors(self.colors[index:color_idx])
                index += 6

    def show_colors(self, colors):
        
        print ("Using colors:", colors)
        assert all([c in self.colors for c in colors]), "Trying to remove unavailable color!"
        
        return_image = cv2.cvtColor(self.palette_image, cv2.COLOR_BGR2HSV)

        white_image = np.ones_like(self.palette_image)*255
        for idx, color in enumerate(colors):
            color_range = self.color_palette[self.colors.index(color)]["color_range"]
            if not color_range is None:
                for palette_range in color_range:
                    color_mask = cv2.inRange(return_image, *palette_range) 
                    white_image[color_mask>0] = self.palette_image[color_mask>0]

        cv2.imshow('f', self.palette_image)
        cv2.imshow('h', white_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def pixel_inrange_boolean(self, pixel, color_ranges):
        
        inside_range = False
        min_distance = 99999

        means = [np.mean(x, axis=0) for x in color_ranges]
        for mean in means:
            min_distance = min(min_distance, np.sum(np.abs(mean - pixel)))  
        return min_distance

        for color_range in color_ranges:
            minrange, maxrange = color_range[0], color_range[1]
            inrange = True
            for imgch in range(3):
                if not (pixel[imgch] > minrange[imgch] and pixel[imgch] < maxrange[imgch]):
                    inrange = False
            if inrange:
                return True
        return False

    def find_color_distribution(self, image, mask):
        
        mask = mask[:,:,0]
        image[mask==0] = 0
        color_counts = [0 for _ in range(len(self.colors))]
        
        """
        for idx, color in enumerate(self.colors):
            color_range = self.color_palette[self.colors.index(color)]["color_range"]
            if not color_range is None:
                for ranges in color_range:
                    cv2.imshow(color, cv2.inRange(image, *ranges))
                    cv2.waitKey()
        exit()
        """
        color_counts = [0 for _ in range(len(self.colors))]
        indices = np.nonzero(mask)
        for mdx, ndx in zip(*indices):
            distances = [1e5 for _ in range(len(self.colors))]
            for idx, color in enumerate(self.colors):
                color_range = self.color_palette[self.colors.index(color)]["color_range"]
                if not color_range is None:
                    distances[idx] = min(distances[idx], self.pixel_inrange_boolean(image[mdx, ndx], color_range))
            color_counts[np.argmin(distances)] += 1
            #print (self.colors[np.argmin(distances)], image[mdx, ndx])
        print (sorted(zip(color_counts, self.colors), key = lambda x: x[0]))
        """
        indices = np.nonzero(mask)
        for mdx, ndx in zip(*indices):
            for idx, color in enumerate(self.colors):
                color_range = self.color_palette[self.colors.index(color)]["color_range"]
                if not color_range is None:
                    if self.pixel_inrange_boolean(image[ndx, mdx], color_range):
                        color_counts[idx] += 1
                        print (mdx, ndx, color)
                        break
        """
        print (color_counts)
        cv2.imshow('f', cv2.cvtColor(image, cv2.COLOR_HSV2BGR))
        cv2.imshow('g', mask)
        cv2.waitKey()

if __name__ == "__main__":

    from .fish import fish_val_dataset
    obj = SegmentColorDistribution(fish_val_dataset)
    
    for img, mask, fname in fish_val_dataset:
        img, mask = (img.transpose((1,2,0))*255).astype(np.uint8), (mask.transpose((1,2,0))).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        obj.find_color_distribution(img, mask)
