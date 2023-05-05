from torch.utils.data import Dataset

class SegmentColorDistribution(Dataset):

    def __init__(self, palette_file, dataset, img_shape=256):

        self.dataset = dataset

        self.palette_image = cv2.imread(palette_file)
        self.palette_image = cv2.resize(self.palette_image, (img_shape, img_shape))
