import torch
from torch.nn import functional as F

class SegmentationLoss:
    def __init__(self, eta=1e-7):
        self.eta = eta

    def cross_entropy(self, ground_truth, predicted_segment):
        # w = (N-p)/N
        # l = -(1/N) \sum_i=1^N w*gt*\log(p) + (1-gt)*\log(1-p)

        N = torch.prod(torch.tensor(ground_truth.shape))
        weight = None #(N - predicted_segment.sum()) / N
        
        ce = F.binary_cross_entropy(ground_truth, predicted_segment, weight)
        #ce = ((-1/N) * torch.sum(weight * ground_truth * torch.log(predicted_segment + self.eta) + \
        #                        (1-weight) * (1-ground_truth) * torch.log(1 - predicted_segment + self.eta)))
        #print (torch.log(predicted_segment + self.eta))
        #print (torch.log(1 - predicted_segment + self.eta))

        return ce

    def dice_score_loss(self, ground_truth, predicted_segment):
        # dc = (\sum p*gt + \eta) / (\sum p + \sum gt + \eta) + \
        #       (\sum (1-p)*(1-gt) + \eta) / ((\sum (1-p) + \sum (1-gt)) + \eta)
        # dice_loss = 1 - dc
        
        G1, P1 = ground_truth, predicted_segment
        G0, P0 = (1-ground_truth), (1-predicted_segment)

        dice_coeff_preds_fg = torch.sum(G1 * P1) + self.eta
        dice_coeff_preds_bg = torch.sum(G0 * P0) + self.eta
        dice_coeff_normalize_fg = torch.sum(G1) + torch.sum(P1) + self.eta
        dice_coeff_normalize_bg = torch.sum(G0) + torch.sum(P0) + self.eta
        
        dc = dice_coeff_preds_fg / dice_coeff_normalize_fg + dice_coeff_preds_bg / dice_coeff_normalize_bg

        return 1 - dc

if __name__ == "__main__":
    
    img_shape = (1,3,512,512)

    loss_fns = SegmentationLoss()
    
    print (F.__file__)

    y = torch.rand(img_shape)
    p = torch.rand(img_shape)
    print (loss_fns.weighted_cross_entropy(y, p))
    print (loss_fns.dice_score_loss(y, p))

