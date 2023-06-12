import torch

class LossList(list):
    def __iadd__(self, other_list):
        assert len(self) == len(other_list), "Lists to be added need the same length! (%d vs %d)" % (len(self), len(other_list))
        return [x+y for x,y in zip(self, other_list)] # returning regular list because += is used only once 
    
    def __mul__(self, w):
        assert isinstance(w, float), "Multiplication supported for numerical weights only! Found %s" % type(w)
        return LossList([x*w for x in self])

# Use dataset.get_relative_ratios for every organ subset
# relative_organ_ratios: [whole_body, ventral_side+dorsal_side, dorsal_side]
def losses_fn(x, g, composite_set_theory=False, background_weight=0, early_stopped=False, relative_set_ratios=[1., 0.43197708, 0.22319692]):
    
    CLASS_INDEX = 1
    
    assert x.shape[CLASS_INDEX] == len(relative_organ_ratios), "Organ ratios size mismatch!"

    if g.shape[CLASS_INDEX] > 1:
        losses = [losses_fn(g[:,idx:idx+1,:,:], x[:,idx:idx+1,:,:]) for idx in range(g.shape[CLASS_INDEX])]
        return LossList([sum(i) for i in zip(*losses)]) # /float(g.shape[CLASS_INDEX]) Using sum loss for now
    
    bce_loss = cross_entropy_loss(x, g, bce=True, background_weight=background_weight)
    ce_loss, fl_loss = cross_entropy_loss(x, g, bce=False, background_weight=background_weight), \
            focal_loss(x, g, factor=1, background_weight=background_weight)

    dice, generalized_dice, twersky_dice, focal_dice = classification_dice_loss(x, g, factor=10, background_weight=background_weight)
    
    return_losses = LossList([ce_loss, bce_loss, fl_loss, dice, generalized_dice, twersky_dice, focal_dice])
    return_losses += return_losses # 2*losses for union and intersection loss based reweighting
    
    if composite_set_theory:
        
        LENGTH = len(g.shape[CLASS_INDEX])
        # For every superset idx and corresponding subset jdx such that #classes in (idx-jdx) >= 1
        for idx in range(LENGTH-1):
            for jdx in range(idx+1, LENGTH):

                w_idx = (1/relative_set_ratios[idx]) * (1 - int(early_stopped) * np.random.choice([0,1]) * np.random.rand())
                w_jdx = (1/relative_set_ratios[jdx]) * (1 - int(early_stopped) * np.random.choice([0,1]) * np.random.rand())
                w_diff = (1/(relative_set_ratios[idx]-relative_set_ratios[jdx])) * \
                        (1 - int(early_stopped) * np.random.choice([0,1]) * np.random.rand())

                # Rigid subset assumption for every composite set
                # INTERSECTION
                return_losses += intersection_loss(x[:,idx,...], x[:,jdx,...], g[:,jdx,...]) * w_jdx
                
                # NUMERICALLY REGULARIZED UNION
                return_losses += union_loss(x[:,idx,...], x[:,jdx,...], g[:,idx,...]) * w_idx
                
                # Per set losses calculated wrt every other subset
                # INTERSECTION
                return_losses += intersection_loss(x[:,idx,...], 
                                                    torch.abs(x[:,idx,...]-x[:,jdx,...]), 
                                                    torch.abs(g[:,idx,...]-g[:,jdx,...])) * w_diff
                
                # NUMERICALLY REGULARIZED UNION
                return_losses += union_loss(x[:,idx,...], 
                                            torch.abs(x[:,idx,...]-x[:,jdx,...]), 
                                            g[:,idx,...]) * w_idx
                
                # Russel's paradox losses
                # INTERSECTION
                return_losses += intersection_loss(x[:,idx,...], 
                                                    torch.abs(x[:,idx,...]-x[:,jdx,...]) * x[:,idx,...], 
                                                    torch.abs(g[:,idx,...]-g[:,jdx,...])) * w_diff

                # NUMERICALLY REGULARIZED UNION
                return_losses += union_loss(x[:,idx,...], 
                                            torch.abs(x[:,idx,...]-x[:,jdx,...]) * x[:,idx,...], 
                                            g[:,idx,...]) * w_idx * w_idx * w_jdx 
                # larger weight to highlight absence of separate class for missing subsets
    
    return return_losses

# by defn, superset_g * set_g = set_g
def insersection_loss(superset_p, set_p, set_g):
    return sum(list(losses_fn(superset_p * set_p, set_g)))

#TODO: per superset n-ary loss regularization with factors 1/(n-1)
# by defn, superset_g * (1-set_g) + 0.5*(superset_g * set_g + set_g) = superset_g
def union_loss(superset_p, set_p, superset_g):
    return sum(list(losses_fn(superset_g, \
                    (superset_p * (1 - set_p) + (superset_p * set_p + set_p)*0.5))))
