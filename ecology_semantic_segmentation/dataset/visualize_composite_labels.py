import cv2
import numpy as np

from . import CPARTS, colors 

# Pytorch dataset --> cv2 images
def display_composite_annotations(image, labels_map, composite_labels, verbose=True,
        min_positivity_ratio = 0.009, hide_whole_body_segment=False, show_composite_parts=True, return_image=False):
    
    return_images = []
    alpha = 0.8
    
    image = image.transpose((1,2,0)).astype(np.uint8)
    #cv2.imshow("image", image)
    
    if hide_whole_body_segment:
        largest_segment_id = np.argmax(labels_map.sum(axis=(1,2)))

        if composite_labels[largest_segment_id] == "whole_body":
            if verbose:
                print ("\nIgnoring largest segment %s!" % composite_labels[largest_segment_id])
        else:
            if verbose:
                print ("\nCannot find whole body segment!")
            largest_segment_id = -1
    else:
        largest_segment_id = 0

    labels_map = labels_map.transpose((1,2,0)).astype(np.uint8)
    
    outer_loop_times = len(CPARTS) if not return_image and \
                        show_composite_parts and \
                        any([x in composite_labels for y in CPARTS for x in y]) else 1
    
    image_copy = image.copy()

    for outer_loop_idx in range(outer_loop_times):
        
        visited_cparts = []
        
        for seg_id in range(labels_map.shape[-1]):
            
            if -1 in labels_map[:,:,seg_id]:
                print ("Label %s will not be learnt by gradient descent algorithm!" % composite_labels[seg_id])
                continue

            if outer_loop_times > 1:
                 
                try:
                    if subset_ratio_denominator == 1.0:
                        seg_mask_ratio = 1.0 if seg_mask_ratio==0 else seg_mask_ratio
                        subset_ratio_denominator = seg_mask_ratio   
                except NameError:
                    subset_ratio_denominator = 1.0

                if composite_labels[seg_id] not in CPARTS[outer_loop_idx]:
                    continue
                else:
                    seg_mask_ratio = np.sum(labels_map[:,:,seg_id]) / (255.0 * np.prod(labels_map.shape[:2]))
                    seg_mask_ratio = seg_mask_ratio / subset_ratio_denominator
            
                if verbose:
                    print ("%s mask ratio wrt image: %f" % (composite_labels[seg_id] + \
                                                    ("" if "whole_body" == composite_labels[seg_id] else (
                                                    " subset ratio wrt whole_body" if subset_ratio_denominator!=1.0 else "")), 
                                                    seg_mask_ratio))

                    if seg_mask_ratio > min_positivity_ratio:
                        visited_cparts.append(CPARTS[outer_loop_idx].index(composite_labels[seg_id]))
                    else:
                        continue
            
            if not return_image:
                cv2.imshow("fish_%s"%composite_labels[seg_id], labels_map[:,:,seg_id])
            
            seg_image = np.expand_dims(labels_map[:,:,seg_id], axis=-1).repeat(3, axis=-1) * np.array(colors[seg_id]).astype(np.uint8)
            seg_image = cv2.addWeighted(image, 1, seg_image, 1, 1.0)
            image = cv2.addWeighted(image, 1-alpha, seg_image, alpha, 1.0)
        
            if return_image:
                return_images.append({composite_labels[seg_id]: image})
        
        missing_annotation_indices = set(range(len(CPARTS[outer_loop_idx]))) - set(visited_cparts)
        if len(missing_annotation_indices) > 0:
            if verbose:
                print ("Cannot find annotations for %s" % ", ".join([CPARTS[outer_loop_idx][x] for x in missing_annotation_indices])) 
            
            if all([x==y for x, y in zip(sorted(missing_annotation_indices), range(len(CPARTS[outer_loop_idx])))]):
                continue
        
        ann_type =  "all_parts" if outer_loop_times == 1 else \
                        ", ".join(CPARTS[outer_loop_idx])
        if not return_image:
            cv2.imshow("fish_%s"%(ann_type), image)
            cv2.waitKey()
        else:
            return_images.append({ann_type: image})
        
        image = image_copy

    if verbose:
        print ("\n", "."*50, "\n")

    if not return_image:
        cv2.destroyAllWindows()
    else:
        return return_images
