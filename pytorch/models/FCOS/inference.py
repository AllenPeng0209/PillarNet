import torch

import math
from IPython import embed

import numpy as np
#modified the maskrcnn Boxlist to 7d

from second.pytorch.structures.bounding_box import BoxList
from second.pytorch.structures.boxlist_ops import cat_boxlist

# need to confirm rotate_nms operate correctly
from second.pytorch.structures.boxlist_ops import boxlist_rotate_nms
from second.pytorch.structures.boxlist_ops import boxes_fillter
from second.pytorch.core import box_torch_ops





def bbox10d_to_7d_xsort( per_locations, per_box_regression, image_sizes , voxel_size):
    x =  per_locations[:, 0]/5 - (per_box_regression[:,0] + per_box_regression[:, 2] + per_box_regression[:, 4] + per_box_regression[:, 6]) / 4/5
    y = ( per_locations[:, 1]/5 - ((per_box_regression[:,1] + per_box_regression[:, 3] + per_box_regression[:, 5] + per_box_regression[:, 7]) / 4/5) )  - 40
    z = per_box_regression[:, 8]
    point1_x = x + (per_box_regression[:, 0]/5)
    point1_y = y + (per_box_regression[:, 1]/5)
    point2_x = x + (per_box_regression[:, 2]/5)
    point2_y = y + (per_box_regression[:, 3]/5)
    point3_x = x + (per_box_regression[:, 4]/5)
    point3_y = y + (per_box_regression[:, 5]/5)
    point4_x = x + (per_box_regression[:, 6]/5)
    point4_y = y + (per_box_regression[:, 7]/5)
    line1= torch.sqrt((point4_x-point1_x)**2 +(point4_y-point1_y)**2)
    line2= torch.sqrt((point3_x-point2_x)**2 +(point3_y-point2_y)**2)
    w = torch.min(line1,line2)
    l = torch.max(line1,line2)
    h = per_box_regression[:, 9]
    x_axis_is_longer = (line1 > line2).type(torch.cuda.FloatTensor)
    y_axis_is_longer = (line1 < line2).type(torch.cuda.FloatTensor)
    angle = x_axis_is_longer * torch.atan((point4_x-point1_x)/(point4_y-point1_y))+ y_axis_is_longer * torch.atan((point3_x-point2_x)/(point3_y-point2_y))
    bbox_7d = torch.stack([x,y,z,w,l,h,angle],dim=1)
    return bbox_7d


def two_points_angle(point1, point2):
    '''
    point1
    point2 : center
    range : (-pi ~ pi)
                          |   
   (pi-atan(y/x))    -+   |  ++ (atan(y/x)) 
                    ______|______
                          |
                     --   |  +-
   (-pi + atan(y/x)       |    (atan(y/x)) 
'''
    point1_x , point1_y = point1
    point2_x , point2_y = point2
    x_positive_ids = point1_x >= point2_x
    y_positive_ids = point1_y > point2_y
    x_negative_ids = point1_x <= point2_x
    y_negative_ids = point1_y < point2_y
    first_quadrant_ids = (x_positive_ids & y_positive_ids).type(torch.cuda.FloatTensor)
    second_quadrant_ids = (x_negative_ids & y_positive_ids).type(torch.cuda.FloatTensor)
    third_quadrant_ids = (x_negative_ids & y_negative_ids).type(torch.cuda.FloatTensor)
    fourth_quadrant_ids = (x_positive_ids & y_negative_ids).type(torch.cuda.FloatTensor)
    first_quadrant_angle = first_quadrant_ids * (torch.atan((point1_y - point2_y) / (point1_x - point2_x)))
    second_quadrant_angle = second_quadrant_ids * (np.pi + torch.atan((point1_y - point2_y) / (point1_x - point2_x)))
    third_quadrant_angle = third_quadrant_ids * (-np.pi + torch.atan((point1_y - point2_y) / (point1_x - point2_x)))
    fourth_quadrant_angle = fourth_quadrant_ids * (torch.atan((point1_y - point2_y) / (point1_x - point2_x)))
    angle  = first_quadrant_angle + second_quadrant_angle + third_quadrant_angle + fourth_quadrant_angle
    return angle 

def angle_limit(angle):
    bigger = (angle > np.pi )
    smaller = (angle < -np.pi)
    within =  1 -(bigger | smaller)
    angle = (bigger.type(torch.cuda.FloatTensor)*(angle-2*np.pi)) + (within.type(torch.cuda.FloatTensor)*angle) + (smaller.type(torch.cuda.FloatTensor)*(angle+2*np.pi))
    return angle

def bbox10d_to_7d_direction( per_locations, per_box_regression, image_sizes , voxel_size):
    '''
    this function assume points is sort by x axis 
    '''
    scaling_ration = 1/ voxel_size[0][0]
    y_offset = image_sizes[0][1]*voxel_size[0][0]/2
    x =  per_locations[:, 0]/scaling_ration - (per_box_regression[:,0] + per_box_regression[:, 2] + per_box_regression[:, 4] + per_box_regression[:, 6]) / 4/ scaling_ration
    y = ( per_locations[:, 1]/scaling_ration - ((per_box_regression[:,1] + per_box_regression[:, 3] + per_box_regression[:, 5] + per_box_regression[:, 7]) / 4/ scaling_ration) )  - y_offset
    z = per_box_regression[:, 8] / 10
    point1_x =  per_locations[:, 0]/scaling_ration - (per_box_regression[:, 0]/scaling_ration)
    point1_y = per_locations[:, 1]/scaling_ration-y_offset - (per_box_regression[:, 1]/scaling_ration)
    point2_x =  per_locations[:, 0]/scaling_ration - (per_box_regression[:, 2]/scaling_ration)
    point2_y = per_locations[:, 1]/scaling_ration-y_offset  - (per_box_regression[:, 3]/scaling_ration)
    point3_x =  per_locations[:, 0]/scaling_ration - (per_box_regression[:, 4]/scaling_ration)
    point3_y = per_locations[:, 1]/scaling_ration-y_offset  - (per_box_regression[:, 5]/scaling_ration)
    point4_x =  per_locations[:, 0]/scaling_ration - (per_box_regression[:, 6]/scaling_ration)
    point4_y = per_locations[:, 1]/scaling_ration-y_offset  - (per_box_regression[:, 7]/scaling_ration)
    w= torch.sqrt((point3_x-point2_x)**2 +(point3_y-point2_y)**2)
    l= torch.sqrt((point4_x-point1_x)**2 +(point4_y-point1_y)**2) 
    h = per_box_regression[:, 9]/ 10 
    #angle = two_points_angle((point1_x,point1_y), (x,y))i

    cross_angle=True
    if cross_angle:
        angle_1 = two_points_angle((point1_x,point1_y),(point4_x,point4_y))   
        angle_2 = angle_limit(two_points_angle((point2_x,point2_y),(point3_x,point3_y))  +np.pi/2)
        angle = (angle_1+angle_2)/2
    two_angle = False
    if two_angle ==True:
      angle_1 =  two_points_angle((point1_x,point1_y), (x,y))
      angle_4 =  two_points_angle((x,y),(point4_x,point4_y))
      angle = (angle_1+angle_4) / 2
    multi_angle = False
    if multi_angle == True :
        angle_1 =  two_points_angle((point1_x,point1_y), (x,y))
        angle_2 =  angle_limit(two_points_angle((point2_x,point2_y),(x,y)) + np.pi/2)
        angle_3 =  angle_limit(two_points_angle((point3_x,point3_y),(x,y)) - np.pi/2)
        #angle_2 =  two_points_angle((point2_x,point2_y),(x,y)) 
        #angle_3 =  two_points_angle((point3_x,point3_y),(x,y)) 
        angle_4 =  two_points_angle((x,y),(point4_x,point4_y))
        angle = (angle_1+angle_2+angle_3 +angle_4) / 4

    
    # print("check box tranform")
  
    
    bbox_7d = torch.stack([x,y,z,w,l,h,angle],dim=1)

    return bbox_7d

 
def compute_centerness(per_locations ,per_box_regression, image_sizes , voxel_size):
    scaling_ration = 1/ voxel_size[0][0]
    y_offset = image_sizes[0][1]*voxel_size[0][0]/2

    x =  per_locations[:, 0]/scaling_ration - (per_box_regression[:,0] + per_box_regression[:, 2] + per_box_regression[:, 4] + per_box_regression[:, 6]) / 4/scaling_ration
    y = ( per_locations[:, 1]/scaling_ration - ((per_box_regression[:,1] + per_box_regression[:, 3] + per_box_regression[:, 5] + per_box_regression[:, 7]) / 4/scaling_ration) )  - y_offset
    distance_to_center = torch.sqrt((per_locations[:, 0]/scaling_ration - x)**2 + (per_locations[:, 1]/scaling_ration-40 - y)**2)
    #it's hard to calculate the longest distance in the box and divide it to normalize, cause we may regress the wrong bbox
    #so it's better to just use relative nomalization method like below
    distance_to_center_normalize = (distance_to_center-distance_to_center.min())/(distance_to_center.max()-distance_to_center.min())  
    centerness_score = 1 - distance_to_center_normalize
    '''
    point1_distance = torch.sqrt(per_box_regression[:,0]**2 + per_box_regression[:,1]**2)    
    point2_distance = torch.sqrt(per_box_regression[:,2]**2 + per_box_regression[:,3]**2)
    point3_distance = torch.sqrt(per_box_regression[:,4]**2 + per_box_regression[:,5]**2)
    point4_distance = torch.sqrt(per_box_regression[:,6]**2 + per_box_regression[:,7]**2)
    centerness_score = (torch.min(point1_distance,point4_distance) /torch.max(point1_distance,point4_distance)) * (torch.min(point2_distance,point3_distance) /torch.max(point2_distance,point3_distance))
    '''
    return centerness_score







class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.

    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, box_score,
            image_sizes, voxel_size):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape
        # from 10d to 7d  -> get x,y,z,w,l,h,zr
        #box_regression =  bbox_frame_middle_points_to_center_location(box_regression)


        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_score  = box_score.view(N, 2, H, W).permute(0, 2, 3, 1)
        box_score  = box_score.reshape(N, -1,2).sigmoid()
        box_regression = box_regression.view(N,10, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 10)
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        #multiply the classification score with box_score
    
        # four option , 
        box_cls =  box_cls
        #box_cls = box_cls * centerness_score
        #box_cls = box_cls * box_score[:, :, None]
        #box_cls = box_cls  * centerness_score * box_score[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_box_score = box_score[i][per_box_loc]
            per_locations = locations[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
        
            #TODO add config here
            centerness_mode = False
            # add centerness score by directly computer through box_reg 
            if centerness_mode ==True:
                centerness_score = compute_centerness(per_locations ,per_box_regression, image_sizes , voxel_size)
                per_box_cls = per_box_cls * centerness_score
            # method2 use below  =>  remake the the 7d by points_distance
            #resize the pred_box_regress first
            #per_box_regression = bbox_resize(per_box_regression, image_sizes, voxel_size)
            #bbox_7d = bbox10d_to_7d_xsort(per_locations ,per_box_regression, image_sizes , voxel_size)
            # if train with direction use below
            bbox_7d = bbox10d_to_7d_direction(per_locations ,per_box_regression, image_sizes , voxel_size)
            detections = torch.stack([
                bbox_7d[:,0],
                bbox_7d[:,1],
                bbox_7d[:,2],
                bbox_7d[:,3],
                bbox_7d[:,4],
                bbox_7d[:,5],
                bbox_7d[:,6]
            ], dim=1)

            h, w = image_sizes[0][0:2]

         
            boxlist = BoxList(detections, (int(w), int(h)))
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_score)
            # To do : write the 7d box_op method to do these
            #boxlist = boxlist.clip_to_image(remove_empty=False)
            #print("check fillter") 
            #embed()
            
            #TODO pass the class_name sheet,add condition to filter each class size
            #boxlist = boxes_fillter(boxlist)
            
            results.append(boxlist)
           
        return results

    def forward(self, locations, box_cls, box_regression, box_score,targets, image_sizes, voxel_size):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (l, o, b ,s) in enumerate(zip(locations, box_cls, box_regression, box_score)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b ,s, image_sizes,voxel_size
                )
            )
        
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        
        #print('check the boxlist before nms')
        #embed()
        boxlists = self.select_over_all_levels(boxlists)
        #print('check the boxlist after nms')
        #embed()
        #print('check the boxlist after nms')
        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.

    # based on voxelnet.predict method, modulize some function into Boxlist and box_ops in the future
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 7)
                boxlist_for_class = BoxList(boxes_j, boxlist.size)
                boxlist_for_class.add_field("scores", scores_j)
               


                boxlist_for_class = boxlist_rotate_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores",
                    merge_nms=True
                )
                num_labels = len(boxlist_for_class)

                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)
            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(num_classes):
    pre_nms_thresh = 0.50
    pre_nms_top_n = 1000
    nms_thresh = 0.01
    fpn_post_nms_top_n = 30

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
	num_classes= num_classes
    )
    # notice the above, background counts 1 class, if we only detect car , num_classes should be 2

    return box_selector
