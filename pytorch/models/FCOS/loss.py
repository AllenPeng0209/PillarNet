"""
This file contains specific functions for computing losses of FCOS
file

To do : merge the cfg 

"""

import torch
from torch.nn import functional as F
from torch import nn
import math
from second.pytorch.core.fcos_loss import IOULoss, FCOS_resgression_Smooth_L1Loss 
import numpy as np
from second.utils.eval import bev_box_overlap , box3d_overlap
from second.pytorch.core.focal_loss import FocalLoss
from second.core.box_np_ops import riou_cc,riou3d

# this is the original FCOS sigmoidFocalLoss , seems there is some problem in C++ code
from second.pytorch.core.sigmoid_focal_loss import SigmoidFocalLoss
from IPython import embed
import cv2
from PIL import Image
import time
INF = 100000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self):
        # TODO : pass the cfg here  1 gamma 2 alpha

        self.cls_loss_func =SigmoidFocalLoss(2, 0.25)
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = FCOS_resgression_Smooth_L1Loss()
        self.box_score_loss_func = torch.nn.BCEWithLogitsLoss()

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, INF],
        ]

        '''
        #this is for fpn head, which have 5 branch

        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        '''
        
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level =  points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1))
        
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first



    def bbox_to_four_frame_middle_points(self,bboxes_7d, grid_size ,pc_range):
        '''
        we know use x axis to sort the point , if the same then sort y
        the better way to do this is to use car head to define the points order, so that we can directly regress the direction
        '''
        scaling_ratio = (grid_size[0]/(pc_range[3]-pc_range[0]))
        y_offset =  pc_range[4]
        x1 =  (bboxes_7d[:, :, 0] +  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        x2 =  (bboxes_7d[:, :, 0] +  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        x3 =  (bboxes_7d[:, :, 0] -  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        x4 =  (bboxes_7d[:, :, 0] -  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        y1 =  (bboxes_7d[:, :, 1] + y_offset +  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        y2 =  (bboxes_7d[:, :, 1] + y_offset + (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        y3 =  (bboxes_7d[:, :, 1] + y_offset -  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        y4 =  (bboxes_7d[:, :, 1] + y_offset -  (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        # find the x1,x2,x3,x4 and sort it , then find the correspondent y
        point_1 = torch.stack([x1,y1], dim=2)
        point_2 = torch.stack([x2,y2], dim=2)
        point_3 = torch.stack([x3,y3], dim=2)  
        point_4 = torch.stack([x4,y4], dim=2)    
        points = torch.stack([point_1,point_2,point_3,point_4], dim=2)           
        #sort y first , then sort x  => will get x as fisrt condition and y as second condition if x is the same    .... pytorch tensor.sort dont have simple conditional sort method, think better sort method here
        sort_list = points[:,:,:,1].sort()[1]
        #sort y first , but still have to paired with x
        x1_ = x1 * (sort_list[:,:,0]==0).type(torch.cuda.FloatTensor) + x2 * (sort_list[:,:,0]==1).type(torch.cuda.FloatTensor) + x3 * (sort_list[:,:,0]==2).type(torch.cuda.FloatTensor) + x4* (sort_list[:,:,0]==3).type(torch.cuda.FloatTensor)
        y1_ = y1 * (sort_list[:,:,0]==0).type(torch.cuda.FloatTensor) + y2 * (sort_list[:,:,0]==1).type(torch.cuda.FloatTensor) + y3 * (sort_list[:,:,0]==2).type(torch.cuda.FloatTensor) + y4* (sort_list[:,:,0]==3).type(torch.cuda.FloatTensor)
        x2_ = x1 * (sort_list[:,:,1]==0).type(torch.cuda.FloatTensor) + x2 * (sort_list[:,:,1]==1).type(torch.cuda.FloatTensor) + x3 * (sort_list[:,:,1]==2).type(torch.cuda.FloatTensor) + x4* (sort_list[:,:,1]==3).type(torch.cuda.FloatTensor)
        y2_ = y1 * (sort_list[:,:,1]==0).type(torch.cuda.FloatTensor) + y2 * (sort_list[:,:,1]==1).type(torch.cuda.FloatTensor) + y3 * (sort_list[:,:,1]==2).type(torch.cuda.FloatTensor) + y4* (sort_list[:,:,1]==3).type(torch.cuda.FloatTensor)
        x3_ = x1 * (sort_list[:,:,2]==0).type(torch.cuda.FloatTensor) + x2 * (sort_list[:,:,2]==1).type(torch.cuda.FloatTensor) + x3 * (sort_list[:,:,2]==2).type(torch.cuda.FloatTensor) + x4* (sort_list[:,:,2]==3).type(torch.cuda.FloatTensor)
        y3_ = y1 * (sort_list[:,:,2]==0).type(torch.cuda.FloatTensor) + y2 * (sort_list[:,:,2]==1).type(torch.cuda.FloatTensor) + y3 * (sort_list[:,:,2]==2).type(torch.cuda.FloatTensor) + y4* (sort_list[:,:,2]==3).type(torch.cuda.FloatTensor)
        x4_ = x1 * (sort_list[:,:,3]==0).type(torch.cuda.FloatTensor) + x2 * (sort_list[:,:,3]==1).type(torch.cuda.FloatTensor) + x3 * (sort_list[:,:,3]==2).type(torch.cuda.FloatTensor) + x4* (sort_list[:,:,3]==3).type(torch.cuda.FloatTensor)
        y4_ = y1 * (sort_list[:,:,3]==0).type(torch.cuda.FloatTensor) + y2 * (sort_list[:,:,3]==1).type(torch.cuda.FloatTensor) + y3 * (sort_list[:,:,3]==2).type(torch.cuda.FloatTensor) + y4* (sort_list[:,:,3]==3).type(torch.cuda.FloatTensor)
        point_1 = torch.stack([x1_,y1_], dim=2)
        point_2 = torch.stack([x2_,y2_], dim=2)
        point_3 = torch.stack([x3_,y3_], dim=2)
        point_4 = torch.stack([x4_,y4_], dim=2)
        points = torch.stack([point_1,point_2,point_3,point_4], dim=2)
        sort_list = points[:,:,:,0].sort()[1]
        # x will pair with right y1 by checking the sort list 
        first_point_x = x1_ * (sort_list[:,:,0]==0).type(torch.cuda.FloatTensor) + x2_ * (sort_list[:,:,0]==1).type(torch.cuda.FloatTensor) + x3_ * (sort_list[:,:,0]==2).type(torch.cuda.FloatTensor) + x4_ * (sort_list[:,:,0]==3).type(torch.cuda.FloatTensor)
        first_point_y = y1_ * (sort_list[:,:,0]==0).type(torch.cuda.FloatTensor) + y2_ * (sort_list[:,:,0]==1).type(torch.cuda.FloatTensor) + y3_ * (sort_list[:,:,0]==2).type(torch.cuda.FloatTensor) + y4_ * (sort_list[:,:,0]==3).type(torch.cuda.FloatTensor)
        second_point_x = x1_ * (sort_list[:,:,1]==0).type(torch.cuda.FloatTensor) + x2_ * (sort_list[:,:,1]==1).type(torch.cuda.FloatTensor) + x3_ * (sort_list[:,:,1]==2).type(torch.cuda.FloatTensor) + x4_* (sort_list[:,:,1]==3).type(torch.cuda.FloatTensor)
        second_point_y = y1_ * (sort_list[:,:,1]==0).type(torch.cuda.FloatTensor) + y2_ * (sort_list[:,:,1]==1).type(torch.cuda.FloatTensor) + y3_ * (sort_list[:,:,1]==2).type(torch.cuda.FloatTensor) + y4_* (sort_list[:,:,1]==3).type(torch.cuda.FloatTensor)
        third_point_x = x1_ * (sort_list[:,:,2]==0).type(torch.cuda.FloatTensor) + x2_ * (sort_list[:,:,2]==1).type(torch.cuda.FloatTensor) + x3_ * (sort_list[:,:,2]==2).type(torch.cuda.FloatTensor) + x4_* (sort_list[:,:,2]==3).type(torch.cuda.FloatTensor)
        third_point_y = y1_ * (sort_list[:,:,2]==0).type(torch.cuda.FloatTensor) + y2_ * (sort_list[:,:,2]==1).type(torch.cuda.FloatTensor) + y3_ * (sort_list[:,:,2]==2).type(torch.cuda.FloatTensor) + y4_* (sort_list[:,:,2]==3).type(torch.cuda.FloatTensor)
        fourth_point_x = x1_ * (sort_list[:,:,3]==0).type(torch.cuda.FloatTensor) + x2_ * (sort_list[:,:,3]==1).type(torch.cuda.FloatTensor) + x3_ * (sort_list[:,:,3]==2).type(torch.cuda.FloatTensor) + x4_* (sort_list[:,:,3]==3).type(torch.cuda.FloatTensor) 
        fourth_point_y = y1_ * (sort_list[:,:,3]==0).type(torch.cuda.FloatTensor) + y2_ * (sort_list[:,:,3]==1).type(torch.cuda.FloatTensor) + y3_ * (sort_list[:,:,3]==2).type(torch.cuda.FloatTensor) + y4_* (sort_list[:,:,3]==3).type(torch.cuda.FloatTensor)
        gt_four_points = torch.stack([first_point_x,first_point_y,second_point_x,second_point_y,third_point_x,third_point_y,fourth_point_x,fourth_point_y], dim=2)
        
        return gt_four_points

    def bbox_to_four_frame_middle_points_direction(self,bboxes_7d, grid_size ,pc_range):
        '''
        if we want to know the direction , we can use this function to define the points order
        point1 is the car head 
        point4 is the car tail 
        point2 is the car head's right 
        point3 is the car head's left
        '''
        scaling_ratio = (grid_size[0]/(pc_range[3]-pc_range[0]))
        y_offset =  pc_range[4]
        x1 =  (bboxes_7d[:, :, 0]            +  (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        y1 =  (bboxes_7d[:, :, 1] + y_offset +  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        x2 =  (bboxes_7d[:, :, 0]            +  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        y2 =  (bboxes_7d[:, :, 1] + y_offset -  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        x3 =  (bboxes_7d[:, :, 0]            -  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        y3 =  (bboxes_7d[:, :, 1] + y_offset +  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        x4 =  (bboxes_7d[:, :, 0]            -  (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        y4 =  (bboxes_7d[:, :, 1] + y_offset -  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        gt_four_points = torch.stack([x1,y1,x2,y2,x3,y3,x4,y4], dim=2)
        
        return gt_four_points


    def bbox_find_min_max_point(self,bboxes_7d, grid_size ,pc_range):
        scaling_ratio = (grid_size[0]/(pc_range[3]-pc_range[0]))
        y_offset =  pc_range[4]
        x1 =  (bboxes_7d[:, :, 0] +  (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6])) +  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio 
        x2 =  (bboxes_7d[:, :, 0] +  (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6])) -  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio 
        x3 =  (bboxes_7d[:, :, 0] -  (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6])) +  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio 
        x4 =  (bboxes_7d[:, :, 0] -  (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6])) -  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
        y1 =  (bboxes_7d[:, :, 1] + y_offset +  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6])) +  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        y2 =  (bboxes_7d[:, :, 1] + y_offset +  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6])) -  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        y3 =  (bboxes_7d[:, :, 1] + y_offset -  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6])) +  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        y4 =  (bboxes_7d[:, :, 1] + y_offset -  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6])) -  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
        point_1 = torch.stack([x1,y1], dim=2)
        point_2 = torch.stack([x2,y2], dim=2)
        point_3 = torch.stack([x3,y3], dim=2)
        point_4 = torch.stack([x4,y4], dim=2)
        points = torch.stack([point_1,point_2,point_3,point_4], dim=2)
        x_min=torch.min(points[:,:,:,0],dim=2)[0]
        x_max=torch.max(points[:,:,:,0],dim=2)[0]
        y_min=torch.min(points[:,:,:,1],dim=2)[0]
        y_max=torch.max(points[:,:,:,1],dim=2)[0]
        gt_min_max_points = torch.stack([x_min,x_max,y_min,y_max], dim=2)
        
        return gt_min_max_points

    def bbox_find_corner_points_without_rotation(self,bboxes_7d, grid_size ,pc_range):
        
        #scale the car for cheching the visualization
        #bboxes_7d[:, :, 4] = bboxes_7d[:, :, 4] * 10 
        #bboxes_7d[:, :, 3] = bboxes_7d[:, :, 3] * 10
        scaling_ratio = (grid_size[0]/(pc_range[3]-pc_range[0]))
        y_offset =  pc_range[4]
        left =  (bboxes_7d[:, :, 0] -  (bboxes_7d[:, :, 4]/2))  * scaling_ratio
        right = (bboxes_7d[:, :, 0] +  (bboxes_7d[:, :, 4]/2))  * scaling_ratio
        top =   (bboxes_7d[:, :, 1] + y_offset  +  (bboxes_7d[:, :, 3]/2)) * scaling_ratio
        bottom =  (bboxes_7d[:, :, 1] + y_offset  -   (bboxes_7d[:, :, 3]/2)) * scaling_ratio
        gt_corner_points = torch.stack([left,top,right,bottom], dim=2)
        
        return gt_corner_points


    def cal_area_from_bbox(self,bboxes_7d, grid_size ,pc_range ):
        area= bboxes_7d[:, :, 3] * bboxes_7d[:, :, 4] * (grid_size[0]/(pc_range[3]-pc_range[0])) *(grid_size[1]/(pc_range[4]-pc_range[1]))
        return area


    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        labels_ = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        zs = torch.tensor(()).new_zeros(len(xs),dtype=torch.float,device='cuda:0')
        angles=torch.tensor(()).new_zeros(len(xs),dtype=torch.float,device='cuda:0')
        voxel_size = targets['voxel_size']
        grid_size = targets['grid_size'][0]
        pc_range = targets['pc_range'][0]
        gt_bboxes = targets['gt_bbox']
        gt_labels = targets['gt_class']
        batch_size = len(gt_bboxes)
        pc_range = torch.from_numpy(pc_range).type(torch.cuda.FloatTensor)
        grid_size = torch.from_numpy(grid_size).type(torch.cuda.FloatTensor) 
        
        for im_i in range(len(targets['gt_bbox'])):
            gt_bboxes_per_im = torch.from_numpy(gt_bboxes[im_i]).type(torch.cuda.FloatTensor)[None,:,:]
            gt_labels_per_im = torch.from_numpy(gt_labels[im_i]).type(torch.cuda.FloatTensor)[None,:]
            voxel_size_per_im = torch.from_numpy(voxel_size[im_i]).type(torch.cuda.FloatTensor)[None,:]
            bbox_min_max_points = self.bbox_find_min_max_point(gt_bboxes_per_im, grid_size, pc_range)
            bbox_frame_middle_points= self.bbox_to_four_frame_middle_points(gt_bboxes_per_im, grid_size, pc_range)
            bbox_to_four_frame_middle_points_direction=self.bbox_to_four_frame_middle_points_direction(gt_bboxes_per_im, grid_size, pc_range)
            areas = self.cal_area_from_bbox(gt_bboxes_per_im, grid_size , pc_range )
            bbox_corner_points_without_rotation = self.bbox_find_corner_points_without_rotation(gt_bboxes_per_im, grid_size, pc_range)
            bbox_min_max_points_per_im = bbox_min_max_points[0]
            bbox_frame_middle_points_per_im = bbox_frame_middle_points[0]
            bbox_corner_points_without_rotate_per_im = bbox_corner_points_without_rotation[0]
            bbox_to_frame_middle_points_direction_per_im = bbox_to_four_frame_middle_points_direction[0]
            labels_per_im   = gt_labels_per_im[0]
            
            area = areas[0]

            # reg_target is for area where we want to regress , and reg_value is the value in that area we what to regress         
            # To do : get rid of the not possitive part bt rotating the target image 
            left = xs[:, None] -bbox_min_max_points_per_im[:, 0][None]
            top = ys[:, None] - bbox_min_max_points_per_im[:, 2][None]
            right =bbox_min_max_points_per_im[:, 1][None] - xs[:, None]
            bottom = bbox_min_max_points_per_im[:, 3][None] - ys[:, None] 
            reg_targets_per_im = torch.stack([left,top,right,bottom], dim=2)
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            '''
            #clean the useless positive area by rotating the original positive metrics
            left = xs[:, None] - bbox_corner_points_without_rotate_per_im[:, 0][None]
            top = bbox_corner_points_without_rotate_per_im[:, 1][None] - ys[:, None]  
            right = bbox_corner_points_without_rotate_per_im[:, 2][None] - xs[:, None]
            bottom = ys[:, None]  - bbox_corner_points_without_rotate_per_im[:, 3][None] 
            reg_targets_per_im = torch.stack([left,top,right,bottom], dim=2)
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            
            # need to check this method works well
            #out put the binary map through cv2
            
            # if we want to config here, we need to pass the fpn stride , and scale accordingly
            # be really careful about the operation below, the x axis and y axis are exchanged, so the rotate_angle also change accordingly.
            scale = 2.5 # 1/voxel_size * fpn_stride[i] 
            bbox_x_center  =  gt_bboxes_per_im[:,:,0] * scale
            bbox_y_center  = (gt_bboxes_per_im[:,:,1]+40) * scale
            # +90 cause the x,y changed
            rotate_angle =  90 - gt_bboxes_per_im[:,:,6] * 180/np.pi
            #why resize(400,352) instead of (352,400) => (352,400) is right but not compatible for cv2 visualization....
            images = is_in_boxes.reshape((200,176,-1)).permute((1,0,2))
            boxes_num = len(gt_bboxes_per_im[0])  
            rotated_boxes=[]
            for i in range(boxes_num):
                image_cv2 = images[:,:,None,i].cpu().numpy()
                
                #cv2.imwrite('/root/second.pytorch/images/pre_rotate_box/pre_img_{}_gt_{}.png'.format(targets['meta'][0]['image_idx'],i), image_cv2*255)
                M = cv2.getRotationMatrix2D((bbox_y_center[0][i],bbox_x_center[0][i]), rotate_angle[0][i] , 1)
                rotated_box=cv2.warpAffine(image_cv2 , M, (200,176))
                #cv2.imwrite('/root/second.pytorch/images/post_rotate_box/post_img_{}_gt_{}_rotate_{}.png'.format(targets['meta'][0]['image_idx'],i,rotate_angle[i]), rotated_box*255)
                rotated_boxes.append(torch.from_numpy(rotated_box)[:,:,None])
                
            is_in_rotated_boxes  = torch.cat(rotated_boxes,dim=2).permute((1,0,2)).reshape((-1,boxes_num)).type(torch.cuda.ByteTensor)
            # change to xi-xc/w , yi-yc/h
            '''
            ''' 
            m#regress the deviation for each points (order is sort by x axis)
            deviation_to_point1_x = (xs[:, None] - bbox_frame_middle_points_per_im[:, 0][None]) 
            deviation_to_point1_y = (ys[:, None] - bbox_frame_middle_points_per_im[:, 1][None]) 
            deviation_to_point2_x = (xs[:, None] - bbox_frame_middle_points_per_im[:, 2][None]) 
            deviation_to_point2_y = (ys[:, None] - bbox_frame_middle_points_per_im[:, 3][None]) 
            deviation_to_point3_x = (xs[:, None] - bbox_frame_middle_points_per_im[:, 4][None])
            deviation_to_point3_y = (ys[:, None] - bbox_frame_middle_points_per_im[:, 5][None]) 
            deviation_to_point4_x = (xs[:, None] - bbox_frame_middle_points_per_im[:, 6][None]) 
            deviation_to_point4_y = (ys[:, None] - bbox_frame_middle_points_per_im[:, 7][None]) 
            z =  zs[:,None] + gt_bboxes_per_im[:, 2][None] 
            h =  zs[:,None] + gt_bboxes_per_im[:, 5][None]
            '''      
            #with direction , aos 
            #regress the deviation for each points (order is define by car head, with head, we can compute direction)
            deviation_to_point1_x = (xs[:, None] - bbox_to_frame_middle_points_direction_per_im[:, 0][None]) 
            deviation_to_point1_y = (ys[:, None] - bbox_to_frame_middle_points_direction_per_im[:, 1][None]) 
            deviation_to_point2_x = (xs[:, None] - bbox_to_frame_middle_points_direction_per_im[:, 2][None]) 
            deviation_to_point2_y = (ys[:, None] - bbox_to_frame_middle_points_direction_per_im[:, 3][None]) 
            deviation_to_point3_x = (xs[:, None] - bbox_to_frame_middle_points_direction_per_im[:, 4][None])
            deviation_to_point3_y = (ys[:, None] - bbox_to_frame_middle_points_direction_per_im[:, 5][None]) 
            deviation_to_point4_x = (xs[:, None] - bbox_to_frame_middle_points_direction_per_im[:, 6][None]) 
            deviation_to_point4_y = (ys[:, None] - bbox_to_frame_middle_points_direction_per_im[:, 7][None])
            z =  zs[:,None] + gt_bboxes_per_im[:,:,2] *10
            h =  zs[:,None] + gt_bboxes_per_im[:,:,5] *10
            #method1
            #reg_value_per_im = torch.stack([point1_x,point1_y,point2_x,point2_y,point3_x,point3_y,point4_x,point4_y,z,h,angle], dim=2)
            #method2 
            # choose regress angle or not , it better to calcualte the angle by points 
            #reg_value_per_im = torch.stack([deviation_to_point1_x,deviation_to_point1_y,deviation_to_point2_x,deviation_to_point2_y,deviation_to_point3_x,deviation_to_point3_y,deviation_to_point4_x,deviation_to_point4_y,z,h,angle], dim=2)
            reg_value_per_im = torch.stack([deviation_to_point1_x,deviation_to_point1_y,deviation_to_point2_x,deviation_to_point2_y,deviation_to_point3_x,deviation_to_point3_y,deviation_to_point4_x,deviation_to_point4_y , z , h], dim=2)


            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]

            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            #find the area that not in the box and set to INF
            locations_to_gt_area[is_in_boxes == 0] = INF
            #find the area that are not cared is this level and set to INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            # location_to_gt_inds is the inds to choose which object we should use in this layer
            locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            # choose the cared obejct to regress [ins, object_nums, resgress_goal] -> [inds , resgress_goal]
            reg_targets_per_im = reg_value_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_aera == INF] = 0
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
             
            #calculate distance from each pixal to four points(frame middle point on rotated rec)

            
        return labels, reg_targets 
    def compute_box_score_targets(self, box_regression_flatten, reg_targets_flatten):
        '''
        we use pred and gt box to compute the IOU ,IOU is can be the confidence score for bbox,  the higher the better 
        '''
        #t = time.time()
        #TODO write the function to do 10d to 7d, consice the code. (make sure the scale is correct in each dimension) 
        #TODO iou function
        pred_center_x = (box_regression_flatten[:,0]+ box_regression_flatten[:,2] +box_regression_flatten[:,4]+box_regression_flatten[:,6])/4
        pred_center_y = (box_regression_flatten[:,1]+ box_regression_flatten[:,3] +box_regression_flatten[:,5]+box_regression_flatten[:,7])/4
        pre_point1_x = pred_center_x + (box_regression_flatten[:, 0])
        pre_point1_y = pred_center_y + (box_regression_flatten[:, 1])
        pre_point2_x = pred_center_x + (box_regression_flatten[:, 2])
        pre_point2_y = pred_center_y + (box_regression_flatten[:, 3])
        pre_point3_x = pred_center_x + (box_regression_flatten[:, 4])
        pre_point3_y = pred_center_y + (box_regression_flatten[:, 5])
        pre_point4_x = pred_center_x + (box_regression_flatten[:, 6])
        pre_point4_y = pred_center_y + (box_regression_flatten[:, 7])
        pre_line1= torch.sqrt((pre_point4_x-pre_point1_x)**2 +(pre_point4_y-pre_point1_y)**2)
        pre_line2= torch.sqrt((pre_point3_x-pre_point2_x)**2 +(pre_point3_y-pre_point2_y)**2)
        pre_w = torch.min(pre_line1,pre_line2)
        pre_l = torch.max(pre_line1,pre_line2)
        pre_x_axis_is_longer = (pre_line1 > pre_line2).type(torch.cuda.FloatTensor)
        pre_y_axis_is_longer = (pre_line1 < pre_line2).type(torch.cuda.FloatTensor)
        pre_angle = pre_x_axis_is_longer * torch.atan((pre_point4_x-pre_point1_x)/(pre_point4_y-pre_point1_y))+ pre_y_axis_is_longer * torch.atan((pre_point3_x-pre_point2_x)/(pre_point3_y-pre_point2_y))
        pre_bbox = torch.stack([ pred_center_x ,pred_center_y, pre_w ,pre_l, pre_angle], dim=1)
        gt_center_x =  (reg_targets_flatten[:,0]+reg_targets_flatten[:,2]+reg_targets_flatten[:,4]+reg_targets_flatten[:,6])/4
        gt_center_y =  (reg_targets_flatten[:,1]+reg_targets_flatten[:,3]+reg_targets_flatten[:,5]+reg_targets_flatten[:,7])/4
        gt_point1_x = gt_center_x + (reg_targets_flatten[:, 0])
        gt_point1_y = gt_center_y + (reg_targets_flatten[:, 1])
        gt_point2_x = gt_center_x + (reg_targets_flatten[:, 2])
        gt_point2_y = gt_center_y + (reg_targets_flatten[:, 3])
        gt_point3_x = gt_center_x + (reg_targets_flatten[:, 4])
        gt_point3_y = gt_center_y + (reg_targets_flatten[:, 5])
        gt_point4_x = gt_center_x + (reg_targets_flatten[:, 6])
        gt_point4_y = gt_center_y + (reg_targets_flatten[:, 7])
        gt_line1= torch.sqrt((gt_point4_x-gt_point1_x)**2 +(gt_point4_y-gt_point1_y)**2)
        gt_line2= torch.sqrt((gt_point3_x-gt_point2_x)**2 +(gt_point3_y-gt_point2_y)**2)
        gt_w = torch.min(gt_line1,gt_line2)
        gt_l = torch.max(gt_line1,gt_line2)
        gt_x_axis_is_longer = (gt_line1 > gt_line2).type(torch.cuda.FloatTensor)
        gt_y_axis_is_longer = (gt_line1 < gt_line2).type(torch.cuda.FloatTensor)
        gt_angle = gt_x_axis_is_longer * torch.atan((gt_point4_x-gt_point1_x)/(gt_point4_y-gt_point1_y))+ gt_y_axis_is_longer * torch.atan((gt_point3_x-gt_point2_x)/(gt_point3_y-gt_point2_y))
        gt_bbox = torch.stack([ gt_center_x ,gt_center_y, gt_w ,gt_l, gt_angle], dim=1)
        
        iou_list = []
        '''
        # cpu cv2 is slow , take 0.2~0.3s 
        for i in range(len(pre_bbox)):
            r1 = ((pre_bbox[i, 1], pre_bbox[i, 0]), (pre_bbox[i, 3], pre_bbox[i, 2]), pre_bbox[i, 4])
            r2 = ((gt_bbox[i, 1], gt_bbox[i, 0]), (gt_bbox[i, 3], gt_bbox[i, 2]), gt_bbox[i, 4])      
            area_r1 = pre_bbox[i, 2] * pre_bbox[i, 3]          
            area_r2 = gt_bbox[i, 2] * gt_bbox[i, 3]              
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                iou = (int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-8)).data[None]
                
            else:
                iou = torch.zeros([1], device='cuda', dtype=torch.float32)
            iou_list.append(iou)
        bbox_score_targets = torch.stack(iou_list) 
        '''

        #print("riou time", time.time() - t)  =====>   (around 0.2~0.3s )
        #TODO do it in gpu to minimize the training time

        iou_score = bev_box_overlap(pre_bbox.cpu().detach().numpy(),gt_bbox.cpu().detach().numpy())
        for i in range(len(iou_score)):
            iou_list.append(iou_score[i][i])

        bev_score_targets = torch.FloatTensor(iou_list).cuda()    
        
        pred_z = box_regression_flatten[:, 8] 
        pred_h = box_regression_flatten[:, 9]
        gt_z = reg_targets_flatten[:, 8]
        gt_h = reg_targets_flatten[:, 9]
        #try 1-mape as zhscore, but it may have some divide zero problem
        z_error = abs(gt_z - pred_z)
        z_error_normalize = (z_error - z_error.min())/(z_error.max()-z_error.min())
        z_center_score =1 - z_error_normalize
        h_error = abs(gt_h- pred_h)
        h_error_normalize = (h_error - h_error.min())/(h_error.max()-h_error.min())
        h_score =1 - h_error_normalize
        z_score = ((z_center_score + h_score)/2)
        bbox_score_targets= torch.stack((bev_score_targets, z_score),dim=1)
        
                                                                      
        return bbox_score_targets
    
    def cross_constraint_loss(self, pred_reg):
        
        vector1_x , vector1_y  = (pred_reg[:,6] - pred_reg[:,0]) , (pred_reg[:,7] - pred_reg[:,1])
        vector1 = torch.stack((vector1_x,vector1_y),dim=1)
        distance1 = torch.sqrt((vector1_x**2) + (vector1_y**2))
        vector2_x , vector2_y  = (pred_reg[:,4] - pred_reg[:,2]), (pred_reg[:,5] - pred_reg[:,3])
        vector2 = torch.stack((vector2_x,vector2_y),dim=1)
        distance2 = torch.sqrt((vector2_x**2) + (vector2_y**2))
        cos_list= []
        for i in range(len(vector1)):
            cos = torch.dot(vector1[i,:],vector2[i,:]) / (distance1[i]*distance2[i])
            cos_list.append(cos)
        # if we use abs , it means we use MAE as loss
        loss = torch.abs(torch.stack(cos_list))
        return loss.mean()

    def __call__(self, locations, box_cls, box_regression, box_score ,targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])
        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)

        num_regressions=box_regression[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)

         
        box_cls_flatten = []
        box_regression_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        box_score_flatten = []

        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, num_regressions))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, num_regressions))
            box_score_flatten.append(box_score[l].reshape(-1,2))
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        box_score_flatten = torch.cat(box_score_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        labels_flatten  = labels_flatten.unsqueeze(dim=1)
        #visualization 
        # blue gree red order for channgel color, red for gt , blue for pred , if match => purple
        #TODO write visualization to config
        visualize = False
        if visualize == True :
            first_image_size =  int(labels_flatten.size()[0] /box_cls[0].size()[0])
            gt_bbox_image = labels[0][:first_image_size].reshape((200,176))[:,:,None].cpu().numpy()*255
            pred_bbox_image = box_cls_flatten[:,0][:first_image_size].sigmoid().reshape((200,176))[:,:,None].cpu().detach().numpy()*255
            other_channel = labels[0][:first_image_size].reshape((200,176))[:,:,None].cpu().numpy()==-1
            mix_image =  np.concatenate((pred_bbox_image,other_channel,gt_bbox_image),axis=2)
            cv2.imwrite('/root/second.pytorch/images/post_rotate_box/gt&pred_bbox_img_{}.png'.format(targets['meta'][0]['image_idx']), mix_image)
            box_score_image = box_score_flatten[:,0][:first_image_size].reshape((200,176))[:,:,None].cpu().detach().numpy().astype(np.uint8)*255
            box_socre_colormap = cv2.applyColorMap(box_score_image, cv2.COLORMAP_JET)
            cv2.imwrite('/root/second.pytorch/images/box_score_image/bbox_score_img_{}.png'.format(targets['meta'][0]['image_idx']), box_socre_colormap)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        box_score_flatten = box_score_flatten[pos_inds]

        if pos_inds.numel() > 0:
            box_score_targets = self.compute_box_score_targets(box_regression_flatten ,reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
            ) / pos_inds.numel() + N
            box_score_loss = self.box_score_loss_func(
              box_score_flatten,
              box_score_targets
            )
            
            cross_constraint_loss = self.cross_constraint_loss(
            box_regression_flatten
            )
            
        else:
            reg_loss = box_regression_flatten.sum()
            box_score_loss = box_score_flatten.sum()
        
        total_loss =  cls_loss.sum() + reg_loss.sum() + box_score_loss.sum() + cross_constraint_loss
        
        #print('check the loss ')
        #embed() 
        #check_the_process_from_target_to_loss(box_cls_flatten,box_regression_flatten,labels_flatten,reg_targets_flatten,centerness_targets,pos_inds,targets)
        #embed()
         # return the labels and care area to cal acc
        if torch.isnan(total_loss):
            embed()
        return total_loss, cls_loss, reg_loss , box_score_loss, labels_flatten


def make_fcos_loss_evaluator():
    loss_evaluator = FCOSLossComputation()
    return loss_evaluator





