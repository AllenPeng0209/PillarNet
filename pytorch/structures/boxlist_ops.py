
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

'''
this is for kitti 7d bbox ops, largly based on maskrcnn 4d boxlist_ops

'''
import numpy as np
import torch

from .bounding_box import BoxList


from second.core.non_max_suppression.nms_gpu import (nms_gpu, rotate_iou_gpu,
                                                       rotate_nms_gpu)
from second.core.non_max_suppression.nms_cpu import rotate_nms_cc,rotate_nms_merge
from second.pytorch.core import box_torch_ops
from IPython import embed


def boxlist_rotate_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores", merge_nms=False):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """


    #nms_func = box_torch_ops.rotate_nms 
    rbboxes = boxlist.bbox[:,[0,1,3,4,6]]
    #TODO try to multiply cls confidence 
    bev_scores = boxlist.get_field(score_field)[:,0]
    z_scores = boxlist.get_field(score_field)[:,1]

    dets = torch.cat([rbboxes, bev_scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
        boxlist = boxlist[keep]
        return boxlist
    if merge_nms== True:
        ret = rotate_nms_merge(dets_np, nms_thresh)
        merge_boxes_list = [] 
        merge_score_list = []
        for i in range(len(ret)):     
             # only merge z, h , cause box_score already choose best bev
             
             zh_max_ids = boxlist[ret[i]].get_field("scores")[:,1].max(0)[1]
             merge_box_zh = boxlist[ret[i]].bbox[zh_max_ids,[2,5]]

             box_max_ids = boxlist[ret[i]].get_field("scores")[:,0].max(0)[1]
             merge_box_bev = boxlist[ret[i]].bbox[box_max_ids,[0,1,3,4,6]]
             

             merge_box_stack = torch.cat((merge_box_bev[0][None],merge_box_bev[1][None],merge_box_zh[0][None],merge_box_bev[2][None],merge_box_bev[3][None],merge_box_zh[1][None],merge_box_bev[4][None]))
             merge_boxes_list.append(merge_box_stack)
             merge_score_list.append(boxlist[ret[i]].get_field("scores").max())
        merge_bboxes = torch.stack(merge_boxes_list)   
        merge_scores = torch.stack(merge_score_list)
        boxlist = BoxList(merge_bboxes,(400,352))
        boxlist.add_field("scores", merge_scores)
        return boxlist

    ret = np.array(rotate_nms_cc(dets_np, nms_thresh), dtype=np.float32)
    keep = ret[:max_proposals]
    '''
    merge rotate nms
    return unmerge bboxlist and use boxlist.merge to merge the box
    '''

    #return empty boxlist 
    #if keep.shape[0] == 0:
    #    return torch.zeros([0]).long().to(rbboxes.device)

    #TODO original return ids , now we return new bboxlist , so we have to write new Boxlist methed to create bboxlist from numpy
    boxlist = boxlist[keep]
    
    return boxlist
    



def boxes_fillter(boxlist):
    """
    Only keep boxes with both sides >= min_size
    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # modified for kitti car , need to change if we want to detect pedestrain or cyclist
    #TODO

    #if labels == "Car"
    min_w , max_w , min_l , max_l = 1.1 , 2.2 ,3 , 5
    bbox_min_w = boxlist.bbox[:,3] > min_w
    bbox_max_w = boxlist.bbox[:,3] < max_w
    bbox_min_l = boxlist.bbox[:,4] > min_l
    bbox_max_l = boxlist.bbox[:,4] < max_l
    keep = (bbox_min_w & bbox_max_w & bbox_min_l & bbox_max_l).nonzero()

   





    return boxlist[keep.flatten()]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList
    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)


    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
