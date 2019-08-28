import math
from pathlib import Path
import numba
import numpy as np
from spconv.utils import (
    non_max_suppression_cpu, rotate_non_max_suppression_cpu)
from second.core import box_np_ops
from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu
from IPython import embed

def nms_cc(dets, thresh):
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    return non_max_suppression_cpu(dets, order, thresh, 1.0)


def rotate_nms_cc(dets, thresh):
    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    dets_corners = box_np_ops.center_to_corner_box2d(dets[:, :2], dets[:, 2:4],
                                                     dets[:, 4])

    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)

    standup_iou = box_np_ops.iou_jit(dets_standup, dets_standup, eps=0.0)

    return rotate_non_max_suppression_cpu(dets_corners, order, standup_iou,thresh)


def rotate_nms_merge(dets, thresh):
    '''
    input dets_corners, order, standup_iou,thresh
    return bbox_list  ( not inds, cause we merges the bboxes, new boxes is not in the original boxlist)
    we can use score or centerness as weight to get better merge method
    parameter for merge nms
    
    iou_thresh : > thresh hold , then we add those bboxes to same list, and merge them
    intersect_bbox_nums_thresh : numbers of intersection bboxes > threshold , including itself, if the number is too small, we can ignore the bboxes
    TODO:
    score weight : with or without centerness.sigmoid
    '''
    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    dets_corners = box_np_ops.center_to_corner_box2d(dets[:, :2], dets[:, 2:4],
                                                     dets[:, 4])

    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)

    standup_iou = box_np_ops.iou_jit(dets_standup, dets_standup, eps=0.0)

    unmerged_boxlist = []


    while order.size > 0:
        i = order[0]
        merged_boxes_ind = np.where(standup_iou[i] > thresh)
        unmerged_boxlist.append(merged_boxes_ind[0])

        # filter out the merge inds in the order
        for ele in merged_boxes_ind[0]:
            if ele in order:
                remove_inds = np.where(order==ele)[0]
                order = np.delete(order, remove_inds[0])
    return unmerged_boxlist

    #depreciate : moving anchor to absord the bbox
    merged_boxes_num = 0
    rounds=0
    thresh= 0.1
    intersect_bbox_nums_thresh = 1
    start = 0


    while order.size > 0 :
        print('we have {} bboxes in round {}'.format(order.size, rounds))
        i = order[0]
        merged_boxes_ind = np.where(standup_iou[i] >= thresh)
        # we can set a thresh hold for intersetction number, if intersetion bbox num too small, we give up the bbox
        if merged_boxes_ind[0].size > intersect_bbox_nums_thresh:
            print('round {} have intersection{} nums'.format(rounds,merged_boxes_ind[0].size))
            if  not merged_boxes_num in unmerged_boxlist:
                unmerged_boxlist[merged_boxes_num]=merged_boxes_ind
                unmerged_boxes_inds = np.where(standup_iou[i] < thresh)
            else:
                unmerged_boxlist[merged_boxes_num] =  np.concatenate((unmerged_boxlist[merged_boxes_num],merged_boxes_ind[1:]),axis=0)
                unmerged_boxes_inds = np.where(standup_iou[i] < thresh)

        else:
             order = order[unmerged_boxes_inds+ 1]
        #rounds+=1
    final_bboxes_list = []
    total_box_check = 0
    for key , bboxes in unmerged_boxlist.items():
        final_bboxes_list.append(bboxes.mean(axis=0))
        total_box_check += unmerged_boxlist[key].shape[0]
    embed()
    return final_bboxes_list

@numba.jit(nopython=True)
def nms_jit(dets, thresh, eps=0.0):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate iou between i and j box
            w = max(min(x2[i], x2[j]) - max(x1[i], x1[j]) + eps, 0.0)
            h = max(min(y2[i], y2[j]) - max(y1[i], y1[j]) + eps, 0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[j] - inter)
            # ovr = inter / areas[j]
            if ovr >= thresh:
                suppressed[j] = 1
    return keep


@numba.jit('float32[:, :], float32, float32, float32, uint32', nopython=True)
def soft_nms_jit(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area -
                               iw * ih)
                    ov = iw * ih / ua  #iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep

