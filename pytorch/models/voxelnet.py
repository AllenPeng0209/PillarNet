
import time
from enum import Enum
from functools import reduce
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from IPython import embed
import torchplus
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                        WeightedSmoothL1LocalizationLoss,
                                        WeightedSoftmaxClassificationLoss)
from second.pytorch.models import middle, pointpillars, rpn, voxel_encoder
from torchplus import metrics

from second.pytorch.models.FCOS.loss import make_fcos_loss_evaluator
from second.pytorch.models.FCOS.inference import make_fcos_postprocessor
from second.pytorch.models.FCOS.compute_locations import compute_locations

# To do : change to backbone module
from second.pytorch.models.FCOS import middle2
import cv2
from second.utils import simplevis
from skimage import transform,data


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_num_input_features=-1,
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 measure_time=False,
                 voxel_generator=None,
                 post_center_range=None,
                 fpn_strides=[2,1,1],
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_direction_classifier = use_direction_classifier
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self.target_assigner = target_assigner
        self.voxel_generator = voxel_generator
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._diff_loc_loss_ftor = WeightedSmoothL1LocalizationLoss()
        self.rpn_class_name = rpn_class_name
        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self._post_center_range = post_center_range or []
        self.measure_time = measure_time
        # pointnet++, pointsift, geocnn
        vfe_class_dict = {
            "VoxelFeatureExtractor": voxel_encoder.VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": voxel_encoder.VoxelFeatureExtractorV2,
            "SimpleVoxel": voxel_encoder.SimpleVoxel,
            "SimpleVoxelRadius": voxel_encoder.SimpleVoxelRadius,
            "PillarFeatureNet": pointpillars.PillarFeatureNet,
        }
        vfe_class = vfe_class_dict[vfe_class_name]
        self.voxel_feature_extractor = vfe_class(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range,
        )
        mid_class_dict = {
            "SparseMiddleExtractor": middle.SparseMiddleExtractor,
            "SpMiddleD4HD": middle.SpMiddleD4HD,
            "SpMiddleD8HD": middle.SpMiddleD8HD,
            "SpMiddleFHD": middle.SpMiddleFHD,
            "SpMiddleFHDV2": middle.SpMiddleFHDV2,
            "SpMiddleFHDLarge": middle.SpMiddleFHDLarge,
            "SpMiddleResNetFHD": middle.SpMiddleResNetFHD,
            "SpMiddleD4HDLite": middle.SpMiddleD4HDLite,
            "SpMiddleFHDLite": middle.SpMiddleFHDLite,
            "SpMiddle2K": middle.SpMiddle2K,
            "SpMiddleFHDPeople": middle.SpMiddleFHDPeople,
            "SpMiddle2KPeople": middle.SpMiddle2KPeople,
            "SpMiddleHDLite": middle.SpMiddleHDLite,
            "PointPillarsScatter": pointpillars.PointPillarsScatter,
            "SpMiddleFHDLiteNoNorm": middle.SpMiddleFHDLiteNoNorm,
        }
        mid_class = mid_class_dict[middle_class_name]
        self.middle_feature_extractor = mid_class(
            output_shape,
            use_norm,
            num_input_features=middle_num_input_features,
            num_filters_down1=middle_num_filters_d1,
            num_filters_down2=middle_num_filters_d2)
        rpn_class_dict = {
            "RPN": rpn.RPN,
            "RPNV2": rpn.RPNV2,
            "ResNetRPN": rpn.ResNetRPN,
            "FCOS": rpn.FCOS,
        }
        rpn_class = rpn_class_dict[rpn_class_name]
        #if not run FCOS, then go original RPN , loss
        if rpn_class_name != "FCOS":
            self.rpn = rpn_class(
                use_norm=True,
                num_class=num_class,
                layer_nums=rpn_layer_nums,
                layer_strides=rpn_layer_strides,
                num_filters=rpn_num_filters,
                upsample_strides=rpn_upsample_strides,
                num_upsample_filters=rpn_num_upsample_filters,
                num_input_features=rpn_num_input_features,
                num_anchor_per_loc=target_assigner.num_anchors_per_location,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=target_assigner.box_coder.code_size)
        #Here is the FCOS self.rpn
        else:
            # need to clean the code, will change to backbone in the future
            self.middle2_feature_extractor = middle2.res_fpn(
                use_norm=True,
                num_class=num_class,
                layer_nums=rpn_layer_nums,
                layer_strides=rpn_layer_strides,
                num_filters=rpn_num_filters,
                upsample_strides=rpn_upsample_strides,
                num_upsample_filters=rpn_num_upsample_filters,
                num_input_features=rpn_num_input_features,
                num_anchor_per_loc=target_assigner.num_anchors_per_location,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=target_assigner.box_coder.code_size)
            self.rpn = rpn_class(
                use_norm=True,
                num_class=num_class,
                layer_nums=rpn_layer_nums,
                layer_strides=rpn_layer_strides,
                num_filters=rpn_num_filters,
                upsample_strides=rpn_upsample_strides,
                num_upsample_filters=rpn_num_upsample_filters,
                num_input_features=rpn_num_input_features,
                num_anchor_per_loc=target_assigner.num_anchors_per_location,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=10)
            #pass the cfg into make_fcos_loss_eval
            self.loss_evaluator = make_fcos_loss_evaluator()
            #pass the cfg into box_selector_test , its handcraft parameter know
            # background count as 1 class, so add 1 
            self.box_selector_test = make_fcos_postprocessor(num_classes = num_class+1)
            self.fpn_strides=fpn_strides           
        self.rpn_acc = metrics.Accuracy(
                dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
                dim=-1,
                thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
                use_sigmoid_score=use_sigmoid_score,
                encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()

        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}

    def start_timer(self, *names):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        for name in names:
            self._time_dict[name] = time.time()

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def clear_global_step(self):
        self.global_step.zero_()

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        points = example['points']
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        voxel_size = example['voxel_size']
        grid_size = example['grid_size']
        pc_range = example['pc_range']
        gt_class = example['gt_class']
        gt_bbox = example['gt_bbox']
        meta = example['metadata']
        # remake example to targets for fcos uses
        targets = {'gt_class':gt_class, 'gt_bbox':gt_bbox, 'voxel_size':voxel_size, 'pc_range':pc_range, 'grid_size':grid_size, 'meta':meta}
       


        if len(num_points.shape) == 2: # multi-gpu
            num_voxel_per_batch = example["num_voxels"].cpu().numpy().reshape(-1)
            voxel_list = []
            num_points_list = []
            coors_list = []
            for i, num_voxel in enumerate(num_voxel_per_batch):
                voxel_list.append(voxels[i, :num_voxel])
                num_points_list.append(num_points[i, :num_voxel])
                coors_list.append(coors[i, :num_voxel])
            voxels = torch.cat(voxel_list, dim=0)
            num_points = torch.cat(num_points_list, dim=0)
            coors = torch.cat(coors_list, dim=0)
        #batch_anchors = example["anchors"]
        batch_size_dev =  len(gt_bbox)     
        t = time.time()
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        self.end_timer("voxel_feature_extractor")

        self.start_timer("middle forward")
        features = self.middle_feature_extractor(
            voxel_features, coors, batch_size_dev)
        self.end_timer("middle forward")
        self.start_timer("rpn forward")
        if self.rpn_class_name == "FCOS":
            fpn_head = self.middle2_feature_extractor(features)
            preds_dict = self.rpn(fpn_head)
            box_preds = preds_dict["box_preds"]
            cls_preds = preds_dict["cls_preds"]
            box_score_preds = preds_dict["box_score_preds"]

            #now we only use one head for detection , in h/2 size 
            #TODO config here
            locations = compute_locations(fpn_head, fpn_strides=[2,1,1])
            #print("check before calcualte the loss-- fpn_head, box_pred, locations...")
            #embed()
            # make  example['targets']
            if self.training:
                #take 0.14s for traing 
                proposals, proposal_losses =  self._forward_train(locations, cls_preds, 
                                                                  box_preds, box_score_preds,targets)

                #check target and inference while trainging and visualize it 
                trainging_visualization = False
                if trainging_visualization  == True :
                    boxlist =  self._forward_test(locations, cls_preds, box_preds,box_score_preds,
                                                   targets ,grid_size, voxel_size)
                    bev_map = simplevis.kitti_vis(points[0], boxlist[0].bbox.cpu().detach().numpy(), gt_bbox[0]) 
                    cv2.imwrite('/root/second.pytorch/images/kitti_training_bev/new_traing_predict_'+str(example['metadata'][0]['image_idx'])+'.jpg', bev_map)
                
                losses = {}
                losses['cls_preds'] = cls_preds
                losses.update(proposal_losses)

                return losses
            else:
                #check the targets <-> gt_pred   without voxelnet
 
                #test path , check the targets <-> gt_pred   without voxelnet
                #predictions_dict  = checking_the_whole_process_without_voxelnet(targets,example)
                #return predictions_dict
                #real path
                self.start_timer("predict")
                boxlist =  self._forward_test(
                locations, cls_preds, box_preds,box_score_preds,
                targets ,grid_size, voxel_size)
                self.end_timer("predict")
                #bev_map = simplevis.kitti_vis(points[0], boxlist[0].bbox.cpu().detach().numpy(), gt_bbox[0])
                #cv2.imwrite('/root/second.pytorch/images/kitti_test_bev/test_predict_'+str(example['metadata'][0]['image_idx'])+'.jpg', bev_map)
                predictions_dict=[]
                for i in range(batch_size_dev):
                    prediction_dict = {
                         "box3d_lidar":boxlist[i].bbox,
                         "scores":boxlist[i].get_field('scores'),
                         "label_preds":boxlist[i].get_field('labels'),
                         "metadata":example['metadata'][i]
                    }
                    predictions_dict.append(prediction_dict)
                return predictions_dict
	# original rpn code      
        else:
            preds_dict = self.rpn(features)
            self.end_timer("rpn forward")
            box_preds = preds_dict["box_preds"]
            cls_preds = preds_dict["cls_preds"]
            if self.training:
                labels = example['labels']
                reg_targets = example['reg_targets']

                cls_weights, reg_weights, cared = prepare_loss_weights(
                    labels,
                    pos_cls_weight=self._pos_cls_weight,
                    neg_cls_weight=self._neg_cls_weight,
                    loss_norm_type=self._loss_norm_type,
                    dtype=voxels.dtype)
                cls_targets = labels * cared.type_as(labels)
                cls_targets = cls_targets.unsqueeze(-1)

                loc_loss, cls_loss = create_loss(
                    self._loc_loss_ftor,
                    self._cls_loss_ftor,
                    box_preds=box_preds,
                    cls_preds=cls_preds,
                    cls_targets=cls_targets,
                    cls_weights=cls_weights,
                    reg_targets=reg_targets,
                    reg_weights=reg_weights,
                    num_class=self._num_class,
                    encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                    encode_background_as_zeros=self._encode_background_as_zeros,
                    box_code_size=self._box_coder.code_size,
                )
                loc_loss_reduced = loc_loss.sum() / batch_size_dev
                loc_loss_reduced *= self._loc_loss_weight
                cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
                cls_pos_loss /= self._pos_cls_weight
                cls_neg_loss /= self._neg_cls_weight
                cls_loss_reduced = cls_loss.sum() / batch_size_dev
                cls_loss_reduced *= self._cls_loss_weight
                loss = loc_loss_reduced + cls_loss_reduced
                if self._use_direction_classifier:
                    dir_targets = get_direction_target(example['anchors'],
                                                       reg_targets)
                    dir_logits = preds_dict["dir_cls_preds"].view(
                        batch_size_dev, -1, 2)
                    weights = (labels > 0).type_as(dir_logits)
                    weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                    dir_loss = self._dir_loss_ftor(
                        dir_logits, dir_targets, weights=weights)
                    dir_loss = dir_loss.sum() / batch_size_dev
                    loss += dir_loss * self._direction_loss_weight
                
                return {
                    "loss": loss,
                    "cls_loss": cls_loss,
                    "loc_loss": loc_loss,
                    "cls_pos_loss": cls_pos_loss,
                    "cls_neg_loss": cls_neg_loss,
                    "cls_preds": cls_preds,
                    "dir_loss_reduced": dir_loss,
                    "cls_loss_reduced": cls_loss_reduced,
                    "loc_loss_reduced": loc_loss_reduced,
                    "cared": cared,
                }
                     
            else:
                self.start_timer("predict")
                with torch.no_grad():
                    res = self.predict(example, preds_dict)
                self.end_timer("predict")
                return res
    def predict(self, example, preds_dict):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx), 
                    for nuscenes, sample_token is saved in it.
            }
        """
        batch_size = example['anchors'].shape[0]
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        post_center_range = None
        if len(self._post_center_range) > 0:
            post_center_range = torch.tensor(
                self._post_center_range,
                dtype=batch_box_preds.dtype,
                device=batch_box_preds.device).float()
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms

            if self._multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_torch_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            torch.full([num_dets], i, dtype=torch.int64))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self._use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)

                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                if self._nms_score_threshold > 0.0:
                    thresh = torch.tensor(
                        [self._nms_score_threshold],
                        device=total_scores.device).type_as(total_scores)
                    top_scores_keep = (top_scores >= thresh)
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                    torch.zeros([0, 7], dtype=dtype, device=device),
                    "scores":
                    torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                    torch.zeros([0], dtype=top_labels.dtype, device=device),
                    "metadata":
                    meta,
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts
    def _forward_train(self, locations, box_cls, box_regression ,box_score_preds,targets):
        # the code is ugly, we need to get labels and area_cared to calcualte the acc, but original code make target(anchor) before traing
        total_loss , loss_box_cls, loss_box_reg, loss_box_score,labels = self.loss_evaluator(
            locations, box_cls, box_regression, box_score_preds, targets
        )
        losses = {
            "total_loss": total_loss,
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_box_score":loss_box_score,
            "box_cls": box_cls,
            "labels" : labels,
        }
        
        return None, losses
    def _forward_test(self, locations, box_cls, box_regression, box_score , targets,image_sizes, voxel_size):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression
            , box_score , targets, image_sizes, voxel_size
        )
        # should output the prediction_dicts like the predict method above, so transform boxlist to prediction_dict
        # To do : redesign the port for the original code or write the transfrom medthod in the boxlist
  
        return boxes
    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self, cls_loss, loc_loss, cls_preds, labels, sampled):
        #TODO it's better to rewrite whole metric for fcos 
        #no fpn , if fpn then change whole metric
        batch_size = cls_preds.shape[0]
        cls_preds = cls_preds.flatten()[:,None]
        num_class = self._num_class
        labels = labels.expand((-1,num_class)).flatten()[:,None]
        if not self._encode_background_as_zeros:
            num_class += 1
        #cls_preds = cls_preds.view(batch_size, -1, num_class).flatten()
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "loss": {
                "cls_loss": float(rpn_cls_loss),
                "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
                'loc_loss': float(rpn_loc_loss),
                "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            },
            "rpn_acc": float(rpn_acc),
            "pr": {},
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret["pr"][f"prec@{int(thresh*100)}"] = float(prec[i])
            ret["pr"][f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(child)
        return net


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets



def checking_the_whole_process_without_voxelnet(targets,example):
    voxel_size = targets['voxel_size']
    grid_size = targets['grid_size'][0]
    pc_range = targets['pc_range'][0]
    gt_bboxes = targets['gt_bbox']
    gt_labels = targets['gt_class']
    meta = targets['meta']
    gt_bboxes = torch.from_numpy(gt_bboxes).type(torch.cuda.FloatTensor)
    gt_labels = torch.from_numpy(gt_labels).type(torch.cuda.FloatTensor)     
    voxel_size = torch.from_numpy(voxel_size).type(torch.cuda.FloatTensor)
    pc_range = torch.from_numpy(pc_range).type(torch.cuda.FloatTensor)
    grid_size = torch.from_numpy(grid_size).type(torch.cuda.FloatTensor)

    bboxes_7d=gt_bboxes
    scaling_ratio = (grid_size[0]/(pc_range[3]-pc_range[0]))
    x1 =  (bboxes_7d[:, :, 0] +  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
    x2 =  (bboxes_7d[:, :, 0] +  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
    x3 =  (bboxes_7d[:, :, 0] -  (bboxes_7d[:, :, 3]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
    x4 =  (bboxes_7d[:, :, 0] -  (bboxes_7d[:, :, 4]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
    y1 =  (bboxes_7d[:, :, 1] + 40 +  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
    y2 =  (bboxes_7d[:, :, 1] + 40 + (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
    y3 =  (bboxes_7d[:, :, 1] + 40 -  (bboxes_7d[:, :, 3]/2 * torch.sin(bboxes_7d[:, :, 6]))) * scaling_ratio
    y4 =  (bboxes_7d[:, :, 1] + 40 -  (bboxes_7d[:, :, 4]/2 * torch.cos(bboxes_7d[:, :, 6]))) * scaling_ratio
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
    bbox = torch.stack([first_point_x,first_point_y,second_point_x,second_point_y,third_point_x,third_point_y,fourth_point_x,fourth_point_y,bboxes_7d[:, :, 2],bboxes_7d[:, :, 5],bboxes_7d[:, :, 6]], dim=2)
    

    #transform back the original state
    line_1  = torch.sqrt((bbox[:,:,6]/5 - bbox[:,:,0]/5)**2 + (bbox[:,:,7]/5 - bbox[:,:,1]/5)**2)
    line_2  = torch.sqrt((bbox[:,:,4]/5 - bbox[:,:,2]/5)**2+ (bbox[:,:,5]/5 - bbox[:,:,3]/5)**2)
    x = ((bbox[:,:,0] + bbox[:,:,2] + bbox[:,:,4] + bbox[:,:,6]) / 4 ) / 5
    y = ((bbox[:,:,1] + bbox[:,:,3] + bbox[:,:,5] + bbox[:,:,7])/4 / 5) - 40
    z = bbox[:,:,8]
    w = torch.min(line_1, line_2) 
    l = torch.max(line_1, line_2)
    h = bbox [:,:,9]
    # we can also use points to calculate the angle
    zr = limit_period(bbox[:,:,10]) + 0.08
    bbox_7d = torch.stack([x,y,z,w,l,h,zr],dim=1)
    bbox_7d = bbox_7d.permute(0,2,1)   

     
    predictions_dict=[]
    # To do : change to batch style
    for i in range(1):
          prediction_dict = {
               "box3d_lidar": bbox_7d[i],
               "scores":gt_labels[i],
               "label_preds":gt_labels[i],
               "metadata":example['metadata'][i]
               }
          predictions_dict.append(prediction_dict)
    
    return predictions_dict
    
def add_pi(angle):
   return angle + np.pi
def limit_period(angle, offset=0.5, period=np.pi):
    return angle - torch.floor(angle/period + offset) * period



