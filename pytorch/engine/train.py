import copy
import json
import os
import pathlib
import pickle
import shutil
import time
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from google.protobuf import text_format
from IPython import embed
import second.data.kitti_common as kitti
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.log_tool import SimpleModelLog
from second.utils.progress_bar import ProgressBar




def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        else:
            example_torch[k] = v
    return example_torch


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net


def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])


def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          resume=False):
    """train a VoxelNet model specified by a config file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = pathlib.Path(model_dir)
    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg).to(device)
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    class_names = target_assigner.classes

    # net_train = torch.nn.DataParallel(net).cuda()
    print("num_trainable parameters:", len(list(net.parameters())))
    # for n, p in net.named_parameters():
    #     print(n, p.shape)
    ######################
    # BUILD OPTIMIZER
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    mixed_optimizer = optimizer_builder.build(
        optimizer_cfg,
        net,
        mixed=train_cfg.enable_mixed_precision,
        loss_scale=loss_scale)
    optimizer = mixed_optimizer
    center_limit_range = model_cfg.post_center_limit_range
    """
    if train_cfg.enable_mixed_precision:
        mixed_optimizer = torchplus.train.MixedPrecisionWrapper(
            optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    """
    # must restore optimizer AFTER using MixedPrecisionWrapper
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [mixed_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer,
                                              train_cfg.steps)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32
    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    data_iter = iter(dataloader)
    print(data_iter)
    ######################
    # TRAINING
    ######################
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")

    total_step_elapsed = 0
    remain_steps = train_cfg.steps - net.get_global_step()
    t = time.time()
    ckpt_start_time = t
    steps_per_eval = train_cfg.steps_per_eval
    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    mixed_optimizer.zero_grad()
    try:
        for _ in range(total_loop):
            if total_step_elapsed + train_cfg.steps_per_eval > train_cfg.steps:
                steps = train_cfg.steps % train_cfg.steps_per_eval
            else:
                steps = train_cfg.steps_per_eval
            for step in range(steps):
                lr_scheduler.step(net.get_global_step())
                try:
                    example = next(data_iter)
                except StopIteration:
                    print("end epoch")
                    if clear_metrics_every_epoch:
                        net.clear_metrics()
                    data_iter = iter(dataloader)
                    example = next(data_iter)
                example_torch = example_convert_to_torch(example, float_dtype)
             
                #batch_size = example["anchors"].shape[0]
                ret_dict = net(example_torch)

                # FCOS
                            
                losses = ret_dict['total_loss']
                loss_cls = ret_dict["loss_cls"]
                loss_reg = ret_dict["loss_reg"]
                cls_preds = ret_dict['cls_preds']
                labels =  ret_dict["labels"]
                cared = ret_dict["labels"]
                
                
                optimizer.zero_grad()
                losses.backward()
                #torch.nn.utils.clip_grad_norm_(net.parameters(),  1)
                # optimizer_step is for updating the parameter, so clip before update
                optimizer.step()
                net.update_global_step()
                #need to unpack the [0] for fpn
                net_metrics = net.update_metrics(loss_cls,loss_reg, cls_preds[0], labels, cared )
                step_time = (time.time() - t)
                t = time.time()
                metrics = {}
                global_step = net.get_global_step()

                #print log 
                if global_step % display_step == 0:
                    metrics["runtime"] = {
                        "step": global_step,
                        "steptime": step_time,
                    }
                 
                    metrics.update(net_metrics)
                    metrics["misc"] = {
                        "num_vox": int(example_torch["voxels"].shape[0]),
                        "lr": float(optimizer.lr),
                    }
                    model_logging.log_metrics(metrics, global_step)
                ckpt_elasped_time = time.time() - ckpt_start_time
                torchplus.train.save_models(model_dir, [net, optimizer],
                                        net.get_global_step())
  
            total_step_elapsed += steps
            torchplus.train.save_models(model_dir, [net, optimizer],
                                        net.get_global_step())
            net.eval()
            result_path_step = result_path / f"step_{net.get_global_step()}"
            result_path_step.mkdir(parents=True, exist_ok=True)
            model_logging.log_text("#################################",
                                   global_step)
            model_logging.log_text("# EVAL", global_step)
            model_logging.log_text("#################################",
                                   global_step)
            model_logging.log_text("Generate output labels...", global_step)
            t = time.time()
            detections = []
            prog_bar = ProgressBar()
            net.clear_timer()
            prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1)
                           // eval_input_cfg.batch_size)
            for example in iter(eval_dataloader):
                example = example_convert_to_torch(example, float_dtype)
                with torch.no_grad():
                    detections += net(example)
                prog_bar.print_bar()

            sec_per_ex = len(eval_dataset) / (time.time() - t)
            model_logging.log_text(
                f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                global_step)
            result_dict = eval_dataset.dataset.evaluation(
                detections, str(result_path_step))
            for k, v in result_dict["results"].items():
                model_logging.log_text("Evaluation {}".format(k), global_step)
                model_logging.log_text(v, global_step)
            model_logging.log_metrics(result_dict["detail"], global_step)
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(detections, f)
            net.train()   
            '''
                new version of evaluation while trainging 
                # do the evaluation while traingingi
                if global_step % steps_per_eval == 0:
                   
                    torchplus.train.save_models(model_dir, [net, optimizer],
                                                net.get_global_step())
                    net.eval()
                    result_path_step = result_path / f"step_{net.get_global_step()}"
                    result_path_step.mkdir(parents=True, exist_ok=True)
                    model_logging.log_text("#################################",
                                        global_step)
                    model_logging.log_text("# EVAL", global_step)
                    model_logging.log_text("#################################",
                                        global_step)
                    model_logging.log_text("Generate output labels...", global_step)
                    t = time.time()
                    detections = []
                    prog_bar = ProgressBar()
                    net.clear_timer()
                    prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1)
                                // eval_input_cfg.batch_size)
                    for example in iter(eval_dataloader):
                        example = example_convert_to_torch(example, float_dtype)
                        with torch.no_grad():
                            detections += net(example)
                        prog_bar.print_bar()

                    sec_per_ex = len(eval_dataset) / (time.time() - t)
                    model_logging.log_text(
                        f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                        global_step)
                    result_dict = eval_dataset.dataset.evaluation(
                        detections, str(result_path_step))
                    for k, v in result_dict["results"].items():
                        model_logging.log_text("Evaluation {}".format(k), global_step)
                        model_logging.log_text(v, global_step)
                    model_logging.log_metrics(result_dict["detail"], global_step)
                    with open(result_path_step / "result.pkl", 'wb') as f:
                        pickle.dump(detections, f)
                    net.train()
            '''

    except Exception as e:
        print("trainging error") 
        raise e
    finally:
        model_logging.close()
    # save model before exit
    torchplus.train.save_models(model_dir, [net, optimizer],
                                net.get_global_step())


def evaluate(config_path,
             model_dir=None,
             result_path=None,
             ckpt_path=None,
             measure_time=False,
             batch_size=None):
    """Don't support pickle_result anymore. if you want to generate kitti label file,
    please use kitti_anno_to_label_file in second.data.kitti_dataset.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_name = 'eval_results'
    if result_path is None:
        model_dir = pathlib.Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    net = build_network(model_cfg, measure_time=measure_time).to(device)
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    class_names = target_assigner.classes

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    detections = []
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()
    
    
    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            torch.cuda.synchronize()
            t1 = time.time()
        example = example_convert_to_torch(example, float_dtype)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)
        with torch.no_grad():
            detections += net(example)
        bar.print_bar()
        if measure_time:
            t2 = time.time()
    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms"
        )
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    with open(result_path_step / "result.pkl", 'wb') as f:
        pickle.dump(detections, f)
    '''
    with open(result_path_step / "result.pkl", 'rb') as f:
        detections = pickle.load(f)
    print(detections[0].keys())
    '''       
    result_dict = eval_dataset.dataset.evaluation(detections, str(result_path_step))
    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print("Evaluation {}".format(k))
            print(v)
        # metric dict: class -> metric type -> diff -> overlap -> recall, prec, ...


def save_config(config_path, save_path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    ret = text_format.MessageToString(config, indent=2)
    with open(save_path, 'w') as f:
        f.write(ret)


if __name__ == '__main__':
    fire.Fire()
