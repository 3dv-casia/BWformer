import torch
import torch.nn as nn
import os
import time
import datetime
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from arguments import get_args_parser
from datasets.outdoor_buildings import OutdoorBuildingDataset
from datasets.data_utils import collate_fn, get_pixel_features
from models.corner_models_3d import HeatCorner3d
from models.corner_models import HeatCorner
from models.edge_models import HeatEdge
from models.resnet import ResNetBackbone
from models.loss import Corner3dCriterion, EdgeCriterion, CornerCriterion
from models.corner_to_edge import prepare_edge_data
from models.matcher import build_matcher
from torch.utils.data.distributed import DistributedSampler
import utils.misc as utils
from scipy.spatial.distance import cdist
from tensorboardX import SummaryWriter
from datetime import datetime
import scipy.ndimage.filters as filters


def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def pad_gt_corners(bs, annots, num_queries):
   
    corner_targets = []
    # padding ground truth on-fly
    for i in range(bs):
        annot = annots[i]
        gt_corners = np.array(list(annot.keys()))
        ind = np.lexsort(gt_corners.T)  
        corners = gt_corners[ind]  

        corners = torch.from_numpy(corners).cuda()

        corners = corners.reshape(-1,1).squeeze()
      
        corners = torch.clip(corners, 0, 255) / 255
        corners_pad = (torch.zeros(len(corners))).cuda()
        corners_pad[:len(corners)] = corners


        center_dict = {}
        corner_length = []
        corner_length.append(len(corners)/3)
        
        labels = torch.ones(int(len(corners)/3), dtype=int)
        labels_pad = torch.zeros(int(len(corners)/3)).cuda()
        labels_pad[:len(labels)] = labels
        
        
        center_dict = {
            'coords': corners_pad.reshape(-1,3),
            'labels': labels_pad.reshape(-1,1),
            'length': torch.tensor(corner_length)
        }
        
        corner_targets.append(center_dict)
 
    return corner_targets

def train_one_epoch(image_size, backbone,corner_model, corner_model3d,edge_model,corner_criterion, corner_criterion3d, edge_criterion, data_loader,
                    optimizer,
                    epoch, max_norm, args):

    backbone.train()
    corner_model.train()
    corner_model3d.train()

    edge_model.train()
    corner_criterion.train()
    corner_criterion3d.train()
    edge_criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    pixels, pixel_features = get_pixel_features(image_size)
    pixel_features = pixel_features.cuda()


    for data in metric_logger.log_every(data_loader, print_freq, header):
        
        loss_3dcenter, loss_labels, loss_cardinality,loss_height, loss_x, loss_y,corner_outputs, corner_loss, corner_recall,  logits_s1, logits_s2_hb, logits_s2_rel, s1_losses, s2_losses_hb, \
        s2_losses_rel, s1_acc, s2_acc_hb, s2_acc_rel = run_model(
            data,
            pixels,
            pixel_features,
            backbone,
            corner_model,
            corner_model3d,
            edge_model,
            epoch,
            corner_criterion,
            corner_criterion3d,
            edge_criterion,
            args)

        loss = s1_losses + s2_losses_hb + s2_losses_rel + ( 2000 * loss_3dcenter + 100000 * loss_labels)  + corner_loss * args.lambda_corner

        loss_dict = {'loss_e_s1': s1_losses, 'loss_e_s2_hb': s2_losses_hb, 'loss_e_s2_rel': s2_losses_rel,
                    'edge_acc_s1': s1_acc, 'edge_acc_s2_hb': s2_acc_hb, 'edge_acc_s2_rel': s2_acc_rel,
                    'loss_3dcenter': loss_3dcenter, 'loss_labels': loss_labels, 'loss_cardinality': loss_cardinality, 'loss_height':loss_height, 'loss_x':loss_x, 'loss_y':loss_y,'loss_c_s1': corner_loss, 'corner_recall': corner_recall}
        loss_value = loss.item()
        
        optimizer.zero_grad()
        loss.backward()

    

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(corner_model.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(corner_model3d.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(edge_model.parameters(), max_norm)

        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
   

def run_model(data, pixels, pixel_features, backbone, corner_model, corner_model3d, edge_model, epoch, corner_criterion,corner_criterion3d, edge_criterion,
              args):
    image = data['img'].cuda()
    bs = image.shape[0]
    annots = data['annot']
    raw_images = data['raw_img']
    pixel_labels = data['pixel_labels'].cuda()
    gauss_labels = data['gauss_labels'].cuda()


    num_queries = args.max_corner_num

    center_targets = pad_gt_corners(bs, annots, num_queries)

    pixel_features = pixel_features.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)

    image_feats, feat_mask, all_image_feats = backbone(image)

    preds_s1 = corner_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats)

    corner_loss_s1, corner_recall = corner_criterion(preds_s1, pixel_labels, gauss_labels, epoch)

    c_outputs = preds_s1
    maxnum = 150
    
    corners2d = prepare_corner_data(c_outputs,maxnum)
    
    corner_logits, corner_coord = corner_model3d(corners2d, image_feats, feat_mask, all_image_feats)
   
    loss_3dcenter, loss_labels, loss_cardinality, loss_height, loss_x, loss_y= corner_criterion3d(corner_logits, corner_coord, center_targets)
    
    maxnum = 150
    
    corner_coord = corner_coord * 255
   
    corner_coord = corner_coord.reshape(-1,1)
    corner_coord = torch.clip(corner_coord, 0, 255) 
    corner_coord = corner_coord.reshape(bs,-1,3)
  
    edge_data = prepare_edge_data(corner_coord,corner_logits, annots, raw_images, maxnum)
   
    edge_coords = edge_data['edge_coords'].cuda()
    edge_mask = edge_data['edge_coords_mask'].cuda()
    edge_lengths = edge_data['edge_coords_lengths'].cuda()
    edge_labels = edge_data['edge_labels'].cuda()
    corner_nums = edge_data['processed_corners_lengths']

    
    max_candidates = torch.stack([corner_nums.max() * args.corner_to_edge_multiplier] * len(corner_nums), dim=0)
    logits_s1, logits_s2_hb, logits_s2_rel, s2_ids, s2_edge_mask, s2_gt_values = edge_model(image_feats, feat_mask,
                                                                                            pixel_features,
                                                                                            edge_coords, edge_mask,
                                                                                            edge_labels,
                                                                                            corner_nums,
                                                                                            max_candidates)

    s1_losses, s1_acc, s2_losses_hb, s2_acc_hb, s2_losses_rel, s2_acc_rel = edge_criterion(logits_s1, logits_s2_hb,
                                                                                           logits_s2_rel, s2_ids,
                                                                                           s2_edge_mask,
                                                                                           edge_labels, edge_lengths,
                                                                                           edge_mask, s2_gt_values)

    return  loss_3dcenter, loss_labels, loss_cardinality,loss_height,loss_x, loss_y, c_outputs, corner_loss_s1, corner_recall,logits_s1, logits_s2_hb, logits_s2_rel, s1_losses, s2_losses_hb, \
            s2_losses_rel, s1_acc, s2_acc_hb, s2_acc_rel


@torch.no_grad()
def evaluate(image_size, backbone, corner_model, edge_model, corner_criterion, edge_criterion, data_loader, epoch,
             args):
    backbone.eval()
    corner_model.eval()
    edge_model.eval()
    corner_criterion.eval()
    edge_criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    pixel_features = get_pixel_features(image_size)
    pixel_features = pixel_features.cuda()

    for data in metric_logger.log_every(data_loader, 10, header):
        loss_3dcenter, loss_labels, loss_cardinality, loss_height, loss_x, loss_y,  logits_s1, \
        logits_s2_hb, logits_s2_rel, s1_losses, s2_losses_hb, s2_losses_rel, s1_acc, s2_acc_hb, s2_acc_rel = run_model(
            data,
            pixel_features,
            backbone,
            corner_model,
            edge_model,
            epoch,
            corner_criterion,
            edge_criterion,
            args)

        loss_dict = {'loss_e_s1': s1_losses,
                     'loss_e_s2_hb': s2_losses_hb,
                     'loss_e_s2_rel': s2_losses_rel,
                     'edge_acc_s1': s1_acc,
                     'edge_acc_s2_hb': s2_acc_hb,
                     'edge_acc_s2_rel': s2_acc_rel,
                     'loss_3dcenter': loss_3dcenter,
                     'loss_labels': loss_labels,
                     'loss_cardinality' : loss_cardinality,
                     'loss_height': loss_height,
                     'loss_x':loss_x,
                     'loss_y':loss_y}

        loss = s1_losses + s2_losses_hb + s2_losses_rel + ( 200 * loss_3dcenter + 100000 * loss_labels) 
        loss_value = loss.item()
        metric_logger.update(loss=loss_value, **loss_dict)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    parser = argparse.ArgumentParser('HEAT training', parents=[get_args_parser()])
    args = parser.parse_args()
    image_size = args.image_size
    log_dir = "./tensorboard/test"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, current_time)
    # 确保目录存在
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    data_path = './building3d/b3d'
    det_path = None
    train_dataset = OutdoorBuildingDataset(data_path, det_path, phase='train', image_size=image_size, rand_aug=True,
                                            inference=False)
    test_dataset = OutdoorBuildingDataset(data_path, det_path, phase='train', image_size=image_size, rand_aug=False,
                                              inference=False)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)

    test_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)


    backbone = ResNetBackbone()
    strides = backbone.strides
    num_channels = backbone.num_channels

    corner_model3d = HeatCorner3d(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                              backbone_num_channels=num_channels)
    
    
    corner_model = HeatCorner(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                              backbone_num_channels=num_channels)

    edge_model = HeatEdge(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                          backbone_num_channels=num_channels)
    

    backbone.to(device)
    corner_model.to(device)
    corner_model3d.to(device)

    edge_model.to(device)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[args.local_rank],
                                                output_device=args.local_rank, find_unused_parameters = True)
        corner_model = nn.parallel.DistributedDataParallel(corner_model, device_ids=[args.local_rank],
                                                output_device=args.local_rank, find_unused_parameters = True)
        corner_model3d = nn.parallel.DistributedDataParallel(corner_model3d, device_ids=[args.local_rank],
                                                output_device=args.local_rank, find_unused_parameters = True)
        
        edge_model= nn.parallel.DistributedDataParallel(edge_model, device_ids=[args.local_rank],
                                                output_device=args.local_rank, find_unused_parameters=True)
  

    matcher = build_matcher()
    corner_criterion = CornerCriterion(image_size=image_size)
    corner_criterion3d = Corner3dCriterion(matcher)
    edge_criterion = EdgeCriterion()

    backbone_params = [p for p in backbone.parameters()]
    corner_params = [p for p in corner_model.parameters()]
    corner_params3d = [p for p in corner_model3d.parameters()]

    edge_params = [p for p in edge_model.parameters()]


    all_params = corner_params + edge_params + backbone_params + corner_params3d 
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    start_epoch = args.start_epoch

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        backbone.load_state_dict(ckpt['backbone'])
        corner_model.load_state_dict(ckpt['corner_model'])
        corner_model3d.load_state_dict(ckpt['corner_model3d'])

        edge_model.load_state_dict(ckpt['edge_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        lr_scheduler.step_size = args.lr_drop

        print('Resume from ckpt file {}, starting from epoch {}'.format(args.resume, ckpt['epoch']))
        start_epoch = ckpt['epoch'] + 1

    n_backbone_parameters = sum(p.numel() for p in backbone_params if p.requires_grad)
    n_corner_parameters = sum(p.numel() for p in corner_params if p.requires_grad)
    n_corner_parameters3d = sum(p.numel() for p in corner_params3d if p.requires_grad)
    n_edge_parameters = sum(p.numel() for p in edge_params if p.requires_grad)
    n_all_parameters = sum(p.numel() for p in all_params if p.requires_grad)
    print('number of trainable backbone params:', n_backbone_parameters)
    print('number of trainable corner params:', n_corner_parameters)
    print('number of trainable corner3d params:', n_corner_parameters3d)
    print('number of trainable edge params:', n_edge_parameters)
    print('number of all trainable params:', n_all_parameters)

    print("Start training")
    start_time = time.time()

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_acc = 0
    for epoch in range(start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            image_size, backbone, corner_model, corner_model3d,  edge_model, corner_criterion, corner_criterion3d, edge_criterion, train_dataloader,
            optimizer,
            epoch, args.clip_max_norm, args)
        lr_scheduler.step()
        for key, value in train_stats.items():
            writer.add_scalar('train/' + key, value, epoch)
        if args.run_validation:
            val_stats = evaluate(
                image_size, backbone, corner_model, edge_model, corner_criterion, edge_criterion, test_dataloader,
                epoch, args
            )

            val_acc = (val_stats['edge_acc_s1'] + val_stats['edge_acc_s2_hb']) / 2
            if val_acc > best_acc:
                is_best = True
                best_acc = val_acc
            else:
                is_best = False
            for key, value in val_stats.items():
                writer.add_scalar('val/' + key, value, epoch)
        
        else:
            val_acc = 0
            is_best = False

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if is_best:
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
            if epoch%20 == 0 and epoch > 0:
                checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
                checkpoint_paths.append(output_dir / checkpoint_filename)


            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'backbone': backbone.state_dict(),
                    'corner_model': corner_model.state_dict(),
                    'corner_model3d': corner_model3d.state_dict(),

                    'edge_model': edge_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_acc': val_acc,
                }, checkpoint_path)
		
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def prepare_corner_data(c_outputs, max_corner_num):
    bs = c_outputs.shape[0]

    all_results = list()

    for b_i in range(bs):
        output = c_outputs[b_i]
        results = process_each_corner_sample({'output': output}, max_corner_num)
        all_results.append(results)

    return torch.tensor(all_results).to(c_outputs.device)

def process_each_corner_sample(data, max_corner_num):
    output = data['output']

    preds = output.detach().cpu().numpy()
    NEIGHBOUR_SIZE = 5
    LOCAL_MAX_THRESH = 0.01
    data_max = filters.maximum_filter(preds, NEIGHBOUR_SIZE)
    maxima = (preds == data_max)
    data_min = filters.minimum_filter(preds, NEIGHBOUR_SIZE)
    diff = ((data_max - data_min) > 0)
    maxima[diff == 0] = 0
    local_maximas = np.where((maxima > 0) & (preds > LOCAL_MAX_THRESH))
    pred_corners = np.stack(local_maximas, axis=-1)[:, [1, 0]]  # to (x, y format)
    if pred_corners.shape[0] > max_corner_num:
        indices = np.random.choice(pred_corners.shape[0], max_corner_num, replace=False)
        pred_corners = pred_corners[indices]
    if pred_corners.shape[0] < max_corner_num:
        num_padding = max_corner_num - pred_corners.shape[0]
        additional_corners = np.ones((num_padding, 2), dtype=pred_corners.dtype) * 255
        pred_corners = np.vstack((pred_corners, additional_corners))
    pred_corners = np.concatenate([pred_corners,pred_corners], axis=0)
    ind = np.lexsort(pred_corners.T)  # sort the g.t. corners to fix the order for the matching later
    pred_corners = pred_corners[ind]   
    return pred_corners
def nms_group(pred_corners, pred_logits, dist_thresh=5.0, score_thresh=0.01):
    new_corners = []

    n = 2  # 每n个为一组，可调整
    for i in range(0, len(pred_logits), n):
        group_corners = pred_corners[i:i+n]
        group_logits = pred_logits[i:i+n]

        # 转为 numpy
        group_corners = np.array(group_corners)
        group_logits = np.array(group_logits).reshape(-1)

        # 去掉低置信度
        valid_mask = group_logits >= score_thresh
        group_corners = group_corners[valid_mask]
        group_logits = group_logits[valid_mask]

        if len(group_corners) == 0:
            continue

        # 排序
        order = np.argsort(-group_logits)
        group_corners = group_corners[order]
        group_logits = group_logits[order]

        keep = []
        suppressed = np.zeros(len(group_corners), dtype=bool)

        for m in range(len(group_corners)):
            if suppressed[m]:
                continue
            keep.append(group_corners[m])
            for n in range(m + 1, len(group_corners)):
                if suppressed[n]:
                    continue
                dist = np.linalg.norm(group_corners[m] - group_corners[n])
                if dist < dist_thresh:
                    suppressed[n] = True

        new_corners.extend(keep)

    return np.array(new_corners)

def prepare_3dcorner_data(corner_coord,corner_logits, annots, images, max_corner_num):

    bs = corner_coord.shape[0]
    
    all_results = list()

    for b_i in range(bs):
        annot = annots[b_i]
        output_coord = corner_coord[b_i]
        output_logit = corner_logits[b_i].sigmoid()
       
        results = process_each_sample_3dcorner({'annot': annot, 'output_coord':output_coord,'output_logit':output_logit, 'viz_img': images[b_i]}, max_corner_num)

        all_results.append(results)
    
    return torch.tensor(all_results, dtype=torch.float).to(corner_coord.device)

def process_each_sample_3dcorner(data, max_corner_num):
    annot = data['annot']
    output_coord = data['output_coord']
    output_logit = data['output_logit']
    

    preds = output_coord.detach().cpu().numpy()

    
    pred_corners = preds
    pred_logits = output_logit.detach().cpu().numpy()
   

    processed_corners = get_edge_label_mix_gt_3dcorner(pred_corners, pred_logits, annot, max_corner_num)
    

    return processed_corners


def get_edge_label_mix_gt_3dcorner(pred_corners,pred_logits, annot, max_corner_num):

    new_corners = []
    new_hs = []

    new_corners = nms_group(pred_corners, pred_logits, dist_thresh=5.0, score_thresh=0.01)
    pred_corners = np.array(new_corners)

    ind = np.lexsort(pred_corners.T)  
    pred_corners = pred_corners[ind] 

    gt_corners, edge_pairs, corner_degrees = process_annot_3dcorner(annot)

  
    output_to_gt = dict()
    gt_to_output = dict()
    diff = np.sqrt(((pred_corners[:, None] - gt_corners) ** 2).sum(-1))
    diff = diff.T
    
    MATCH_THRESH = 5
    if len(pred_corners) > 0:
        for target_i, target in enumerate(gt_corners):
            dist = diff[target_i]
            
            if len(output_to_gt) > 0:
                dist[list(output_to_gt.keys())] = 1000  # ignore already matched pred corners
            min_dist = dist.min()
            min_idx = dist.argmin()
           
            if min_dist < MATCH_THRESH and min_idx not in output_to_gt:    
                output_to_gt[min_idx] = (target_i, min_dist)
                gt_to_output[target_i] = min_idx

    all_corners = gt_corners.copy()

    for gt_i in range(len(gt_corners)):
       if gt_i in gt_to_output:
            all_corners[gt_i] = pred_corners[gt_to_output[gt_i]]

    nm_pred_ids = [i for i in range(len(pred_corners)) if i not in output_to_gt]
    nm_pred_ids = np.random.permutation(nm_pred_ids)
    if len(nm_pred_ids) > 0:
        nm_pred_corners = pred_corners[nm_pred_ids]
        if len(nm_pred_ids) + len(all_corners) <= max_corner_num:
            all_corners = np.concatenate([all_corners, nm_pred_corners], axis=0)
            num_padding = max_corner_num - (len(nm_pred_ids) + len(all_corners))

            additional_indices = np.random.choice(all_corners.shape[0], num_padding, replace=True)
            additional_corners = all_corners[additional_indices]
            all_corners = np.vstack((all_corners, additional_corners))

        else:
            all_corners = np.concatenate([all_corners, nm_pred_corners[:(max_corner_num - len(gt_corners)), :]], axis=0)

    return all_corners

def process_annot_3dcorner(annot, do_round=True):
    corners = np.array(list(annot.keys()))
   
    ind = np.lexsort(corners.T)  
    corners = corners[ind]  
    corner_mapping = {tuple(k): v for v, k in enumerate(corners)}

    edges = list()
    for c, connections in annot.items():
        for other_c in connections:
            edge_pair = (corner_mapping[c], corner_mapping[tuple(other_c)])
            edges.append(edge_pair)
    corner_degrees = [len(annot[tuple(c)]) for c in corners]
    if do_round:
        corners = corners.round()
    return corners, edges, corner_degrees

if __name__ == '__main__':
    main()
