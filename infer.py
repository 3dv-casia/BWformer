import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.outdoor_buildings import OutdoorBuildingDataset
from datasets.test_outdoor_buildings import testOutdoorBuildingDataset
from datasets.data_utils import collate_fn, get_pixel_features
from models.resnet import ResNetBackbone
from models.corner_models import HeatCorner
from models.corner_models_3d import HeatCorner3d
from models.edge_models import HeatEdge
from models.corner_to_edge import get_infer_edge_pairs
import numpy as np
import cv2
import glob
import os
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import skimage
import argparse
def calculate_distance(point1, point2):

    return np.linalg.norm(point1 - point2)


def save_wireframe(vertices, edges, wireframe_file):
    with open(wireframe_file, 'w') as f:
        for vertex in vertices:
            line = ' '.join(map(str, vertex))
            f.write('v ' + line + '\n')
        for edge in edges:
            edge = ' '.join(map(str, edge + 1))
            f.write('l ' + edge + '\n')



def corner_nms(preds, confs, image_size):
    data = np.zeros([image_size, image_size])
    neighborhood_size = 5
    
    threshold = 0

    for i in range(len(preds)):
        data[preds[i, 1], preds[i, 0]] = confs[i]

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = list()
    for i, pred in enumerate(filtered_preds):
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)

    return filtered_preds, new_confs


def main(dataset, ckpt_path, image_size,  infer_times):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print('Load from ckpts of epoch {}'.format(ckpt['epoch']))
    ckpt_args = ckpt['args']
    det_path = None
    data_path = './building3d/b3d'
    
    test_dataset = testOutdoorBuildingDataset(data_path, det_path, phase='test', image_size=image_size, rand_aug=False,
                                              inference=True)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                 collate_fn=collate_fn)

    backbone = ResNetBackbone()
    strides = backbone.strides
    num_channels = backbone.num_channels
    backbone = nn.DataParallel(backbone)
    backbone = backbone.cuda()
    backbone.eval()
    corner_model = HeatCorner(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                              backbone_num_channels=num_channels)
    corner_model = nn.DataParallel(corner_model)
    corner_model = corner_model.cuda()
    corner_model.eval()

    corner_model3d = HeatCorner3d(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                              backbone_num_channels=num_channels)
    corner_model3d = nn.DataParallel(corner_model3d)
    corner_model3d = corner_model3d.cuda()
    corner_model3d.eval()

    edge_model = HeatEdge(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                          backbone_num_channels=num_channels)
    edge_model = nn.DataParallel(edge_model)
    edge_model = edge_model.cuda()
    edge_model.eval()

    backbone.load_state_dict(ckpt['backbone'])
    corner_model.load_state_dict(ckpt['corner_model'])
    corner_model3d.load_state_dict(ckpt['corner_model3d'])
    edge_model.load_state_dict(ckpt['edge_model'])
    print('Loaded saved model from {}'.format(ckpt_path))



    pixels, pixel_features = get_pixel_features(image_size=image_size)
    num=0
    roof_dir = "./building3d/Tallinn/test/"
    result_dir = "./results/"
    
    for data_i, data in enumerate(test_dataloader):
        print(num)
        num = num + 1
        image = data['img'].cuda()
        img_path = data['img_path'][0]
      
        annot = None

        img_name = data['name'][0]
        pc_file = os.path.join(roof_dir, 'xyz', img_name+'.xyz')
        pc = np.loadtxt(pc_file, dtype=np.float64)

        point_cloud = pc[:, 0:3]

        print('Finish inference for sample No.'+img_name)
        try:
            with torch.no_grad():
                pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np = get_results(image, backbone,
                                                                                            corner_model,
                                                                                            corner_model3d,
                                                                                            edge_model,
                                                                                            pixels,
                                                                                            pixel_features,
                                                                                            ckpt_args, infer_times,
                                                                                            corner_thresh=0.01,
                                                                                            image_size=image_size)

            centroid = np.mean(point_cloud[:, 0:3], axis=0)
            point_cloud[:, 0:3] -= centroid

            max_distance = np.max(np.linalg.norm(np.vstack(point_cloud[:, 0:3]), axis=1))

            point_cloud[:, 0:3] /= (max_distance)
            point_cloud[:, 0:3] = (point_cloud[:, 0:3] +  np.ones_like(point_cloud[:, 0:3]))  * 127.5

            pred_corners = ((pred_corners-127.5*np.ones_like(pred_corners))/127.5) * (max_distance)  + centroid

            pred_corners, pred_confs, pos_edges = postprocess_preds(pred_corners, pred_confs, pos_edges)

            pred_data = {
                'corners': pred_corners,
                'edges': pos_edges,
            }

            objpath = os.path.join(result_dir, img_name+'.obj')
            save_wireframe(pred_corners, pos_edges, objpath)
        except Exception as e:
            pred_data = {
                'corners': [],
                'edges': [],
            }

            objpath = os.path.join(result_dir, img_name+'.obj')
            save_wireframe(pred_corners, pos_edges, objpath)
          
     

def get_results(image, backbone, corner_model, corner_model3d, edge_model,  pixels, pixel_features,
                args, infer_times, corner_thresh=0.5, image_size=256):
    image_feats, feat_mask, all_image_feats = backbone(image)
    pixel_features = pixel_features.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)
    preds_s1 = corner_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats)
    c_outputs = preds_s1
    maxnum = 150
    
    corners2d = prepare_corner_data(c_outputs,maxnum)

    corner_logits, corner_coord = corner_model3d(corners2d, image_feats, feat_mask, all_image_feats)

    c_outputs = corner_logits.sigmoid()
    
    corner_coord = corner_coord*255
    corner_coord = torch.clip(corner_coord, 0, 255) 
    c_outputs_np = c_outputs[0].detach().cpu().numpy()
    c_coord_np = corner_coord[0].detach().cpu().numpy()
    
    pos_indices = np.where(c_outputs_np >= corner_thresh)
    coor_indices = np.array(pos_indices[0])
    pred_corners = c_coord_np[coor_indices,:]
   
    pred_confs = np.array(c_outputs_np[pos_indices])

    sorted_indices = np.argsort(-pred_confs)  

    pred_corners_sorted = pred_corners[sorted_indices]
    pred_confs_sorted = pred_confs[sorted_indices]
    newpred_corners = []
    newpred_conf = []
    for i in range(pred_confs_sorted.shape[0]):
        addyn = True
        for j in range(len(newpred_corners)):
            if calculate_distance(newpred_corners[j], pred_corners_sorted[i])<5:
                addyn = False
                continue
        if addyn:
            newpred_corners.append(pred_corners_sorted[i])
            newpred_conf.append(pred_confs_sorted[i])
    pred_confs = np.array(newpred_conf)
    pred_corners = np.array(newpred_corners)
    
    pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(pred_corners, pred_confs)
    corner_nums = torch.tensor([len(pred_corners)]).to(image.device)
    max_candidates = torch.stack([corner_nums.max() * args.corner_to_edge_multiplier] * len(corner_nums), dim=0)

    all_pos_ids = set()
    all_edge_confs = dict()

    for tt in range(infer_times):
        if tt == 0:
            gt_values = torch.zeros_like(edge_mask).long()
            gt_values[:, :] = 2

        s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, s2_gt_values = edge_model(image_feats, feat_mask,
                                                                                                 pixel_features,
                                                                                                 edge_coords, edge_mask,
                                                                                                 gt_values, corner_nums,
                                                                                                 max_candidates,
                                                                                                 True)
        
        num_total = s1_logits.shape[2]
        num_selected = selected_ids.shape[1]
        num_filtered = num_total - num_selected

        s1_preds = s1_logits.squeeze().softmax(0)
        s2_preds_rel = s2_logits_rel.squeeze().softmax(0)
        s2_preds_hb = s2_logits_hb.squeeze().softmax(0)
        s1_preds_np = s1_preds[1, :].detach().cpu().numpy()
        s2_preds_rel_np = s2_preds_rel[1, :].detach().cpu().numpy()
        s2_preds_hb_np = s2_preds_hb[1, :].detach().cpu().numpy()

        selected_ids = selected_ids.squeeze().detach().cpu().numpy()
        if tt != infer_times - 1:
            s2_preds_np = s2_preds_hb_np

            pos_edge_ids = np.where(s2_preds_np >= 0.9)
            neg_edge_ids = np.where(s2_preds_np <= 0.01)
            for pos_id in pos_edge_ids[0]:
                actual_id = selected_ids[pos_id]
                if gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = s2_preds_np[pos_id]
                gt_values[0, actual_id] = 1
            for neg_id in neg_edge_ids[0]:
                actual_id = selected_ids[neg_id]
                if gt_values[0, actual_id] != 2:
                    continue
                gt_values[0, actual_id] = 0
            num_to_pred = (gt_values == 2).sum()
            if num_to_pred <= num_filtered:
                break
        else:
            s2_preds_np = s2_preds_hb_np

            pos_edge_ids = np.where(s2_preds_np >= 0.5)
            for pos_id in pos_edge_ids[0]:
                actual_id = selected_ids[pos_id]
                if s2_mask[0][pos_id] is True or gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = s2_preds_np[pos_id]

    
    pos_edge_ids = list(all_pos_ids)
    edge_confs = [all_edge_confs[idx] for idx in pos_edge_ids]
    pos_edges = edge_ids[pos_edge_ids].cpu().numpy()
    edge_confs = np.array(edge_confs)

    if image_size != 256:
        pred_corners = pred_corners / (image_size / 256)

    return pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np


def postprocess_preds(corners, confs, edges):
    corner_degrees = dict()
    for edge_i, edge_pair in enumerate(edges):
        corner_degrees[edge_pair[0]] = corner_degrees.setdefault(edge_pair[0], 0) + 1
        corner_degrees[edge_pair[1]] = corner_degrees.setdefault(edge_pair[1], 0) + 1
    good_ids = [i for i in range(len(corners)) if i in corner_degrees]
    if len(good_ids) == len(corners):
        return corners, confs, edges
    else:
        good_corners = corners[good_ids]
        good_confs = confs[good_ids]
        id_mapping = {value: idx for idx, value in enumerate(good_ids)}
        new_edges = list()
        for edge_pair in edges:
            new_pair = (id_mapping[edge_pair[0]], id_mapping[edge_pair[1]])
            new_edges.append(new_pair)
        new_edges = np.array(new_edges)
        return good_corners, good_confs, new_edges


def process_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = skimage.img_as_float(img)
    img = img.transpose((2, 0, 1))
    img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
    img = torch.Tensor(img).cuda()
    img = img.unsqueeze(0)
    return img


def plot_heatmap(results, filename):
    y, x = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))

    z = results[::-1, :]
    z = z[:-1, :-1]

    fig, ax = plt.subplots()

    c = ax.pcolormesh(y, x, z, cmap='RdBu', vmin=0, vmax=1)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    fig.savefig(filename)
    plt.close()


def convert_annot(annot):
    corners = np.array(list(annot.keys()))
    corners_mapping = {tuple(c): idx for idx, c in enumerate(corners)}
    edges = set()
    for corner, connections in annot.items():
        idx_c = corners_mapping[tuple(corner)]
        for other_c in connections:
            idx_other_c = corners_mapping[tuple(other_c)]
            if (idx_c, idx_other_c) not in edges and (idx_other_c, idx_c) not in edges:
                edges.add((idx_c, idx_other_c))
    edges = np.array(list(edges))
    gt_data = {
        'corners': corners,
        'edges': edges
    }
    return gt_data

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
    pred_corners = np.stack(local_maximas, axis=-1)[:, [1, 0]]  
    if pred_corners.shape[0] > max_corner_num:
        indices = np.random.choice(pred_corners.shape[0], max_corner_num, replace=False)
        pred_corners = pred_corners[indices]
    if pred_corners.shape[0] < max_corner_num:
        num_padding = max_corner_num - pred_corners.shape[0]
    
        additional_corners = np.ones((num_padding, 2), dtype=pred_corners.dtype) * 255
      
        pred_corners = np.vstack((pred_corners, additional_corners))

    pred_corners = np.concatenate([pred_corners,pred_corners], axis=0)
    ind = np.lexsort(pred_corners.T)  
    pred_corners = pred_corners[ind]   
    return pred_corners


def get_args_parser():
    parser = argparse.ArgumentParser('Holistic edge attention transformer', add_help=False)
    parser.add_argument('--dataset', default='outdoor',
                        help='the dataset for experiments, outdoor/s3d_floorplan')
    parser.add_argument('--checkpoint_path', default='',
                        help='path to the checkpoints of the model')
    parser.add_argument('--image_size', default=256, type=int)
    
    parser.add_argument('--infer_times', default=3, type=int)
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HEAT inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args.dataset, args.checkpoint_path, args.image_size, infer_times=args.infer_times)