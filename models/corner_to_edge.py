import torch
import numpy as np
import scipy.ndimage.filters as filters
import cv2
import itertools

NEIGHBOUR_SIZE = 5
MATCH_THRESH = 10
LOCAL_MAX_THRESH = 0.01
viz_count = 0

all_combibations = dict()
for length in range(2, 351):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 2)))
    all_combibations[length] = combs

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
    
def calculate_distance(point1, point2):
    
    return np.linalg.norm(point1 - point2)

def prepare_edge_data(corner_coord,corner_logits, annots, images, max_corner_num):

    bs = corner_coord.shape[0]
   
    all_results = list()

    for b_i in range(bs):
        annot = annots[b_i]
        output_coord = corner_coord[b_i]
        output_logit = corner_logits[b_i].sigmoid()
        
        results = process_each_sample({'annot': annot, 'output_coord':output_coord,'output_logit':output_logit, 'viz_img': images[b_i]}, max_corner_num)
        all_results.append(results)

    processed_corners = [item['corners'] for item in all_results]
    edge_coords = [item['edges'] for item in all_results]
    edge_labels = [item['labels'] for item in all_results]

    edge_info = {
        'edge_coords': edge_coords,
        'edge_labels': edge_labels,
        'processed_corners': processed_corners
    }

    edge_data = collate_edge_info(edge_info)
    return edge_data


def process_annot(annot, do_round=True):
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


def process_each_sample(data, max_corner_num):
    annot = data['annot']
    output_coord = data['output_coord']
    output_logit = data['output_logit']


    preds = output_coord.detach().cpu().numpy()

    pred_corners = preds
    pred_logits = output_logit.detach().cpu().numpy()


    processed_corners, edges, labels = get_edge_label_mix_gt(pred_corners, pred_logits, annot, max_corner_num)
   

    results = {
        'corners': processed_corners,
        'edges': edges,
        'labels': labels,
    }
    return results



def get_edge_label_mix_gt(pred_corners,pred_logits, annot, max_corner_num):
    new_corners = []
    new_hs = []
    new_corners = nms_group(pred_corners, pred_logits, dist_thresh=5.0, score_thresh=0.01)
    pred_corners = np.array(new_corners)
    ind = np.lexsort(pred_corners.T)  
    pred_corners = pred_corners[ind] 



    gt_corners, edge_pairs, corner_degrees = process_annot(annot)

  
    output_to_gt = dict()
    gt_to_output = dict()
    diff = np.sqrt(((pred_corners[:, None] - gt_corners) ** 2).sum(-1))
    diff = diff.T

    if len(pred_corners) > 0:
        for target_i, target in enumerate(gt_corners):
            dist = diff[target_i]
            if len(output_to_gt) > 0:
                dist[list(output_to_gt.keys())] = 1000 
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
        else:
            all_corners = np.concatenate([all_corners, nm_pred_corners[:(max_corner_num - len(gt_corners)), :]], axis=0)

    processed_corners, edges, edge_ids, labels = _get_edges(all_corners, edge_pairs)

    return processed_corners, edges, labels


def _get_edges(corners, edge_pairs):
    ind = np.lexsort(corners.T)
    corners = corners[ind]  # sorted by y, then x
    corners = corners.round()
    id_mapping = {old: new for new, old in enumerate(ind)}

    all_ids = all_combibations[len(corners)]
    edges = corners[all_ids]
    labels = np.zeros(edges.shape[0])

    N = len(corners)
    edge_pairs = [(id_mapping[p[0]], id_mapping[p[1]]) for p in edge_pairs]
    edge_pairs = [p for p in edge_pairs if p[0] < p[1]]
    pos_ids = [int((2 * N - 1 - p[0]) * p[0] / 2 + p[1] - p[0] - 1) for p in edge_pairs]
    labels[pos_ids] = 1

    edge_ids = np.array(all_ids)
    return corners, edges, edge_ids, labels


def collate_edge_info(data):
    batched_data = {}
    lengths_info = {}
    for field in data.keys():
        batch_values = data[field]
        all_lens = [len(value) for value in batch_values]
        max_len = max(all_lens)
        pad_value = 0
        batch_values = [pad_sequence(value, max_len, pad_value) for value in batch_values]
        batch_values = np.stack(batch_values, axis=0)

        if field in ['edge_coords', 'edge_labels', 'gt_values']:
            batch_values = torch.Tensor(batch_values).long()
        if field in ['processed_corners', 'edge_coords']:
            lengths_info[field] = all_lens
        batched_data[field] = batch_values

    for field, lengths in lengths_info.items():
        lengths_str = field + '_lengths'
        batched_data[lengths_str] = torch.Tensor(lengths).long()
        mask = torch.arange(max(lengths))
        mask = mask.unsqueeze(0).repeat(batched_data[field].shape[0], 1)
        mask = mask >= batched_data[lengths_str].unsqueeze(-1)
        mask_str = field + '_mask'
        batched_data[mask_str] = mask

    return batched_data


def pad_sequence(seq, length, pad_value=0):
    if len(seq) == length:
        return seq
    else:
        pad_len = length - len(seq)
        if len(seq.shape) == 1:
            if pad_value == 0:
                paddings = np.zeros([pad_len, ])
            else:
                paddings = np.ones([pad_len, ]) * pad_value
        else:
            if pad_value == 0:
                paddings = np.zeros([pad_len, ] + list(seq.shape[1:]))
            else:
                paddings = np.ones([pad_len, ] + list(seq.shape[1:])) * pad_value
        padded_seq = np.concatenate([seq, paddings], axis=0)
        return padded_seq


def get_infer_edge_pairs(corners, confs):
    ind = np.lexsort(corners.T)
    
    corners = corners[ind]  # sorted by y, then x
    confs = confs[ind]

    edge_ids = all_combibations[len(corners)]
    edge_coords = corners[edge_ids]
    

    edge_coords = torch.tensor(np.array(edge_coords)).unsqueeze(0).long()
  
    mask = torch.zeros([edge_coords.shape[0], edge_coords.shape[1]]).bool()
    edge_ids = torch.tensor(np.array(edge_ids))
    return corners, confs, edge_coords, mask, edge_ids



