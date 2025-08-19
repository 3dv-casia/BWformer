import os
import sys
import glob
import numpy as np
from collections import defaultdict
import cv2
import json
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
def merge_adjacent_regions(labels, region_masks):
    region_areas = [np.sum(region_mask) for region_mask in region_masks]

    for label in range(1, labels):
        current_mask = region_masks[label]
        adjacent_labels = set()

        for dy, dx in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            y, x = np.where(current_mask == 1)
            y_adjacent = np.clip(y + dy, 0, labels.shape[0] - 1)
            x_adjacent = np.clip(x + dx, 0, labels.shape[1] - 1)
            adjacent_labels.update(labels[y_adjacent, x_adjacent])

        adjacent_labels.remove(label)  

        max_area = 0
        max_area_label = None
        for adj_label in adjacent_labels:
            area = region_areas[adj_label]
            if area > max_area:
                max_area = area
                max_area_label = adj_label

        if max_area_label is not None:
            labels[labels == label] = max_area_label

    return labels
def detect_outliers(points, k=5, threshold_factor=20):
   
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points)
    distances, _ = neigh.kneighbors(points)
    mean_distances = np.mean(distances, axis=1)
    
  
    threshold = threshold_factor * np.mean(mean_distances)

    outliers_idx = np.where(mean_distances > threshold)[0]
    
    return outliers_idx


def render(corners, edges, render_pad=0, edge_linewidth=2, corner_size=3, scale=1.):
    size = int(256 * scale)
    mask = np.ones((2, size, size)) * render_pad

    corners = np.round(corners.copy() * scale).astype(int)
    for edge_i in range(edges.shape[0]):
        a = edges[edge_i, 0]
        b = edges[edge_i, 1]
        mask[0] = cv2.line(mask[0], (int(corners[a, 0]), int(corners[a, 1])),
                           (int(corners[b, 0]), int(corners[b, 1])), 1.0, thickness=edge_linewidth)
    for corner_i in range(corners.shape[0]):
        mask[1] = cv2.circle(mask[1], (int(corners[corner_i, 0]), int(corners[corner_i, 1])), corner_size, 1.0, -1)

    return mask

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
# load pc and wireframe
def load_files(pc_dir, train_list_path, test_list_path):
    wf_dir = "xxx/wf"

    with open(train_list_path, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    with open(test_list_path, 'r') as f:
        test_names = [line.strip() for line in f if line.strip()]
    
    all_names = sorted(set(train_names + test_names))  # 合并并去重

    pc_files = [os.path.join(pc_dir, name + '.ply') for name in all_names]
    wireframe_files = [os.path.join(wf_dir,  name + '.obj') for name in all_names]

    return pc_files, wireframe_files

def load_wireframe(wireframe_file):
    vertices = []
    edges = set()
    with open(wireframe_file) as f:
        for lines in f.readlines():
            line = lines.strip().split(' ')
            if line[0] == 'v':
                vertices.append(line[1:])
            else:
                if line[0] == '#':
                    continue
                obj_data = np.array(line[1:], dtype=np.int32).reshape(2) - 1
                edges.add(tuple(sorted(obj_data)))
    vertices = np.array(vertices, dtype=np.float64)
    edges = np.array(list(edges))
    return vertices, edges

def proj_img(pc, index, output_dir):

    x_pixels = np.floor(pc[:, 0]).astype(int)
    y_pixels = np.floor(pc[:, 1]).astype(int)


    image = np.zeros((256, 256, 3), dtype=np.uint8)


    for i in range(len(pc)):
        if image[y_pixels[i], x_pixels[i]][0] == 0:
            image[y_pixels[i], x_pixels[i]] = [pc[i, 2], pc[i, 2], pc[i, 2]]
        else:
            if pc[i, 2] < image[y_pixels[i], x_pixels[i]][0]:
                image[y_pixels[i], x_pixels[i]] = [pc[i, 2], pc[i, 2], pc[i, 2]]


    cv2.imwrite(os.path.join(output_dir, f"{index}.jpg"), image)
    return image

def proj_maskimg(pc, index, output_dir):

    x_pixels = np.floor(pc[:, 0]).astype(int)
    y_pixels = np.floor(pc[:, 1]).astype(int)

    image = np.zeros((256, 256, 3), dtype=np.uint8)


    for i in range(len(pc)):
        
        image[y_pixels[i], x_pixels[i]] = [pc[:, 2], pc[:, 2], pc[:, 2]]


    cv2.imwrite(os.path.join(output_dir, f"{index}.jpg"), image)

def visualize_image(image, data_dict, output_dir, index):
    image = image.copy()  # get a new copy of the original image

    for vertex, connected_vertices in data_dict.items():

        x, y, _ = vertex
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # 红色圆点

    for vertex, connected_vertices in data_dict.items():

        x, y, _ = vertex

        for connected_vertex in connected_vertices:
            x2, y2, _ = connected_vertex
            cv2.line(image, (int(x2), int(y2)), (int(x), int(y)), (255, 0, 0), 2)  # 蓝色线段

    cv2.imwrite(os.path.join(output_dir, f"{index}_vis.jpg"), image)

def main():
    pc_dir = "xxx/pc"
    proj_dir = "xxx/rgb"
    npy_dir = "xxx/annot"
    train_list_path = "xxx/train_list.txt"
    test_list_path = "xxx/test_list.txt"
    
    pc_files, wireframe_files = load_files(pc_dir, train_list_path, test_list_path)
    names = []
    annotations = []
    annotation_id = 0
    images = []
    image_id = 0

    for index in range(len(pc_files)):
        print(index)
        image_id = image_id + 1
        pc_file = pc_files[index]
        pcd = o3d.io.read_point_cloud(pc_file)  # 读取 .ply 文件
        point_cloud = np.asarray(pcd.points) 
        name =  os.path.splitext(os.path.basename(pc_file))[0]
        print(name)
    
        names.append(name)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

        # ------------------------------- Wireframe ------------------------------
        # load wireframe
        wireframe_file = wireframe_files[index]
        wf_vertices, wf_edges = load_wireframe(wireframe_file)

        
        centroid = np.mean(point_cloud[:, 0:3], axis=0)
        point_cloud[:, 0:3] -= centroid
        wf_vertices -= centroid
        max_distance = np.max(np.linalg.norm(np.vstack((point_cloud[:, 0:3], wf_vertices)), axis=1))

                
        point_cloud[:, 0:3] /= (max_distance)
        point_cloud[:, 0:3] = (point_cloud[:, 0:3] +  np.ones_like(point_cloud[:, 0:3]))  * 127.5

        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(point_cloud)

        wf_vertices /= (max_distance)

        wf_vertices = (wf_vertices +  np.ones_like(wf_vertices))  *  127.5
     
        vertex_con = {}

        for edge in wf_edges:
            vertex1, vertex2 = edge
            if vertex1 not in vertex_con:
        
                vertex_con[vertex1] = [vertex2]
            else:
                vertex_con[vertex1].append(vertex2)

            if vertex2 not in vertex_con:
                vertex_con[vertex2] = [vertex1]
            else:
                vertex_con[vertex2].append(vertex1)

        vertex_connections = {}
        for vertex, edge in vertex_con.items():
            vertex_connections[tuple(wf_vertices[vertex])] = []
            for edge_vertex in edge:
                vertex_connections[tuple(wf_vertices[vertex])].append(np.array(wf_vertices[edge_vertex]))
        vertex_connections1 = {}
        for vertex, edge in vertex_con.items():
            vertex_connections1[tuple([wf_vertices[vertex][0], wf_vertices[vertex][1], wf_vertices[vertex][2]])] = []
            for edge_vertex in edge:
                vertex_connections1[tuple([wf_vertices[vertex][0], wf_vertices[vertex][1], wf_vertices[vertex][2]])].append(np.array([wf_vertices[edge_vertex][0], wf_vertices[edge_vertex][1], wf_vertices[edge_vertex][2]]))
        npypath = os.path.join(npy_dir, f"{name}.npy")
        np.save(npypath, vertex_connections1)
        image = proj_img(point_cloud, name, proj_dir)
        
if __name__ == '__main__':
    main()
        



