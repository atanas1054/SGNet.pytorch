import numpy as np
import random
import pickle as pkl
import math
import os
import copy
import torch
import torch.utils.data as data
from lib.dataloaders import build_dataset

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def random_translate(trajs_obs, trajs_pred):
    """
            A random translation
            Input parameter:
              NxMx6
              trajs - should be Ntrajectories x Mtime_step x xy_coords, xy_velocity, xy_acceleration
            Output parameter:
              new_trajs - the matrix storing the new, translated trajectories.
            """

    translated_pred = np.zeros(trajs_pred.shape)
    translated_obs = np.zeros(trajs_obs.shape)

    low = -2
    high = 2

    for i in range(trajs_obs.shape[0]):
        translation = np.random.rand() * (high - low) + low
        translated_obs[i, :, 0] = trajs_obs[i, :, 0] + translation
        translated_obs[i, :, 1] = trajs_obs[i, :, 1] + translation
        translated_obs[i, :, 2] = trajs_obs[i, :, 2] + translation
        translated_obs[i, :, 3] = trajs_obs[i, :, 3] + translation
        translated_obs[i, :, 4] = trajs_obs[i, :, 4] + translation
        translated_obs[i, :, 5] = trajs_obs[i, :, 5] + translation

        translated_pred[i, :, :, 0] = trajs_pred[i, :, :, 0] + translation
        translated_pred[i, :, :, 1] = trajs_pred[i, :, :, 1] + translation

    return translated_obs, translated_pred

def random_rotate(trajs_obs, trajs_pred, origin=None):
        """
        A random rotation angle would be generated and the 2D rotation
        of the generated angle about the origin would be applied to
        the trajectories stored in trajs. As the rotation is about
        the origin, the trajectories should have been normalized
        so that they are defined around the origin.
        Input parameter:
          NxMx6
          trajs - should be Ntrajectories x Mtime_step x xy_coords, xy_velocity, xy_acceleration
        Output parameter:
          new_trajs - the matrix storing the new, rotated trajectories.
        """
        rotated_obs = np.zeros(trajs_obs.shape)
        rotated_pred = np.zeros(trajs_pred.shape)

        pi2 = 2 * np.pi
        low = -0.25
        high = 0.25

        if origin is None:
            orig = np.zeros((2, 1))
        else:
            orig = np.array(origin).reshape(2, 1)
        for i in range(trajs_obs.shape[0]):
            angle = np.random.rand() * (high - low) + low * pi2
            cangle, sangle = math.cos(angle), math.sin(angle)
            rot_mat = np.array([[cangle, -sangle], [sangle, cangle]])
            # rotated[i, :, [0,1]] = np.matmul(rot_mat, trajs[i, :, [0,1]].T - orig).T
            # rotated[i, :, [2,3]] = np.matmul(rot_mat, trajs[i, :, [2,3]].T - orig).T
            # rotated[i, :, [4,5]] = np.matmul(rot_mat, trajs[i, :, [4,5]].T - orig).T

            rotated_obs[i, :, [0, 1]] = np.matmul(rot_mat, trajs_obs[i, :, [0, 1]])
            rotated_obs[i, :, [2, 3]] = np.matmul(rot_mat, trajs_obs[i, :, [2, 3]])
            rotated_obs[i, :, [4, 5]] = np.matmul(rot_mat, trajs_obs[i, :, [4, 5]])

            rotated_pred[i] = trajs_pred[i].dot(rot_mat.T)

        return rotated_obs, rotated_pred


def build_data_loader(args, phase='train',batch_size=None):
    data_loaders = data.DataLoader(
        dataset=build_dataset(args, phase),
        batch_size=args.batch_size if batch_size is None else batch_size,
        shuffle=phase=='train',
        num_workers=args.num_workers,
        collate_fn=my_collate_fn if batch_size is not None else None)

    return data_loaders

def my_collate_fn(batch):
    return batch[0]

def cxcywh_to_x1y1x2y2(boxes):
    '''
    Params:
        boxes:(Cx, Cy, w, h)
    Returns:
        (x1, y1, x2, y2 or tlbr
    '''
    new_boxes = np.zeros_like(boxes)
    new_boxes[...,0] = boxes[...,0] - boxes[...,2]/2
    new_boxes[...,1] = boxes[...,1] - boxes[...,3]/2
    new_boxes[...,2] = boxes[...,0] + boxes[...,2]/2
    new_boxes[...,3] = boxes[...,1] + boxes[...,3]/2
    return new_boxes


def bbox_normalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[:,0] /= W
    new_bbox[:,1] /= H
    new_bbox[:,2] /= W
    new_bbox[:,3] /= H
    
    return new_bbox

def bbox_denormalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[..., 0] *= W
    new_bbox[..., 1] *= H
    new_bbox[..., 2] *= W
    new_bbox[..., 3] *= H
    
    return new_bbox


# FLow loading code adapted from:
# http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

def load_flow(flow_folder):
    '''
    Given video key, load the corresponding flow file
    '''
    flow_files = sorted(glob.glob(flow_folder + '*.flo'))
    flows = []
    for file in flow_files:
        flow = read_flo(file)
        flows.append(flow)
    return flows

TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = int(np.fromfile(f, np.int32, count=1))
    h = int(np.fromfile(f, np.int32, count=1))
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))	
    f.close()

    return flow

