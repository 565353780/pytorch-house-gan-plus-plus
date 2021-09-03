import os
import numpy as np
import math
import sys
import random
import time

from dataset.floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.utils import save_image

from PIL import Image, ImageDraw, ImageFont
import svgwrite
from models.models import Generator
# from models.models_improved import Generator

from misc.utils import _init_input, ID_COLOR, draw_masks, draw_graph, estimate_graph
from collections import defaultdict

import matplotlib.pyplot as plt
import cv2

import networkx as nx
import glob
import webcolors

class HouseGanPlusPlusDetector:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self.device = None
        self.checkpoint = None
        self.model = None
        return

    def loadModel(self, checkpoint):
        self.checkpoint = checkpoint

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        self.model = Generator()
        if self.checkpoint is not None:
            self.model.load_state_dict(torch.load(self.checkpoint, map_location='cpu'), strict=True)
        self.model.eval()
        self.model.to(self.device)
        return

    def detect(self, graph, prev_state=None):
        z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)

        with torch.no_grad():
            masks = self.model(z.to(self.device), given_masks_in.to(self.device), given_nds.to(self.device), given_eds.to(self.device))
            masks = masks.detach().cpu().numpy()
        return masks

    def test(self):
        dataset_path = "./data/sample_list.txt"
        batch_size = 1
        out = "./dump"
        # Create output dir
        os.makedirs(out, exist_ok=True)

        # initialize dataset iterator
        fp_dataset_test = FloorplanGraphDataset(
            dataset_path,
            transforms.Normalize(mean=[0.5], std=[0.5]),
            split='test')
        fp_loader = torch.utils.data.DataLoader(
            fp_dataset_test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=floorplan_collate_fn)

        # optimizers
        #  Tensor = torch.FloatTensor.to(self.device)

        globalIndex = 0
        for i, sample in enumerate(fp_loader):

            # draw real graph and groundtruth
            mks, nds, eds, _, _ = sample
            real_nodes = np.where(nds.detach().cpu()==1)[-1]
            graph = [nds, eds]
            true_graph_obj, graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
            graph_im.save('./{}/graph_{}.png'.format(out, i)) # save graph

            # add room types incrementally
            _types = sorted(list(set(real_nodes)))
            selected_types = [_types[:k+1] for k in range(10)]
            os.makedirs('./{}/'.format(out), exist_ok=True)
            _round = 0
            
            # initialize layout
            state = {'masks': None, 'fixed_nodes': []}
            masks = self.detect(graph, state)
            im0 = draw_masks(masks.copy(), real_nodes)
            im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0 
            # save_image(im0, './{}/fp_init_{}.png'.format(out, i), nrow=1, normalize=False) # visualize init image

            # generate per room type
            for _iter, _types in enumerate(selected_types):
                _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
                    if len(_types) > 0 else np.array([]) 
                state = {'masks': masks, 'fixed_nodes': _fixed_nds}
                masks = self.detect(graph, state)
                
            # save final floorplans
            imk = draw_masks(masks.copy(), real_nodes)
            imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0 
            save_image(imk, './{}/fp_final_{}.png'.format(out, i), nrow=1, normalize=False)
     
        return

if __name__ == '__main__':
    #  n_cpu = 20
    checkpoint = "./checkpoints/pretrained.pth"

    houseganplusplus_detector = HouseGanPlusPlusDetector()

    houseganplusplus_detector.loadModel(checkpoint)

    houseganplusplus_detector.test()

