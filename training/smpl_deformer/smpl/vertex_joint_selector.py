# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn

from .utils import to_tensor


class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=None,
                 use_hands=True,
                 use_feet_keypoints=True,
                 use_face_keypoints=False, **kwargs):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        if use_face_keypoints:
            face_keyp_idxs = np.array([
                vertex_ids['nose'],
                vertex_ids['reye'],
                vertex_ids['leye'],
                vertex_ids['rear'],
                vertex_ids['lear']], dtype=np.int64)
            extra_joints_idxs.append(face_keyp_idxs)

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int64)
            extra_joints_idxs.append(feet_keyp_idxs)

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            tips_idxs = np.array(tips_idxs, dtype=np.int64)
            extra_joints_idxs.append(tips_idxs)

        if len(extra_joints_idxs) > 0:
            extra_joints_idxs = np.concatenate(extra_joints_idxs)
        else:
            extra_joints_idxs = np.array([], dtype=np.int64)

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        if self.extra_joints_idxs.numel() == 0:
            return joints

        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([joints, extra_joints], dim=1)
        return joints
