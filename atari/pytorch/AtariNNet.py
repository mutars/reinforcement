import sys

import math

sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class AtariNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.observation_shape = game.getBoardSize()
        # assert self.observation_shape.size == 3
        self.action_size = game.getActionSize()
        self.args = args

        super(AtariNNet, self).__init__()
        self.conv1 = nn.Conv2d(args.num_backward_frames * self.observation_shape[2], args.num_channels, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, (3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, (3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, (3, 3), stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.view2 = math.floor(self.args.num_channels * (math.floor(math.floor(math.floor(self.observation_shape[1]/2)/2)/2)) * (math.floor(math.floor(math.floor(self.observation_shape[0]/2)/2)/2)))
        self.fc1 = nn.Linear(self.view2, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

        self.view1 = [ self.args.num_backward_frames * self.observation_shape[2], self.observation_shape[1],
                       self.observation_shape[0]]


    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1,self.view1[0],self.view1[1],self.view1[2])  # batch_size x 1 x board_x x board_y
        s = self.pool1(F.relu(self.bn1(self.conv1(s))))  # batch_size x num_channels x board_x x board_y
        s = self.pool2(F.relu(self.bn2(self.conv2(s))))  # batch_size x num_channels x board_x x board_y
        s = self.pool3(F.relu(self.bn3(self.conv3(s))))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.view2)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), F.tanh(v)
