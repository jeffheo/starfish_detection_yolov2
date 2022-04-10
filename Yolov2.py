import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Darknet import Darknet19, conv_bn_relu
import config as cfg

class Yolov2(torch.nn):
	def __init__(self, backbone_path='./e4_checkpoint.pth.tar'):
		# backbone: filename of loaded model, Darknet19
		self.backbone = Darknet19()
		self.backbone.load_state_dict(torch.load(backbone_path))

		layers = [self.backbone.layer_0, self.backbone.layer_1, self.backbone.layer_2,\
			 self.backbone.layer_3, self.backbone.layer_4, self.backbone.layer_5]

		self.layer_0 = nn.Sequential(*layers[:-1])
		self.layer_1 = layers[-1]
		self.layer_2 = nn.Sequential(conv_bn_relu(1024, 1024, 3), conv_bn_relu(1024, 1024, 3))
		# skip connection happens here
		self.downsample = conv_bn_relu(512, 64, 1)
		self.layer_3 = nn.Sequential(conv_bn_relu(1280, 1024, 3), nn.Conv2d(1024, 25, kernel_size=1, stride=1))


	self.forward(self, x):
		x = self.layer_0(x) # 512 * 14 *14
		shortcut = self.downsample(x).view(-1, 256, 7, 7)
		x = self.layer_1(x) # 1024 * 7 * 7
		x = self.layer_2(x) # 1024 * 7 * 7
		x = torch.cat((x, shortcut), 1) # 1280 * 7 * 7
		x = self.layer_3(x)

		return x
