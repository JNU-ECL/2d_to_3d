# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from base import SetType

from mocap import Mocap
from utils import config, ConsoleLogger
from utils import evaluate, io
from model import TempModel

LOGGER = ConsoleLogger("Main")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main():
	"""Main"""

	LOGGER.info('Starting demo...')

	# ------------------- Data loader -------------------


	# let's load data from validation set as example
	data = Mocap(
		config.dataset.test,
		SetType.VAL,
		)
	data_loader = DataLoader(
		data,
		batch_size=config.data_loader.batch_size,
		shuffle=config.data_loader.shuffle)

	# ------------------- Evaluation -------------------

	eval_body = evaluate.EvalBody()
	eval_upper = evaluate.EvalUpperBody()
	eval_lower = evaluate.EvalUpperBody()

	# ------------------- Read dataset frames -------------------

	model = TempModel().to(device)
	model = torch.nn.DataParallel(model,device_ids=[0,1])
	model.eval()

	for it, test_batch in enumerate(data_loader):
		img = test_batch['image']
		p2d = test_batch['fisheye_joints_2d']
		p3d = test_batch['joints_3d_cam']
		action = test_batch['action']
		input_img = {'image':img}
		LOGGER.info('Iteration: {}'.format(it))
		LOGGER.info('Images: {}'.format(img.shape))
		LOGGER.info('p2ds: {}'.format(p2d.shape))
		LOGGER.info('p3ds: {}'.format(p3d.shape))
		LOGGER.info('Actions: {}'.format(action))

		# -----------------------------------------------------------
		pred_output = model(input_img)
		# -----------------------------------------------------------

		# TODO: replace p3d_hat with model preditions
		p3d_hat = pred_output['regressor_dict']['pred_joint']

		# Evaluate results using different evaluation metrices
		y_output = p3d_hat.data.cpu().numpy() * 100
		y_target = p3d.data.cpu().numpy() * 100

		eval_body.eval(y_output, y_target, action)
		eval_upper.eval(y_output, y_target, action)
		eval_lower.eval(y_output, y_target, action)

		# TODO: remove break
		break

	# ------------------- Save results -------------------

	LOGGER.info('Saving evaluation results...')
	res = {'FullBody': eval_body.get_results(),
		   'UpperBody': eval_upper.get_results(),
		   'LowerBody': eval_lower.get_results()}

	io.write_json(config.eval.output_file, res)

	LOGGER.info('Done.')


if __name__ == "__main__":
	main()
