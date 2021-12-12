import argparse
import os
import torch

class Arguments():
	def __int__(self):
		return
	
	def parse(self):
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument('--save_freq', type=int, default=10)
		parser.add_argument('--print_freq', type=int, default=100)
		parser.add_argument('--batch_size', type=int, default=1)
		parser.add_argument('--dataset_path', type=str, default='')
		parser.add_argument('--is_train', action='store_true', default=False)
		parser.add_argument('--is_gpu', action='store_true', default=False)
		parser.add_argument('--dropout', action='store_true', default=False)
		parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
		parser.add_argument('--print_dir', type=str, default='./result')
		parser.add_argument('--name', type=str, default='face2comic')
		parser.add_argument('--output_nc', type=int, default=3)
		parser.add_argument('--input_nc', type=int, default=3)
		parser.add_argument('--n_layers', type=int, default=3)
		parser.add_argument('--n_filter', type=int, default=64)
		parser.add_argument('--n_epochs', type=int, default=150)
		parser.add_argument('--n_epochs_decay', type=int, default=150)
		parser.add_argument('--norm', type=str, default='batch')
		parser.add_argument('--init_type', type=str, default='normal')
		parser.add_argument('--init_gain', type=float, default=0.02)
		parser.add_argument('--beta', type=float, default=0.5)
		parser.add_argument('--epoch_count', type=int, default=1)
		parser.add_argument('--lr', type=float, default=0.0002)
		parser.add_argument('--lr_decay_iters', type=int, default=50)
		parser.add_argument('--lr_policy', type=str, default='linear')
		parser.add_argument('--lambda_A', type=float, default=10.0)
		parser.add_argument('--lambda_B', type=float, default=10.0)
		parser.add_argument('--resize', type=int, default=286)
		parser.add_argument('--crop_size', type=int, default=256)

		self.parser = parser
		return parser.parse_args()
		
