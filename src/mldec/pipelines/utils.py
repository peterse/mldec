import torch


def gpu_init_pytorch(gpu_num):
	'''
		Initialize device
	'''
	# torch.cuda.set_device(int(gpu_num))
	if torch.cuda.is_available():
		device = torch.device("cuda:{}".format(gpu_num))
	else:
		device = torch.device("cpu")
	return device
