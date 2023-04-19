import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import math
model_folder = r'C:\Users\user\Documents\GitHub\smplx'
model_type = 'smpl'
plot_joints = 'true'
use_face_contour = False
gender = 'female'
ext = 'npz'
# plotting_module = 'matplotlib'
num_betas = 10
num_expression_coeffs = 10
# sample_shape = True
# sample_expression = True

SMPL_2_xR=[
	2,
	31,
	61,
	62,
	27,
	57,
	63,
	4,
	34,
	64,
	29,
	59,
	0,
	28,
	58,
	1,
	3,
	33,
	5,
	35,
	6,
	36,
	11,
	41
	]

skip_num = [13, 14]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		
class CustomLoss_joint(nn.Module):
	def __init__(self, weight=None,
				 gamma=2., reduction='mean'):
		nn.Module.__init__(self)
		self.weight = weight
		self.gamma = gamma
		self.reduction = reduction
		self.SMPL= smplx.create(model_folder, model_type=model_type,
						 gender=gender, use_face_contour=use_face_contour,
						 num_betas=num_betas,
						 num_expression_coeffs=num_expression_coeffs,
						 ext=ext)
		self.l2loss=nn.MSELoss()
		# self.cos=nn.CosineSimilarity()
		

	def joint_MSE(self, input_tensor, target_tensor):
		batch_size=input_tensor.shape[0]
		pred_tensor=[]
		custom_target_tensor=[]
		for batch,vector_82 in enumerate(input_tensor):
			target_joints=target_tensor[batch]
			target_joints=target_joints.T
			temp_joints=[]
			go=vector_82[:3].unsqueeze(0).float().to('cpu')
			pose=vector_82[3:72].unsqueeze(0).float().to('cpu')
			shape=vector_82[72:].unsqueeze(0).float().to('cpu')
			output = self.SMPL(betas=shape,global_orient=go,body_pose=pose,return_verts=True)
			pred_joints=output.joints[0]
			pred_tensor.append(pred_joints[:22]) # 23까지 바꿀것
			for xR_idx in SMPL_2_xR:
				temp_joints.append(target_joints[xR_idx])
			custom_target_tensor.append(torch.stack(temp_joints,0))

		pred_tensor=torch.stack(pred_tensor,0)
		custom_target_tensor=torch.stack(custom_target_tensor,0).to('cpu')
		return self.l2loss(pred_tensor,custom_target_tensor),pred_tensor,custom_target_tensor

	def find_p_vec(self,ktree,idx,joint,x,y,z):
		if ktree[idx]==-1:
			return x,y,z
		temp_j=(x,y,z)-joint[ktree[idx]]
		self.find_p_vec(ktree,ktree[idx],joint,*temp_j)
	"""
	def cosine_similarity(self, pred_joints,target_joints):
		error=0
		pred_list=[]
		target_list=[]
		ktree=[-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13,
				14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32,
				20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49,
				50, 21, 52, 53]
		for idx,(pred_,target_) in enumerate(zip(pred_joints,target_joints)):
			pred_list.append(self.find_p_vec(ktree,idx,pred_joints,*pred_joints[idx]))
			target_list.append(self.find_p_vec(ktree,idx,target_joints,*target_joints[idx]))

		
		pred_list=torch.tensor(pred_list).to(device)
		target_list=torch.tensor(target_list).to(device)
		error=nn.CosineSimilarity()(pred_list,target_list).abs().mean()

		return error
	"""
	def forward(self, input_tensor, target_tensor):
		"""
		input_tensor.shape
		torch.Size([10, 82])
		target_joints.shape
		torch.Size([10, 3, 65])
		"""
		error,pred_joints,target_joints=self.joint_MSE(input_tensor, target_tensor)
		error+=self.cosine_similarity(pred_joints,target_joints)
		return error

class Depth_loss(nn.Module):
	def __init__(self, weight=None,
				 gamma=2., reduction='mean'):
		nn.Module.__init__(self)
		self.loss=nn.MSELoss()

	def forward(self,x,label):
		res=None
		res=self.loss(x,label)
		return res

class kp_3d_loss(nn.Module):
	def __init__(self) -> None:
		nn.Module.__init__(self)
		self.loss = nn.MSELoss()
	def forward(self,x,label):
		res=None
		res=self.loss(x,label)
		return res

class kp_2d_loss(nn.Module):
	def __init__(self) -> None:
		nn.Module.__init__(self)
		self.loss=nn.MSELoss()
	
	def forward(self,x,label):
		res=None
		res=self.loss(x,label)
		return res

class cam_loss(nn.Module):
	def __init__(self) -> None:
		nn.Module.__init__(self)
		self.loss=nn.MSELoss()
	
	def forward(self,x,label):
		res=None
		res=self.loss(x,label)
		return res

class heatmap_loss(nn.Module):
	def __init__(self) -> None:
		nn.Module.__init__(self)
		self.loss = nn.MSELoss()

	def forward(self,x,label):
		res = None
		res = self.loss(x,label)
		return res

# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
	def __init__(self, weight=None,
				 gamma=2., reduction='mean'):
		nn.Module.__init__(self)
		self.weight = weight
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, input_tensor, target_tensor):
		log_prob = F.log_softmax(input_tensor, dim=-1)
		prob = torch.exp(log_prob)
		return F.nll_loss(
			((1 - prob) ** self.gamma) * log_prob,
			target_tensor,
			weight=self.weight,
			reduction=self.reduction
		)


class LabelSmoothingLoss(nn.Module):
	def __init__(self, classes=3, smoothing=0.0, dim=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.cls = classes
		self.dim = dim

	def forward(self, pred, target):
		pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
	def __init__(self, classes=3, epsilon=1e-7):
		super().__init__()
		self.classes = classes
		self.epsilon = epsilon

	def forward(self, y_pred, y_true):
		assert y_pred.ndim == 2
		assert y_true.ndim == 1
		y_true = F.one_hot(y_true, self.classes).to(torch.float32)
		y_pred = F.softmax(y_pred, dim=1)

		tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
		tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
		fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
		fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

		precision = tp / (tp + fp + self.epsilon)
		recall = tp / (tp + fn + self.epsilon)

		f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
		f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
		return 1 - f1.mean()


_criterion_entrypoints = {
	'cross_entropy': nn.CrossEntropyLoss,
	'focal': FocalLoss,
	'label_smoothing': LabelSmoothingLoss,
	'f1': F1Loss,
	'CustomLoss_joint' : CustomLoss_joint,
	'Depth_loss' : Depth_loss,
	'2d_loss' : kp_2d_loss,
	'3d_loss' : kp_3d_loss,
	'cam_loss' : cam_loss,
	'heatmap_loss' : heatmap_loss,
}


def criterion_entrypoint(criterion_name):
	return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
	return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
	if is_criterion(criterion_name):
		create_fn = criterion_entrypoint(criterion_name)
		criterion = create_fn(**kwargs)
	else:
		raise RuntimeError('Unknown loss (%s)' % criterion_name)
	return criterion
