import torch
import torch.nn as nn
import torch.nn.functional as F


class heatmap_loss_(nn.Module):
	def __init__(self) -> None:
		nn.Module.__init__(self)
		self.loss = nn.MSELoss()

	def forward(self,x,label):
		res = None
		res = self.loss(x,label)
		return res
	
class cosine_similarity_loss_(nn.Module):
	def __init__(self) -> None:
		nn.Module.__init__(self)
		self.loss = nn.CosineSimilarity()
		
	def forward(self,x,label):
		res = None
		res = torch.sum((1-self.loss(x,label))/2)
		return res
	
class kp_3d_loss_(nn.Module):
	def __init__(self) -> None:
		nn.Module.__init__(self)
		self.loss = nn.MSELoss()

	def forward(self,x,label):
		res = None	
		res = self.loss(x,label)
		return res

class depth_loss_(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.loss=nn.MSELoss()

    def forward(self,x,label):
        res=None
        res=self.loss(x,label)
        return res
    
class silhouette_loss_(nn.Module):
	def __init__(self) -> None:
		nn.Module.__init__(self)
		self.loss = nn.BCEWithLogitsLoss()

	def forward(self,x,label):
		res = None
		res = self.loss(x,label)
		return res