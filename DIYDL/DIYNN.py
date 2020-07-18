"""===============================================
	pytorch from scratch just for fun !!	
=================================================="""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""---------------------------------------------------------------------------------------------------------
	2020/07/18
	- Self-Attention with Linear Complexity: 	https://arxiv.org/pdf/2006.04768.pdf
	- Attention Is All You Need: 				https://arxiv.org/abs/1706.03762
-------------------------------------------------------------------------------------------------------------"""

class SelfAttention(nn.Module):  # (B,n,d) ---> (B,n,d)
	def __init__(self,n,d,k):  # n = 子集合大小(句子長度) , d = 元素維度 , k = 降維處理
		super(SelfAttention, self).__init__()
		# 模型大小
		self.n = n
		self.d = d
		# 學習權重
		self.Q = nn.Linear(d,d,bias=False)
		self.K = nn.Linear(d,d,bias=False)
		self.V = nn.Linear(d,d,bias=False)
		self.E = nn.Linear(n,k,bias=False)
		self.F = nn.Linear(n,k,bias=False)
		self.dropout = nn.Dropout(p=0.1)
	def forward(self,I): # I  = input embeddings (B x n x d) B = batchsize
		#--------------------------------------------------------
		Q = self.Q(I)  # (B,n,d) ---> (B,n,d)
		K = self.K(I)
		V = self.V(I)
		#------------------------------------------------------------
		# Linear Projection
		V = V.transpose(1,2) # (B,n,d) ---> (B,d,n)
		V = self.F(V) #  (B,d,n) ---> (B,d,k) 
		V = V.transpose(1,2) # (B,d,k) ---> (B,k,d)
		K = K.transpose(1,2)
		K = self.E(K) #  (B,d,n) ---> (B,d,k) 
		#-------------------------------------------------------------
		# Scaled and Normalize 
		sqrtd = torch.sqrt(torch.tensor(self.d).type(Q.type())) 
		PBar = torch.matmul(Q,K)/sqrtd  # (B,n,d)(B,d,k) ----> (B,n,k)
		PBar = PBar.softmax(dim=2) #  on dim = 2 ----> (B,n,k)
		PBar = self.dropout(PBar)
		#-------------------------------------------------------------
		# QK x V 
		Head = torch.matmul(PBar,V) # (B,n,k)(B,k,d) ----> (B,n,d) 
		return Head # 輸出的 embedding dim = d 
				

class MultiHeadSelfAttention(nn.Module):  # (B,n,d) ------> (B,n,hd) ----->(B,n,o)
	def __init__(self,h,n,d,k,o):  # h = num_heads , o = output dim
		super(MultiHeadSelfAttention, self).__init__()
		self.h = h
		self.o = o
		self.O = nn.Linear(h*d,o,bias=False)
		self.attns = nn.ModuleList()
		# put Self Attentions !!
		for i in range(self.h):
			self.attns.append(SelfAttention(n,d,k))
	def forward(self,I):  # (B,n,d)
		heads = []
		for i in range(self.h):
			heads.append(self.attns[i](I)) 
		heads = torch.cat(heads,dim=2) # [(B,n,d)] ---> (B,n,hd)
		output = self.O(heads) # (B,n,hd) ----> (B,n,o)
		return output 


"""------------------------------------------------------------------------------------
	2020/07/19
	- Implicit Neural Representations with Periodic Activation Functions
		https://vsitzmann.github.io/siren/
	
----------------------------------------------------------------------------------------"""
class SirenActivation(nn.Module):  # D^n ---> D^m , D = [-1,1]
	def __init__(self,dimI,dimO,w0=1.,c=6.):
		super(SirenActivation,self).__init__()
		W = torch.zeros(dimO,dimI)
		b = torch.zeros(dimO)
		# ----------- 初始化 -------------
		std = 1.0/math.sqrt(dimI)
		w_std = math.sqrt(c)*std/w0
		W.uniform_(-w_std, w_std)
		b.uniform_(-std, std)
		#------------------------------------
		self.w0 = w0
		self.W = nn.Parameter(W)
		self.b = nn.Parameter(b)
        
	def forward(self, x):  # (B,dimI) -----> (B,dimO)
		x =  F.linear(x,self.W, self.b)
		x = torch.sin(self.w0*x)
		return x

class Siren(nn.Module):
	def __init__(self,dims=[2,256,256,256,256,256,3],ws=[30.,1.,1.,1.,1.,1.]):
		super(Siren,self).__init__()
		self.nlayers = len(dims)-1
		self.acts = nn.ModuleList()
		for i in range(self.nlayers):
			self.acts.append(SirenActivation(dimI=dims[i],dimO=dims[i+1],w0=ws[i]))

	def forward(self,x):
		for i in range(self.nlayers):
			x = self.acts[i](x)
		return x

"""------------------------------------------------------------------------
	2020/07/19:
	- https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
---------------------------------------------------------------------------"""
class SkipGram(nn.Module):   #  input :  center = (B) neighbors = (B) negs = (B,neg_size) , output: loss
	def __init__(self,dimV=20000,dimE=300):
		super(SkipGram,self).__init__()
		self.embedI = nn.Embedding(dimV,dimE)
		self.embedO = nn.Embedding(dimV,dimE)
		# Xavier initialization
		initrange = (2.0/(dimV+dimE))**0.5  
		self.embedI.weight.data.uniform_(-initrange, initrange)
		self.embedO.weight.data.uniform_(-0, 0)
		self.log_sigmoid = nn.LogSigmoid()
	
	def forward(self,center,nbrs,negs): 	 
		v =  self.embedI(center) 			 							  # (B) -----> (B,dimE)
		u = self.embedO(nbrs) 				 							  # (B) -----> (B,dimE)
		uv = torch.sum(u*v,dim=1).squeeze()  							  # (B,dimE) -----> (B,1) -----> (B)
		u_neg = self.embedO(negs) 			 							  # (B,neg_size) -----> (B,neg_size,dimE)
		u_negv = torch.bmm(u_neg,v.unsqueeze(2)).squeeze(2) 			  # (B,neg_size,dimE) x (B,dimE,1) = (B,neg_size,1) ---> (B,neg_size)
		positives = self.log_sigmoid(uv)
		negatives = self.log_sigmoid(-torch.sum(u_negv, dim=1)).squeeze() # (B,neg_size) -----> (B,1) -----> (B)
		loss = -(positives + negatives)
		return loss.mean() # batch mean

"""-------------------------------------------------------------------------------------
	2020/07/19  YOLOv3  - DarkNet53
----------------------------------------------------------------------------------------"""
def DarkNet_Conv2d(num_in,num_out,kernel_size,padding,stride):
	return nn.Sequential(
		nn.Conv2d(num_in,num_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
		nn.BatchNorm2d(num_out),
		nn.LeakyReLU()
	)

class DarkNet_Residual(nn.Module):   # input : (B,C,N,N) ----> (B,C,N,N)
	def __init__(self,num_in):
		super(DarkNet_Residual,self).__init__()
		self.conv1 = DarkNet_Conv2d(num_in,num_in//2, kernel_size=1, padding=0,stride=1) # floor[(N-k+2p)/s]+1 = N 
		self.conv2 = DarkNet_Conv2d(num_in//2,num_in, kernel_size=3, padding=1,stride=1) # floor[(N-k+2p)/s]+1 = N  
	def forward(self,x):
		out = self.conv1(x)
		out = self.conv2(out)
		out += x
		return out

def DarkNet_ResBlocks(num_in,num_blocks):
	layers = []
	for i in range(0,num_blocks):
		layers.append(DarkNet_Residual(num_in))
	return nn.Sequential(*layers)


class DarkNet53(nn.Module):   # 6 (conv2d) + [1+2+8+8+4]x2 (res) = 52 convs 
	def __init__(self):     
		super(DarkNet53,self).__init__()
		self.conv1 = DarkNet_Conv2d(3,32,kernel_size=3, padding=1,stride=1) 	# floor[(N-k+2p)/s]+1 = N = 256
		self.conv2 = DarkNet_Conv2d(32,64,kernel_size=3, padding=1,stride=2)   # floor[(N-k+2p)/s]+1 = floor[(N-1)/2]+1 = 128
		self.res1 = DarkNet_ResBlocks(64,1)
		self.conv3 = DarkNet_Conv2d(64,128,kernel_size=3,padding=1,stride=2)   # floor[(N-k+2p)/s]+1 = floor[(N-1)/2]+1 = 64
		self.res2 = DarkNet_ResBlocks(128,2)
		self.conv4 = DarkNet_Conv2d(128,256,kernel_size=3,padding=1,stride=2)  # floor[(N-k+2p)/s]+1 = floor[(N-1)/2]+1 = 32
		self.res3 = DarkNet_ResBlocks(256,8)   # b3 = 32 x 32 x 256 
		self.conv5 = DarkNet_Conv2d(256,512,kernel_size=3,padding=1,stride=2)  # floor[(N-k+2p)/s]+1 = floor[(N-1)/2]+1 = 16
		self.res4 = DarkNet_ResBlocks(512,8)   # b4 = 16 x 16 x 512 
		self.conv6 = DarkNet_Conv2d(512,1024,kernel_size=3,padding=1,stride=2) # floor[(N-k+2p)/s]+1 = floor[(N-1)/2]+1 = 8
		self.res5 = DarkNet_ResBlocks(1024,4)  # b5 = 8 x 8 x 1024 

	def forward(self,x):
		z = self.conv1(x)
		z = self.conv2(z)
		z = self.res1(z)
		z = self.conv3(z)
		z = self.res2(z)
		z = self.conv4(z)
		#-------------------------------------#
		b3 = self.res3(z)
		z = self.conv5(b3)
		b4 = self.res4(z)
		z = self.conv6(b4)
		b5 = self.res5(z)
		return b3,b4,b5  # output has 3 tensors , to connect FPN  
	












if __name__ == '__main__':
	

	x = torch.ones(1,512,300)
	MHSA = MultiHeadSelfAttention(h=4,n=512,d=300,k=128,o=3)
	print(MHSA(x).shape)


	x = torch.ones(100,2)
	siren = Siren()
	print(siren(x).shape)
	

	center = torch.LongTensor([0,0,0])
	nbrs = torch.LongTensor([1,2,3])
	negs = torch.LongTensor([[4,5],[6,7],[8,9]])
	skipgram = SkipGram()
	print(skipgram(center,nbrs,negs))

	darknet = DarkNet53()
	x = torch.zeros(1,3,256,256)
	b3 , b4 , b5 = darknet(x)
	print(b3.shape)
	print(b4.shape)
	print(b5.shape)
