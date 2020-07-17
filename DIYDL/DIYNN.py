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











if __name__ == '__main__':

	x = torch.ones(1,512,300)
	MHSA = MultiHeadSelfAttention(h=4,n=512,d=300,k=128,o=3)
	print(MHSA(x).shape)

	x = torch.ones(100,2)
	siren = Siren()
	print(siren(x).shape)








