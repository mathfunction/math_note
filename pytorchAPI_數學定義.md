# 微積分 + AI Deep Learning + PyTorch

#### 註記:

$$
括號符號 \bigg( scalar \bigg) , \bigg[ tensor \bigg] , || norm || \\
    如: 向量可使用  v_{I} = [v_{i}]_{i \in I} \\
    如: 矩陣可使用  a_{IJ} = [a_{ij}]_{(i,j)\in I\times J} 表示 \\
    
    單一大寫字母代表集合，如:  I,J,K,L ... \text{ 代表指標集} ,如太多維可使用 I^1,I^2,... \\
    對於取用同個指標集I子集的元素，可使用 i,i',i'' , 如太多個可使用 i^{(2)},i^{(3)}... \\
    {\color{red}{紅色}}代表為資料流,運算結果流 \\
    {\color{blue}{藍色}} 代表模型參數(parameters),訓練時會受到最佳化演算法改變\\
    {\color{green}{綠色}} 代表模型超參數(hyper-parameters),不會受訓練影響 \\
    {\color{purple}{紫色括號} [ P ] } \in \{0,1\} ， 代表 \text{ Iverson Bracket }\\
$$





## torch.nn

#### 1. Linear Layers

$$
Linear({\color{red}x_{J}};{\color{blue}w_{IJ}},{\color{blue}b_{I}}) := {\color{blue}w_{IJ}}\cdot {\color{red}x_{J}}+{\color{blue}b_{I}} = \bigg[\sum_{j \in J}{\color{blue}w_{ij}} {\color{red}x_j} + {\color{blue}b_{i}} \bigg]_{i \in I}
$$

---------------------

Gradient :
$$
\nabla_{\color{blue}\omega_{IJ}}Linear =  [{\color{red}x_j}]_{(i,j)\in I\times J} \\
	\nabla_{\color{blue}b_{I}}Linear = 1_{I}
$$

---------------------------



#### 2. Nonlinear Activations  

$$
Softmax({\color{red}x_{I}}) = \bigg[\frac{e^{\color{red}x_{i}}}{||e^{\color{red}x_{I}}||_{1}} \bigg]_{i\in I} \quad \text{其中分母 } ||e^{\color{red}x_{I}}||_1 := \sum_{i'\in I} e^{\color{red}x_{i'}}
$$

$$
Tanh({\color{red}x_{I}}) := \bigg[\frac{e^{\color{red}x_{i}}-e^{- \color{red}x_{i}}}{e^{ \color{red}x_{i}}+e^{- \color{red}x_{i}}} \bigg]_{i \in I}
$$

----------------------------------

Gradient :
$$
\nabla_{\color{red}x_{I}}Tanh({\color{red}x_{I}}) = [1 - Tanh^2({\color{red}x_i})]_{i\in I}
$$

---------------------------



$$
Sigmoid({\color{red}x_{I}}) := \bigg[ \frac{1}{1+e^{-\color{red}x_{i}}}  \bigg]_{i\in I}
$$

--------------------------------------------

Gradient :
$$
\nabla_{\color{red}x_{I}} Sigmoid({\color{red}x_{I}}) = Sigmoid({\color{red}x_{I}})\odot(1_{I}-Sigmoid({\color{red}x_{I}}))
$$

-----------------------------------------------------------


$$
LogSoftmax({\color{red}x_{I}}) := ln \circ Softmax({\color{red}x_{I}}) = \bigg[ln\left(\frac{e^{\color{red}x_{i}}}{||e^{\color{red}x_{I}}||_{1}}\right)\bigg]_{i \in I}
$$

$$
Softplus({\color{red}x_{I}};{\color{green}\beta}) = \bigg[ \frac{1}{\color{green}\beta}ln(1+e^{{\color{green}\beta} \color{red}x_{i}}) \bigg]_{i \in I} \quad \text{註: } {\color{green}\beta }  \text{ 每個維度都是同個參數}
$$

$$
Softsign({\color{red}x_{I}}) = \bigg[\frac{\color{red}x_{i}}{1+|{\color{red}x_i}|}\bigg]_{i\in I}
$$


$$
Thershold({\color{red}x_{I}},{\color{green}\alpha}) = {\bigg[ \color{red}x_{i}} \cdot{\color{purple}[} {\color{red}x_{i}}> {\color{green}\alpha}{\color{purple}]} + {\color{green}\alpha} \cdot {\color{purple}[} {\color{red}x_{i}}\leq {\color{green}\alpha} {\color{purple}]} \bigg]_{i\in I}
$$

$$
	ReLU({\color{red}x_{I}}) = \bigg[ max(0,{\color{red}x_{i}}) \bigg]_{i \in I} 

$$

$$
PReLU({\color{red}x_{I}}) = \bigg[ max(0,{\color{red}x_{i}})+ {\color{green}a}\cdot min(0,{\color{red}x_{i}}) \bigg]_{i \in I}
$$

$$
註 : 激發函數通常寫成 \sigma({\color{red}x_{I}})
$$



#### 3. Dropout Layers

$$
Dropout({\color{red}x_{I}};{\color{green}p}) = \bigg[ {0\cdot {\color{purple}[} \#_{i} \leq {\color{green}p} {\color{purple}]+\color{red}x_{i}}\cdot{\color{purple}[} \#_{i} > {  \color{green}p} {\color{purple}]}}  \bigg]_{i \in I} \quad 其中  \#_{I} \sim uniform\bigg((0,1)^{|I|}\bigg) 為隨機向量
$$



#### 4. Sparse Layers

$$
Embedding({\color{red}z_{J}};{\color{red}I},{\color{green}D}):= \bigg[ {\color{blue}w_{z_{j}d}} \bigg]_{{\color{red}J}\times {\color{green}D}} \quad 其中 {\color{red}z_{J}} \in \color{}I^{|J|} , 參數矩陣為 { \color{blue} w_{ID} := [w_{id}]_{(i,d)\in I \times D}}  \\
$$

$$
註: 在 \text{NLP(Natural Language Processing)} 領域 \\
I \text{代表詞種類(words)集合}  \\
|{\color{green}D}| 為 \text{ Word Embedding Dimension} \\
不同的詞，可用 1,2,3,4...|I| 編號 ，即  I \underset{1-1}{\leftrightarrow} \{1,2,3,...|I|\} \\
word_{i} 的詞向量(word2vec) 即為 {\color{blue} [\omega_{id}]_{d\in D}}  \\
一句有|J|個詞的句子 = z_{J}= [z_j] = [第 j 個詞]_{j=1,2,...|J|} \\
Embedding(句子) = [詞實向量] = 實矩陣 \\
核心量化概念 :  詞 \equiv 向量 , 句子 \equiv 矩陣(有順序概念) , j 代表位置 , d 代表詞特徵 , 詞特徵是演算法學來的 !!
$$

#### 5.Distance Functions

$$
CosineSimilarity(u_{I},v_{I}) = \frac{u_{I}\cdot v_{I}}{max(||u_{I}||_{2}||v_{I}||_{2},\epsilon)} =\frac{\sum_{i\in I} u_iv_i}{max\bigg(\sqrt{\sum_{i\in I}u_{i}^2 \sum_{i\in I} v_{i}^2},\epsilon\bigg)}
$$

$$
PairwiseDistance(u_{I};p) := ||u_{I}||_{p} =  \bigg( \sum_{i \in I}|x_{i}|^{p}  \bigg)^{\frac{1}{p}}
$$

$$
y^{pred}_{I}  代表經由數學模型計算後的預測向量 \\
    y^{target}_{I}  代表原始資料的目標向量(正確答案) \\
    \mathcal{|B|}  代表 \text{batchsize} \\
    y^{pred}_{ib},y^{target}_{ib}  代表樣本 b 的 y 值 \\
$$

註:  模型架構好以後 Pytorch 支援 Batch Input !!
$$
Model([x_{Ib}]_{b \in \mathcal{B}}) := \bigg[Model(x_{Ib}) \bigg]_{b \in \mathcal{B}}
$$


#### 6. Loss Functions (Sum Overall Batch Samples)

$$
L1Loss\bigg(\bigg[(y^{pred}_{I},y^{target}_{I})\bigg]_{b \in \mathcal{B}}\bigg) = \sum_{b\in \mathcal{B}} \left(\frac{1}{|I|} \sum_{i\in I}|y^{pred}_{ib}-y^{target}_{ib}| \right)
$$

$$
MSELoss\bigg(\bigg[(y^{pred}_{I},y^{target}_{I})\bigg]_{b \in \mathcal{B}}\bigg) = \sum_{b \in \mathcal{B}}\bigg(\frac{1}{|I|} \sum_{i \in I} (y^{pred}_{ib}-y^{target}_{ib})^2  \bigg)
$$

$$
CrossEntropyLoss\bigg(\bigg[(y^{pred}_{I},y^{target}_{I})\bigg]_{b \in \mathcal{B}}\bigg) = H\bigg(y^{target}_{I},Softmax(y^{pred}_{I})\bigg) = -\sum_{i \in I}y^{target}_{i}\ln\bigg( \frac{y^{pred}_{i}}{\sum_{i'\in I}y^{pred}_{i'}} \bigg) \\
註:  y^{target}_{I}  \text{ is one hot encoding } , \text{API use integer input}  
$$

$$
CRF({\color{red}s_{IY}};{\color{blue}\omega_{YY}})=-\bigg(\sum_{i=1}^{|I|}{\color{red}s_{iy_i}}+\sum_{i=1}^{|I|-1}{\color{blue}\omega_{y_i,y_{i+1}}}\bigg) \\

註: |I| 為句子長度，|Y|為 \text{ Label } 種類集 , s_{i,y_i} 又稱為 \text{ emission score}, y_{I} 為 \text{ target label vector} \\
\color{purple} 註2: \text{Bi-LSTM} 輸出為 x_{ID} 向量，|D| 為 \text{ hidden dimension} ，需要再作線性轉換 S_{IY} ,Linear(x_{ID}) = s_{IY} 
$$





#### 7.Recurrent Layers (Share Weight Matrix)

$$
{\color{red} h_{D} }為接口(\text{hidden dimension}) , h^{0}_{D} 可 \text{fixed} 或可加入一起學習
$$




$$
RNNCell({\color{red}x_{I},h_{D}};{\color{blue}\omega_{DI},\omega_{DD} ,b_{D}}) = \tanh({\color{blue}w_{DI}}{\color{red}x_{I}}+{\color{blue}\omega_{DD}}{\color{red}h_{D}}+{\color{blue}b_{D}})
$$

$$
LSTMCell({\color{red}x_{I},c_{D},h_{D}};{\overbrace{\color{blue}\omega^{x\rightarrow i}_{DI},\omega^{h \rightarrow i}_{DD},\omega^{x\rightarrow f}_{DI},\omega^{h \rightarrow f }_{DD},\omega^{x \rightarrow g}_{DI},\omega^{h \rightarrow g}_{DD},\omega^{x\rightarrow o}_{DI},\omega^{h \rightarrow o}_{DD}}^{8\text{ weight matrixs}},\underbrace{{\color{blue}b^{i}_{D},b^{f}_{D},b^{g}_{D},b^{o}_{D}}}_{4 \text{ bias vectors}}})
\\
$$

-------------------------------

結構細節:
$$
f_{D}  := \sigma({\color{blue}\omega^{x\rightarrow f}_{DI}}{\color{red}x_{I}}+{\color{blue}\omega^{h\rightarrow f}_{DD}}{\color{red}h_{D}}+{\color{blue}b^{f}_{D}}) \\

i_{D}:= \sigma({\color{blue}\omega^{x\rightarrow i}_{DI}}{\color{red}x_{I}}+{\color{blue}\omega^{h \rightarrow i}_{DD}}{\color{red}h_{D}}+{\color{blue}b^{i}_{D}}) \\
o_{D} := \sigma({\color{blue}\omega_{DI}^{x\rightarrow o}}{\color{red}x_{I}}+{\color{blue}\omega^{h \rightarrow o}_{DD}}{\color{red} h_{D}}+{\color{blue}b^{o}_{D}}) \\
g_{D} := tanh({\color{blue}\omega^{x\rightarrow g}_{DI}} {\color{red}x_{I}}+{\color{blue}\omega^{h \rightarrow g}_{DD}} {\color{red}h_{D}}+{\color{blue}b^{g}_{D}}) \\
LSTMCell({\color{red}x_{I},c_{D},h_{D}};{\color{blue}...}) = o_{D}  \odot tanh(f_{D}\odot {\color{red}c_{D}} + i_{D} \odot g_{D})
$$

------------------------


$$
GRUCell({\color{red}x_{I}},{\color{red}h_{D}};{\color{blue}\omega^{x\rightarrow r}_{DI},\omega^{h\rightarrow r}_{DD},\omega^{x \rightarrow n}_{DD},\omega^{x\rightarrow z}_{DI},
\omega^{h \rightarrow z}_{DD},\omega^{h \rightarrow n}_{DD},b^{r}_{D},b^{n}_{D},b^{z}_{D}})
$$

-------------------------------------------------------

結構細節:
$$
r_{D} = \sigma({\color{blue}\omega^{x\rightarrow r}_{DI}}{\color{red}x_{I}}+{\color{blue}\omega^{h \rightarrow r}_{DD}}{\color{red}h_{D}}+{\color{blue}b^{r}_{D}})\\
z_{D} = \sigma({\color{blue}\omega^{x\rightarrow z}_{DI}}{\color{red}x_{I}}+{\color{blue}\omega^{h \rightarrow z}_{DD}}{\color{red}h_{D}}+{\color{blue}b^{z}_{D}})\\
n_{D} = tanh({\color{blue}\omega^{x\rightarrow n}_{DD}}{\color{red}x_{I}}+r_{D} \odot({\color{blue}\omega^{h \rightarrow n}_{DD}}{\color{red}h_{D}})+ {\color{blue}b^{n}_{D}}) \\
GRUCell({\color{red}x_{I},h_{D}};...)= (1_{D}-z_{D})\odot {\color{red}h_{D}} + z_{D} \odot n_{D}
$$

-----------















##  Pseudocode of General BackPropagation Algorithm 

```c++










```



























