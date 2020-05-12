=====================================================================
$$
Y(N,C_2,H_2,W_2) := Conv2d\bigg[X(N,C_1,H_1,W_1)\bigg] \\= \bigwedge_{c_2 \in C_2} \bigg[ \sum_{c_1 \in C_1} \overbrace{\color{blue}\omega(c_2,c_1,K, K)}^{\text{weights }} * \overbrace{\color{red}X(N,c_1,H_1,W_1)}^{data / features} + \overbrace{\color{green}b(N,c_2,H_2,W_2)}^{bias}\bigg]\\ = \bigwedge_{c_2 \in C_2}\bigg[ \sum_{c_1 \in C_1} {\color{blue}\omega_{c_2 c_1}}* {\color{red}X_{c_1}} + {\color{green}b_{c_2}} \bigg]
$$

$$
H_2 = \bigg\lfloor \frac{H_1+2p-d(K-1)-1}{s} +1 \bigg\rfloor\\
W_2 = \bigg\lfloor \frac{W_2+2p-d(K-1)-1}{s}+1\bigg\rfloor \\
$$

$$
\text{bias shape is } (N,c_2,H_2,W_2)  \text{ but only has } (c_2 \times H_2 \times W_2) \text{ parameters } , \text{ identical on batch_dimension}
$$



- N = batch size  (GPU parallel computation)
- C = channels / num of filters / RGB(3)
- H = height (row indexs)
- W = width (column indexs) 
- p = padding = 0  
- d = dilation  = 1
- s = stride = 1

=====================================================================
$$
{\color{blue}\omega(c_2,c_1,K, K)}*{\color{red}X(N,c_1,H_1,W_1)} = \bigwedge_{i_1\in \hat{H_1}} \bigwedge_{j_1 \in \hat{W_1}} \bigg[ \sum_{i'\in [i_1\pm\frac{K-1}{2}]} \sum_{j'\in [j_1 \pm \frac{K-1}{2}]} \omega(c_2,c_1,i',j')\cdot X(N,c_1,i',j') \bigg]
$$



so we can have 
$$
Y(N,c_2,i_2,j_2) =\overset{filters 個數}{\bigwedge_{c_2\in C_2}}\bigwedge_{i_2\in H_2} \bigwedge_{j_2 \in W_2} \bigg[ \overset{\text{input-channels 加總}}{\sum_{c_1\in C_1}}\sum_{i'\in [i_1(i_2)\pm\frac{K-1}{2}]} \sum_{j'\in [j_1(j_2) \pm \frac{K-1}{2}]} \omega(c_2,c_1,i',j')\cdot X(N,c_1,i',j') \bigg]
$$

$$
其中 : i_1(i_2) = i_2 + \frac{K-1}{2} , j_1(j_2) = j_2 + \frac{K-1}{2}  \\
$$

finally :
$$
Y(N,C_2,H_2,W_2) \\=\bigwedge_{c_2\in C_2}\bigwedge_{i_2\in H_2} \bigwedge_{j_2 \in W_2} \bigg[\sum_{c_1\in C_1}\sum_{k\in [0,K-1]} \sum_{k\in [0,K-1]} \omega(c_2,c_1,i_2+k,j_2+k)\cdot X(N,c_1,i_2+k,j_2+k) \bigg] 
$$


