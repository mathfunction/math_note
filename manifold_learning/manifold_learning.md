# Manifold Learning 筆記

講義連結 : https://ppt.cc/fPQvFx



- 找 eigenfunction
- universal approximation thm.
- manifold distribution , cluster distribution
- 相似的東西是同一個  cluster 

$$
\mathbb{R}^{28\times28} \longrightarrow  \mathbb{R}^2
$$

- 找 local chart 
- VAE (variational auto-encoder)
- 距離平方最小
- PCA , MDS = SVD 分解
- Diffusion Map
- Graph: Vertices + Edges affintety w(E) 
- eigen function 夠多，就可以把它分開







## PCA 

$$
\begin{array}{l}

n \text{ 樣本數} , p \text{高維度} , k \text{ 低維度} \\
\text{Given} \{ X_1,X_2 ,...X_n \} \subset \mathbb{R}^{p} \\
\text{Find a affine $k$ - dimensional space $P_k \quad k << p $ }\\ 
\text{To minimize } \sum_{i=1}^{n} dist^{2}\left( X_i,P_k \right) \\
\text{Let } X = [X_1,X_2,...X_n] \in \mathbb{R}^{p\times n} \quad X_i \in \mathbb{R}^{p} \\
\color{red}\mu + \sum_{i=1}^{k}\beta_i\color{red}{u_i} \quad 
X_i \underset{\text{projection to $\mu + S_k$}}{\approx} \mu + \color{red}{\mathcal{U}} \beta_i \\
\color{red}{\mathcal{U}} \text{ orthonormal } \\ 

(1) \text{ Compute the matrix } YY^{T}    \\
(2) \text{ Diagonalize } \\
(3) \text{ largest $k$ orthonormal eigenvectors }
\end{array}
$$



## MDS

$$
H \geq 0 , q^{t}Hq \geq 0\\
H = I - \frac{1}{n}11^{T}\\
(XH)^{T}(XH) = -\frac{1}{2} HDH \\

\text{Modified MDS algorithm} \\

算出 最大的 \text{eigenvalue} \\

找出前三大的正特徵值 
$$

