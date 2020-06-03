## 賽局理論數學小記

## regret matching :

- normal form game (just one action and get reward immediately)

$$
\sigma : \bigcup_{i\in \mathcal{P}}\mathcal{A_i} \longrightarrow [0,1] \in \Sigma 

,\quad u^{\text{pure}}: \bigcup_{i\in \mathcal{P}}\mathcal{A_i} \longrightarrow \mathbb{R}^{|\mathcal{P}|} , u^{\text{mixed}}: \Sigma \rightarrow \mathbb{R}^{|\mathcal{P}|}\\ 
{\color{red}u^{\text{pure}}_i(a_{i},a_{-i})} \equiv\text{utility of play $i$ by choosing action $a_{i}$, others $a_{-i}$} \\

 
{\color{blue}u_i^{\text{mixed}}(\sigma_i,\sigma_{-i})} := \sum_{a\in \mathcal{A_i}} \sum_{a' \in \mathcal{A_{-i}}} \underbrace{\sigma_{i}(a_{i}) \sigma_{-i}(a_{-i})}_{\text{probabilities}} {\color{red}u_i^{\text{pure}}(a_{i},a_{-i})} \equiv \text{expected utility !!}
$$


$$
\text{Best Response Function :  } \\\sigma^{*}_{i} :=\mathcal{BR}_i(\sigma_{-i}) = \underset{\sigma_{i} \in \Sigma_{i}}  {\text{argmax }} {\color{blue}u^{\text{mixed}}_i(\sigma_{i},\sigma_{-i})}\\
\text{Nash Equilibrium : } \\
\sigma^{*} \equiv \text{best response for each player !!}
$$



$$
\begin{array}{l}
\sigma \text{ is } |\mathcal{P}|\times|\mathcal{A}| \text{ matrix}\\
\sigma_{i} \text{ is column $|\mathcal{A}|$ vector called stratgies } 
\end{array}
$$

$$
\mu^{regret}_i :  \mathcal{A}_i\longrightarrow \mathbb{R}\\
\Delta_i^{regret}(a'_i) := \bigg[u^{\text{pure}}(a'_{i},a_{-i}) - u^{\text{pure}}(a)\bigg] \\

\text{regret-matching : select } a_i  \text{ in probability } \sigma_{i}(a_i)= \frac{\mu^{regret}_{i}(a_i)}{\sum_{a_j \in \mathcal{A}_i}\mu^{regret}_i(a_j)}
$$




