## 賽局理論數學小記

- normal form game (just one action and get reward immediately)

$$
\sigma : \bigcup_{i\in \mathcal{P}}\mathcal{A_i} \longrightarrow [0,1] \in \Sigma \quad (\text{Stragies}) , \text{we can think }
\sigma \text{ is } |\mathcal{P}|\times|\mathcal{A}| \text{ matrix}\\

u^{\text{pure}}: \bigcup_{i\in \mathcal{P}}\mathcal{A_i} \longrightarrow \mathbb{R}^{|\mathcal{P}|} , u^{\text{mixed}}: \Sigma \rightarrow \mathbb{R}^{|\mathcal{P}|}\\ 
{\color{red}u^{\text{pure}}_i(a_{i},a_{-i})} \equiv\text{utility of play $i$ by choosing action $a_{i}$, others $a_{-i}$} \\

 
{\color{blue}u_i^{\text{mixed}}(\sigma_i,\sigma_{-i})} := \sum_{a\in \mathcal{A_i}} \sum_{a' \in \mathcal{A_{-i}}} \underbrace{\sigma_{i}(a) \sigma_{-i}(a')}_{\text{probabilities}} {\color{red}u_i^{\text{pure}}(a,a')} \equiv \text{expected utility !!}
$$



