# Truncation trade-off (8 samples per cell)

| Prompt | Class | CLIP @ 0.2 | CLIP @ 0.5 | CLIP @ 1.0 | Div @ 0.2 | Div @ 0.5 | Div @ 1.0 |
|---|---|---|---|---|---|---|---|
| a friendly family dog with golden fur | golden retriever | 29.6 | 29.4 | 29.1 | 0.025 | 0.049 | 0.084 |
| a mountain erupting with lava | volcano | 26.8 | 26.0 | 26.0 | 0.021 | 0.063 | 0.145 |
| a slice of italian food with cheese and tomato | pizza | 27.7 | 27.7 | 27.7 | 0.029 | 0.043 | 0.093 |
| an old wooden ship with pirate flags | schooner | 26.5 | 26.8 | 26.4 | 0.028 | 0.045 | 0.108 |
| an orange and black butterfly that migrates | monarch | 31.2 | 30.6 | 30.6 | 0.013 | 0.026 | 0.076 |
| **Mean** | | **28.4** | **28.1** | **28.0** | **0.023** | **0.045** | **0.101** |

Diversity = mean pairwise (1 − cosine) of CLIP image embeddings. Higher truncation → more diverse samples.
