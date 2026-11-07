# BigGAN vs diffusion (sd-turbo), CLIP score against the free-text prompt

sd-turbo backend: local · BigGAN truncation 0.4, best of 4 by CLIP

| Prompt | BigGAN class | BigGAN CLIP | sd-turbo CLIP | BigGAN time | sd-turbo time |
|---|---|---|---|---|---|
| a friendly family dog with golden fur | golden retriever | 29.7 | 30.1 | 8.6 s* | 35.2 s |
| a mountain erupting with lava | volcano | 28.5 | 30.7 | 1.8 s | 35.5 s |
| a slice of italian food with cheese and tomato | pizza | 28.5 | 30.6 | 1.8 s | 37.9 s |
| an orange and black butterfly that migrates | monarch | 31.6 | 33.4 | 2.0 s | 36.1 s |
| a cat riding a bicycle | bicycle-built-for-two | 24.3 | 36.1 | 1.9 s | 35.1 s |
| **Mean** | | **28.5** | **32.2** | | |

BigGAN can only draw the 1000 ImageNet classes at 128 px, so attributes and compositions in the text ("riding a bicycle", "orange and black") are lost. Diffusion follows the full text but costs more compute.

\* first call includes one-off CLIP model loading; warm BigGAN generation + CLIP ranking of 4 images takes about 2 s on CPU vs about 36 s for one 512 px sd-turbo image.
