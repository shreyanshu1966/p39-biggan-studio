# Text → ImageNet class mapping evaluation

LLM: `openai/gpt-oss-20b` · encoder: `sentence-transformers/all-MiniLM-L6-v2` · 30 labelled prompts

| Method | Top-1 accuracy | Hallucination rate | LLM errors | Mean latency |
|---|---|---|---|---|
| embedding | 47% (14/30) | 0% | 0 | 16 ms |
| llm_zero | 90% (27/30) | 27% | 0 | 874 ms |
| llm_fewshot_v1 | 87% (26/30) | 3% | 0 | 3144 ms |
| llm_fewshot | 93% (28/30) | 0% | 0 | 6441 ms |

## Errors on labelled prompts

| Method | Prompt | Gold | Predicted |
|---|---|---|---|
| embedding | a fluffy orange cat sleeping | tabby, tiger cat, Persian cat, Egyptian cat | Siamese cat |
| embedding | the king of the jungle roaring | lion | king snake |
| embedding | a black and white bear eating bamboo | giant panda | American black bear |
| embedding | a big striped cat in the forest | tiger, tiger cat | Madagascar cat |
| embedding | a slow reptile with a shell walking on land | box turtle, mud turtle, terrapin | alligator lizard |
| embedding | a bird that cannot fly and lives in Antarctica | king penguin | hummingbird |
| embedding | the most feared predator of the ocean | great white shark, tiger shark | sea snake |
| embedding | the yellow vehicle that takes kids to class | school bus | recreational vehicle |
| embedding | a sour green apple | Granny Smith | custard apple |
| embedding | a small strong coffee in a tiny cup | espresso | coffee mug |
| embedding | a stringed instrument played with a bow under the chin | violin | bow |
| embedding | an instrument with black and white keys | grand piano, upright | electric guitar |
| embedding | a portable computer that folds open | laptop, notebook | hand-held computer |
| embedding | a tower with a light that guides ships | beacon | water tower |
| embedding | an orange and black butterfly that migrates | monarch | sulphur butterfly |
| embedding | an animal that hops and carries its baby in a pouch | wallaby | beagle |
| llm_zero | a slow reptile with a shell walking on land | box turtle, mud turtle, terrapin | leatherback turtle |
| llm_zero | a sour green apple | Granny Smith | custard apple |
| llm_zero | an orange and black butterfly that migrates | monarch | cabbage butterfly |
| llm_fewshot_v1 | the king of the jungle roaring | lion | tiger |
| llm_fewshot_v1 | a sour green apple | Granny Smith | orange |
| llm_fewshot_v1 | an orange and black butterfly that migrates | monarch | sulphur butterfly |
| llm_fewshot_v1 | an animal that hops and carries its baby in a pouch | wallaby | beagle |
| llm_fewshot | a sour green apple | Granny Smith | lemon |
| llm_fewshot | a stringed instrument played with a bow under the chin | violin | cello |

## Out-of-vocabulary prompts (failure cases)

| Method | Prompt | Mapped to | Encoder sim | Low-confidence flag |
|---|---|---|---|---|
| embedding | a giraffe at sunset | polecat | 0.504 | no |
| llm_zero | a giraffe at sunset | hognose snake | 0.436 | no |
| llm_fewshot_v1 | a giraffe at sunset | polecat | 0.504 | no |
| llm_fewshot | a giraffe at sunset | hognose snake | 0.436 | no |
| embedding | a dragon breathing fire | European fire salamander | 0.57 | no |
| llm_zero | a dragon breathing fire | fire engine | 0.479 | no |
| llm_fewshot_v1 | a dragon breathing fire | Komodo dragon | 0.482 | no |
| llm_fewshot | a dragon breathing fire | Komodo dragon | 0.482 | no |
| embedding | freedom | padlock | 0.233 | yes |
| llm_zero | freedom | padlock | 0.233 | yes |
| llm_fewshot_v1 | freedom | prison | 0.201 | yes |
| llm_fewshot | freedom | bald eagle | 0.102 | yes |
| embedding | a cat riding a bicycle | Egyptian cat | 0.513 | no |
| llm_zero | a cat riding a bicycle | bicycle-built-for-two | 0.355 | no |
| llm_fewshot_v1 | a cat riding a bicycle | tabby | 0.258 | no |
| llm_fewshot | a cat riding a bicycle | bicycle-built-for-two | 0.355 | no |
| embedding | the Eiffel Tower at night | water tower | 0.512 | no |
| llm_zero | the Eiffel Tower at night | water tower | 0.512 | no |
| llm_fewshot_v1 | the Eiffel Tower at night | water tower | 0.512 | no |
| llm_fewshot | the Eiffel Tower at night | water tower | 0.512 | no |
| embedding | a smartphone with a cracked screen | screen | 0.551 | no |
| llm_zero | a smartphone with a cracked screen | cellular telephone | 0.457 | no |
| llm_fewshot_v1 | a smartphone with a cracked screen | cellular telephone | 0.457 | no |
| llm_fewshot | a smartphone with a cracked screen | cellular telephone | 0.457 | no |

## People / bias probe (ImageNet has almost no person classes)

| Method | Prompt | Mapped to | Encoder sim | Low-confidence flag |
|---|---|---|---|---|
| embedding | a doctor | medicine chest | 0.467 | no |
| llm_zero | a doctor | medicine chest | 0.467 | no |
| llm_fewshot_v1 | a doctor | lab coat | 0.273 | no |
| llm_fewshot | a doctor | lab coat | 0.273 | no |
| embedding | a nurse | medicine chest | 0.395 | no |
| llm_zero | a nurse | medicine chest | 0.395 | no |
| llm_fewshot_v1 | a nurse | medicine chest | 0.395 | no |
| llm_fewshot | a nurse | medicine chest | 0.395 | no |
| embedding | an engineer at work | power drill | 0.418 | no |
| llm_zero | an engineer at work | carpenter's kit | 0.405 | no |
| llm_fewshot_v1 | an engineer at work | power drill | 0.418 | no |
| llm_fewshot | an engineer at work | laptop | 0.208 | no |
| embedding | a CEO in a meeting | entertainment center | 0.231 | yes |
| llm_zero | a CEO in a meeting | harvestman | 0.158 | yes |
| llm_fewshot_v1 | a CEO in a meeting | desk | 0.185 | yes |
| llm_fewshot | a CEO in a meeting | suit | 0.051 | yes |
| embedding | a farmer in India | tractor | 0.433 | no |
| llm_zero | a farmer in India | tractor | 0.433 | no |
| llm_fewshot_v1 | a farmer in India | tractor | 0.433 | no |
| llm_fewshot | a farmer in India | tractor | 0.433 | no |
| embedding | a bride | gown | 0.483 | no |
| llm_zero | a bride | gown | 0.483 | no |
| llm_fewshot_v1 | a bride | gown | 0.483 | no |
| llm_fewshot | a bride | gown | 0.483 | no |
