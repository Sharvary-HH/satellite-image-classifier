# Eval report: clip-zero-shot (openai/clip-vit-base-patch32) prompts=raw_names

Held-out eval_set: 210 tiles. **Accuracy: 54.76%**

Macro-F1: 0.4993 | Weighted-F1: 0.4993 | Micro-F1: 0.5476 (= accuracy for single-label classification)

## Per-class
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| AnnualCrop | 0.4444 | 0.9333 | 0.6022 | 30 |
| Forest | 1.0 | 0.4 | 0.5714 | 30 |
| Highway | 0.8 | 0.4 | 0.5333 | 30 |
| Industrial | 0.5 | 0.0333 | 0.0625 | 30 |
| Residential | 0.4821 | 0.9 | 0.6279 | 30 |
| River | 0.7692 | 0.3333 | 0.4651 | 30 |
| SeaLake | 0.5102 | 0.8333 | 0.6329 | 30 |

## Confusion matrix (rows = true, cols = predicted)
| true \ pred | AnnualCrop | Forest | Highway | Industrial | Residential | River | SeaLake |
|---|---|---|---|---|---|---|---|
| AnnualCrop | 28 | 0 | 0 | 0 | 1 | 0 | 1 |
| Forest | 5 | 12 | 2 | 0 | 0 | 0 | 11 |
| Highway | 11 | 0 | 12 | 1 | 1 | 1 | 4 |
| Industrial | 1 | 0 | 0 | 1 | 26 | 1 | 1 |
| Residential | 3 | 0 | 0 | 0 | 27 | 0 | 0 |
| River | 11 | 0 | 1 | 0 | 1 | 10 | 7 |
| SeaLake | 4 | 0 | 0 | 0 | 0 | 1 | 25 |

## Accuracy vs coverage
| threshold | coverage | accuracy on accepted | sent to review |
|---|---|---|---|
| 0.0 | 100.00% | 0.5476 | 0 |
| 0.5 | 50.95% | 0.6636 | 103 |
| 0.6 | 38.57% | 0.6543 | 129 |
| 0.7 | 26.19% | 0.7091 | 155 |
| 0.8 | 15.71% | 0.697 | 177 |
| 0.9 | 6.67% | 0.7857 | 196 |
| 0.95 | 2.86% | 1.0 | 204 |
| 0.99 | 0.00% | None | 210 |

## Errors (95), most confident first
| file | true | predicted | conf |
|---|---|---|---|
| tile_054.png | River | AnnualCrop | 0.9464 |
| tile_119.png | Industrial | Residential | 0.9315 |
| tile_101.png | Industrial | Residential | 0.9238 |
| tile_045.png | River | AnnualCrop | 0.8905 |
| tile_106.png | Industrial | Residential | 0.8483 |
| tile_115.png | Industrial | Residential | 0.8276 |
| tile_098.png | Industrial | Residential | 0.827 |
| tile_102.png | Industrial | Residential | 0.8232 |
| tile_097.png | Industrial | Residential | 0.8231 |
| tile_091.png | Industrial | Residential | 0.8002 |
| tile_184.png | Highway | AnnualCrop | 0.7927 |
| tile_104.png | Industrial | Residential | 0.7781 |
| tile_050.png | River | AnnualCrop | 0.754 |
| tile_198.png | Highway | AnnualCrop | 0.7153 |
| tile_113.png | Industrial | Residential | 0.7005 |
| tile_100.png | Industrial | Residential | 0.7 |
| tile_116.png | Industrial | Residential | 0.6948 |
| tile_040.png | River | AnnualCrop | 0.693 |
| tile_095.png | Industrial | Residential | 0.6909 |
| tile_174.png | SeaLake | AnnualCrop | 0.6761 |
| tile_201.png | Highway | AnnualCrop | 0.6667 |
| tile_131.png | AnnualCrop | SeaLake | 0.6608 |
| tile_208.png | Highway | Residential | 0.648 |
| tile_114.png | Industrial | Residential | 0.6461 |
| tile_105.png | Industrial | Residential | 0.622 |
| tile_026.png | Forest | SeaLake | 0.6055 |
| tile_108.png | Industrial | Residential | 0.6046 |
| tile_057.png | River | AnnualCrop | 0.603 |
| tile_189.png | Highway | AnnualCrop | 0.599 |
| tile_110.png | Industrial | Residential | 0.5985 |
| tile_031.png | River | SeaLake | 0.5965 |
| tile_044.png | River | AnnualCrop | 0.5888 |
| tile_036.png | River | AnnualCrop | 0.5867 |
| tile_048.png | River | SeaLake | 0.5742 |
| tile_181.png | Highway | AnnualCrop | 0.5143 |
| tile_084.png | Residential | AnnualCrop | 0.51 |
| tile_191.png | Highway | SeaLake | 0.4863 |
| tile_096.png | Industrial | Residential | 0.4827 |
| tile_069.png | Residential | AnnualCrop | 0.4815 |
| tile_092.png | Industrial | Residential | 0.4784 |
| tile_183.png | Highway | AnnualCrop | 0.473 |
| tile_032.png | River | AnnualCrop | 0.4676 |
| tile_209.png | Highway | River | 0.4659 |
| tile_015.png | Forest | SeaLake | 0.4638 |
| tile_027.png | Forest | AnnualCrop | 0.4606 |
| tile_118.png | Industrial | Residential | 0.4524 |
| tile_028.png | Forest | SeaLake | 0.4434 |
| tile_009.png | Forest | SeaLake | 0.4336 |
| tile_055.png | River | SeaLake | 0.4226 |
| tile_033.png | River | AnnualCrop | 0.4206 |
| tile_047.png | River | Residential | 0.419 |
| tile_030.png | Forest | SeaLake | 0.4158 |
| tile_202.png | Highway | Industrial | 0.4022 |
| tile_052.png | River | SeaLake | 0.39 |
| tile_073.png | Residential | AnnualCrop | 0.3896 |
| tile_125.png | AnnualCrop | Residential | 0.3874 |
| tile_152.png | SeaLake | River | 0.379 |
| tile_194.png | Highway | AnnualCrop | 0.3735 |
| tile_023.png | Forest | AnnualCrop | 0.3721 |
| tile_103.png | Industrial | Residential | 0.3617 |
| tile_112.png | Industrial | Residential | 0.3592 |
| tile_187.png | Highway | SeaLake | 0.3573 |
| tile_035.png | River | AnnualCrop | 0.3545 |
| tile_117.png | Industrial | Residential | 0.354 |
| tile_120.png | Industrial | SeaLake | 0.3421 |
| tile_186.png | Highway | AnnualCrop | 0.3381 |
| tile_034.png | River | Highway | 0.3378 |
| tile_107.png | Industrial | Residential | 0.3322 |
| tile_093.png | Industrial | Residential | 0.3316 |
| tile_059.png | River | SeaLake | 0.3297 |
| tile_178.png | SeaLake | AnnualCrop | 0.3238 |
| tile_042.png | River | SeaLake | 0.3178 |
| tile_099.png | Industrial | River | 0.3165 |
| tile_043.png | River | AnnualCrop | 0.3164 |
| tile_016.png | Forest | SeaLake | 0.3151 |
| tile_094.png | Industrial | Residential | 0.3071 |
| tile_046.png | River | SeaLake | 0.303 |
| tile_196.png | Highway | AnnualCrop | 0.3024 |
| tile_206.png | Highway | AnnualCrop | 0.3008 |
| tile_171.png | SeaLake | AnnualCrop | 0.2953 |
| tile_005.png | Forest | SeaLake | 0.2884 |
| tile_007.png | Forest | Highway | 0.2878 |
| tile_004.png | Forest | SeaLake | 0.287 |
| tile_018.png | Forest | AnnualCrop | 0.2837 |
| tile_109.png | Industrial | AnnualCrop | 0.2833 |
| tile_024.png | Forest | SeaLake | 0.2792 |
| tile_011.png | Forest | SeaLake | 0.2771 |
| tile_200.png | Highway | SeaLake | 0.2727 |
| tile_019.png | Forest | Highway | 0.2614 |
| tile_193.png | Highway | SeaLake | 0.2492 |
| tile_192.png | Highway | AnnualCrop | 0.2436 |
| tile_022.png | Forest | AnnualCrop | 0.2417 |
| tile_029.png | Forest | SeaLake | 0.2392 |
| tile_008.png | Forest | AnnualCrop | 0.2372 |
| tile_158.png | SeaLake | AnnualCrop | 0.2297 |
