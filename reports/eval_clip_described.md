# Eval report: clip-zero-shot (openai/clip-vit-base-patch32) prompts=described

Held-out eval_set: 210 tiles. **Accuracy: 61.43%**

Macro-F1: 0.5997 | Weighted-F1: 0.5997 | Micro-F1: 0.6143 (= accuracy for single-label classification)

## Per-class
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| AnnualCrop | 0.5556 | 0.8333 | 0.6667 | 30 |
| Forest | 0.8333 | 0.5 | 0.625 | 30 |
| Highway | 0.4091 | 0.6 | 0.4865 | 30 |
| Industrial | 0.6667 | 0.7333 | 0.6984 | 30 |
| Residential | 0.7568 | 0.9333 | 0.8358 | 30 |
| River | 0.5385 | 0.2333 | 0.3256 | 30 |
| SeaLake | 0.7 | 0.4667 | 0.56 | 30 |

## Confusion matrix (rows = true, cols = predicted)
| true \ pred | AnnualCrop | Forest | Highway | Industrial | Residential | River | SeaLake |
|---|---|---|---|---|---|---|---|
| AnnualCrop | 25 | 1 | 2 | 1 | 0 | 0 | 1 |
| Forest | 0 | 15 | 11 | 0 | 0 | 0 | 4 |
| Highway | 7 | 0 | 18 | 4 | 1 | 0 | 0 |
| Industrial | 0 | 0 | 0 | 22 | 8 | 0 | 0 |
| Residential | 1 | 0 | 0 | 1 | 28 | 0 | 0 |
| River | 11 | 0 | 7 | 4 | 0 | 7 | 1 |
| SeaLake | 1 | 2 | 6 | 1 | 0 | 6 | 14 |

## Accuracy vs coverage
| threshold | coverage | accuracy on accepted | sent to review |
|---|---|---|---|
| 0.0 | 100.00% | 0.6143 | 0 |
| 0.5 | 57.62% | 0.7438 | 89 |
| 0.6 | 42.86% | 0.8222 | 120 |
| 0.7 | 34.29% | 0.8611 | 138 |
| 0.8 | 26.19% | 0.8909 | 155 |
| 0.9 | 14.29% | 0.9667 | 180 |
| 0.95 | 6.19% | 0.9231 | 197 |
| 0.99 | 0.00% | None | 210 |

## Errors (81), most confident first
| file | true | predicted | conf |
|---|---|---|---|
| tile_054.png | River | AnnualCrop | 0.9713 |
| tile_174.png | SeaLake | AnnualCrop | 0.8835 |
| tile_045.png | River | AnnualCrop | 0.8756 |
| tile_091.png | Industrial | Residential | 0.8634 |
| tile_050.png | River | AnnualCrop | 0.8629 |
| tile_101.png | Industrial | Residential | 0.8286 |
| tile_057.png | River | AnnualCrop | 0.7632 |
| tile_034.png | River | Highway | 0.7448 |
| tile_102.png | Industrial | Residential | 0.7293 |
| tile_097.png | Industrial | Residential | 0.7091 |
| tile_042.png | River | Highway | 0.6672 |
| tile_104.png | Industrial | Residential | 0.6569 |
| tile_106.png | Industrial | Residential | 0.6566 |
| tile_043.png | River | Industrial | 0.6199 |
| tile_040.png | River | AnnualCrop | 0.6149 |
| tile_077.png | Residential | Industrial | 0.6127 |
| tile_189.png | Highway | AnnualCrop | 0.5995 |
| tile_098.png | Industrial | Residential | 0.5871 |
| tile_208.png | Highway | Residential | 0.583 |
| tile_200.png | Highway | Industrial | 0.5787 |
| tile_119.png | Industrial | Residential | 0.5713 |
| tile_044.png | River | AnnualCrop | 0.5593 |
| tile_048.png | River | AnnualCrop | 0.5394 |
| tile_130.png | AnnualCrop | Forest | 0.5325 |
| tile_201.png | Highway | AnnualCrop | 0.532 |
| tile_049.png | River | Industrial | 0.5317 |
| tile_007.png | Forest | Highway | 0.5314 |
| tile_184.png | Highway | AnnualCrop | 0.5222 |
| tile_183.png | Highway | AnnualCrop | 0.506 |
| tile_181.png | Highway | AnnualCrop | 0.5027 |
| tile_019.png | Forest | Highway | 0.5009 |
| tile_058.png | River | Industrial | 0.4987 |
| tile_026.png | Forest | SeaLake | 0.4945 |
| tile_047.png | River | Industrial | 0.4908 |
| tile_041.png | River | Highway | 0.4751 |
| tile_033.png | River | AnnualCrop | 0.4699 |
| tile_139.png | AnnualCrop | Highway | 0.4694 |
| tile_052.png | River | Highway | 0.468 |
| tile_032.png | River | Highway | 0.4537 |
| tile_031.png | River | SeaLake | 0.4504 |
| tile_059.png | River | Highway | 0.4409 |
| tile_046.png | River | Highway | 0.4342 |
| tile_036.png | River | AnnualCrop | 0.4282 |
| tile_186.png | Highway | Industrial | 0.4225 |
| tile_164.png | SeaLake | Industrial | 0.4201 |
| tile_005.png | Forest | Highway | 0.4191 |
| tile_015.png | Forest | SeaLake | 0.4072 |
| tile_069.png | Residential | AnnualCrop | 0.4044 |
| tile_198.png | Highway | AnnualCrop | 0.4027 |
| tile_035.png | River | AnnualCrop | 0.4019 |
| tile_004.png | Forest | Highway | 0.3907 |
| tile_008.png | Forest | Highway | 0.3814 |
| tile_038.png | River | AnnualCrop | 0.3734 |
| tile_009.png | Forest | Highway | 0.3628 |
| tile_192.png | Highway | Industrial | 0.3628 |
| tile_024.png | Forest | SeaLake | 0.3592 |
| tile_021.png | Forest | Highway | 0.3542 |
| tile_011.png | Forest | Highway | 0.3485 |
| tile_187.png | Highway | Industrial | 0.3443 |
| tile_194.png | Highway | AnnualCrop | 0.3353 |
| tile_171.png | SeaLake | Forest | 0.3142 |
| tile_123.png | AnnualCrop | Industrial | 0.3069 |
| tile_125.png | AnnualCrop | Highway | 0.3061 |
| tile_022.png | Forest | Highway | 0.3029 |
| tile_018.png | Forest | Highway | 0.2833 |
| tile_166.png | SeaLake | Highway | 0.2828 |
| tile_016.png | Forest | SeaLake | 0.27 |
| tile_160.png | SeaLake | Highway | 0.265 |
| tile_131.png | AnnualCrop | SeaLake | 0.2594 |
| tile_180.png | SeaLake | River | 0.2592 |
| tile_156.png | SeaLake | Highway | 0.2582 |
| tile_028.png | Forest | Highway | 0.2517 |
| tile_178.png | SeaLake | Highway | 0.2476 |
| tile_158.png | SeaLake | Forest | 0.2463 |
| tile_151.png | SeaLake | Highway | 0.2427 |
| tile_153.png | SeaLake | River | 0.233 |
| tile_170.png | SeaLake | River | 0.2293 |
| tile_169.png | SeaLake | River | 0.2291 |
| tile_177.png | SeaLake | River | 0.2203 |
| tile_172.png | SeaLake | Highway | 0.2145 |
| tile_157.png | SeaLake | River | 0.2094 |
