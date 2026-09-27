# Eval report: resnet18-eurosat 20260926-113004

Held-out eval_set: 210 tiles. **Accuracy: 95.24%**

Macro-F1: 0.9529 | Weighted-F1: 0.9529 | Micro-F1: 0.9524 (= accuracy for single-label classification)

## Per-class
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| AnnualCrop | 0.9355 | 0.9667 | 0.9508 | 30 |
| Forest | 1.0 | 1.0 | 1.0 | 30 |
| Highway | 0.9286 | 0.8667 | 0.8966 | 30 |
| Industrial | 1.0 | 0.9 | 0.9474 | 30 |
| Residential | 1.0 | 1.0 | 1.0 | 30 |
| River | 0.8286 | 0.9667 | 0.8923 | 30 |
| SeaLake | 1.0 | 0.9667 | 0.9831 | 30 |

## Confusion matrix (rows = true, cols = predicted)
| true \ pred | AnnualCrop | Forest | Highway | Industrial | Residential | River | SeaLake |
|---|---|---|---|---|---|---|---|
| AnnualCrop | 29 | 0 | 0 | 0 | 0 | 1 | 0 |
| Forest | 0 | 30 | 0 | 0 | 0 | 0 | 0 |
| Highway | 1 | 0 | 26 | 0 | 0 | 3 | 0 |
| Industrial | 0 | 0 | 2 | 27 | 0 | 1 | 0 |
| Residential | 0 | 0 | 0 | 0 | 30 | 0 | 0 |
| River | 1 | 0 | 0 | 0 | 0 | 29 | 0 |
| SeaLake | 0 | 0 | 0 | 0 | 0 | 1 | 29 |

## Accuracy vs coverage
| threshold | coverage | accuracy on accepted | sent to review |
|---|---|---|---|
| 0.0 | 100.00% | 0.9524 | 0 |
| 0.5 | 98.57% | 0.9614 | 3 |
| 0.6 | 97.62% | 0.9659 | 5 |
| 0.7 | 96.19% | 0.9802 | 8 |
| 0.8 | 91.43% | 0.9948 | 18 |
| 0.9 | 88.57% | 0.9946 | 24 |
| 0.95 | 81.90% | 1.0 | 38 |
| 0.99 | 70.48% | 1.0 | 62 |

## Errors (10), most confident first
| file | true | predicted | conf |
|---|---|---|---|
| tile_057.png | River | AnnualCrop | 0.9009 |
| tile_206.png | Highway | River | 0.7553 |
| tile_198.png | Highway | River | 0.7492 |
| tile_103.png | Industrial | Highway | 0.7153 |
| tile_114.png | Industrial | Highway | 0.6918 |
| tile_130.png | AnnualCrop | River | 0.6522 |
| tile_182.png | Highway | River | 0.6323 |
| tile_187.png | Highway | AnnualCrop | 0.5017 |
| tile_111.png | Industrial | River | 0.4856 |
| tile_174.png | SeaLake | River | 0.4347 |
