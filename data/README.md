# data/

The dataset is not included in this repo. Unzip `Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles.zip`
into `data/raw/` so it looks like this:

```
data/raw/
  candidate_tiles/<Class>/*.png   1,050 labelled tiles (7 classes x 150), used for training
  eval_set/*.png                  210 held-out tiles
  eval_labels.csv                 ground truth for eval_set (filename,true_label)
  README.txt
```

At runtime the service also creates `data/results.db` (SQLite) and `data/tiles/`
(raw uploaded tiles). Both are git-ignored.

Dataset: subset of EuroSAT (Helber et al., 2018/2019), Sentinel-2 imagery (Copernicus), MIT licensed.
