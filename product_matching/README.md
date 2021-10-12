## Paper:
### Path of paper's tex file: paper/samples/sample-sigconf.tex

## Model Codes:
### euclidean_intersection.py
### hyperboloid.py

## Requirements
```
tensorflow_manopt
```

## Integrate with Matchzoo
Download the development version of Matchzoo: https://github.com/NTMC-Community/MatchZoo/
```
pip install tensorflow_manopt
mv euclidean_intersection.py MatchZoo/matchzoo/models/
mv hyperboloid.py MatchZoo/matchzoo/models/
```
Add the following two lines to MatchZoo/matchzoo/models/__init__.py
```
from .euclidean_intersection import EuclideanIntersection
from .hyperboloid import Hyperboloid
```

MatchZoo documentation covers the running of ranking and classification tasks.
To use the given models, use:
```
model = matchzoo.models.EuclideanIntersection
model = matchzoo.models.Hyperboloid
```

experiments/ directory contains the experiment scripts
```
python experiments/euclidean_product_matching.py
python experiments/hyperboloid_product_matching.py
python experiments/baseline_product_matching.py
```
