## Implementation code for the paper:

**ANTHEM: Attentive Hyperbolic Entity Model for Product Search**, Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian and Chandan Reddy, WSDM 2022.

## Model Codes:
```
euclidean_intersection.py
hyperboloid.py
```

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
## Directory Structure
```
product_matching/
├── experiments/
│   ├── baseline_product_matching.py - Product matching experiments on the baselines
│   ├── euclidean_product_matching.py - Euclidean model for product matching
│   ├── hyperboloid_product_matching.py - Hyperboloid model for product matching
├── euclidean_intersection.py - Euclidean Intersection layer
├── hyperboloid.py - Hyperboloid Intersection layer
```
## Please refer this work if you find the code useful
```
@inproceedings{choudhary2022anthem,
author = {Choudhary, Nurendra and Rao, Nikhil and Katariya, Sumeet and Subbian, Karthik and Reddy, Chandan K.},
title = {ANTHEM: Attentive Hyperbolic Entity Model for Product Search},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {{WSDM} '22: The Fifteenth {ACM} International Conference on Web Search
               and Data Mining, Phoenix, AZ, USA, February 21-25, 2022},
location = {Phoenix, AZ, USA},
series = {WSDM '22}
}
```
