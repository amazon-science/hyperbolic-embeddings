
## Implementation  for the papers:


**Self-Supervised Hyperboloid Representations from Logical Queries over Knowledge Graphs**, Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian and Chandan Reddy, WWW 2021.

Code directory: ```codes/```

**ANTHEM: Attentive Hyperbolic Entity Model for Product Search**, Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian and Chandan Reddy, WSDM 2022.

Code directory: ```product_matching/```

## Requirements
```
torch==1.2.0
tensorboardX==1.6
```

## Run
To reproduce the results on FB15k, FB15k-237 and NELL, the hyperparameters are set in `example.sh`.
```
cd code/
bash example_hype.sh
bash example_q2b.sh
```

## Arguments:
```
--do_train : Boolean that indicates if model should be trained
--cuda : Boolean that indicates if cuda should be used
--do_valid : Boolean that indicates if model should use validation
--do_test : Boolean that indicates if model should be tested to log metrics
--data_path : Folder that contains train, test and validation data
--model : Use 2-dimensions or one dimension for the model
-n : Number of negative samples per positive sample
-b : Batch size for training
-d : Dimension of embeddings (should be equal to semantic vector dimensions)
-lr : Learning rate of the model
--max_steps : Max number of epochs
--cpu_num : number of CPUs
--test_batch_size : Batch size for testing
--center_reg : Regularization factor for center updates
--geo : Box embeddings or Vec embeddings
--task : Tasks for training
--stepsforpath : Same as number of epochs
--offset_deepsets : Aggregation methods for offsets
--center_deepsets : Aggregation methods for centers
--manifold : Set manifolds
-c : Initial curvature value
--use_semantics : Boolean that indicates if model should use semantic embeddings
--trainable_curvature : Boolean that indicates if model should use trainable curvature
--print_on_screen : Output should print on screen
```

## Code details
```
codes/
├── dataloader.py - File to load data for the HypE models
├── manifolds
│   ├── __init__.py
│   ├── base.py - Base file with function definitions for the manifolds
│   ├── euclidean.py - Implementation of the functions for Euclidean manifold
│   ├── lorentz.py - Implementation of the functions for Lorentz manifold
│   └── poincare.py - Implementation of the functions for Poincaré manifold
├── model.py - File with the main model class (Query2Manifold)
├── optimizers
│   ├── __init__.py
│   └── radam.py - Reimannian Optimizer for Hyperbolic gradient descent
├── run.py - File to run the model for different experiments
└── utils
    ├── __init__.py
    ├── hyperbolicity.py - Math functions
    └── math_utils.py - Math functions
```
```
product_matching/
├── experiments/
│   ├── baseline_product_matching.py - Product matching experiments on the baselines
│   ├── euclidean_product_matching.py - Euclidean model for product matching
│   ├── hyperboloid_product_matching.py - Hyperboloid model for product matching
├── euclidean_intersection.py - Euclidean Intersection layer
├── hyperboloid.py - Hyperboloid Intersection layer
```
## Please refer these works if you find the code useful:
```
@inproceedings{10.1145/3442381.3449974,
author = {Choudhary, Nurendra and Rao, Nikhil and Katariya, Sumeet and Subbian, Karthik and Reddy, Chandan K.},
title = {Self-Supervised Hyperboloid Representations from Logical Queries over Knowledge Graphs},
year = {2021},
isbn = {9781450383127},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3442381.3449974},
doi = {10.1145/3442381.3449974},
booktitle = {Proceedings of the Web Conference 2021},
pages = {1373–1384},
numpages = {12},
keywords = {knowledge graphs, hyperbolic space, Representation learning, reasoning queries},
location = {Ljubljana, Slovenia},
series = {WWW '21}
}
```
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

