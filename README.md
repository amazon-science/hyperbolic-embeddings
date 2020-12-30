
## Codes:
### Query2Box
### HypE

## Requirements
```
torch==1.2.0
tensorboadX==1.6
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
--data_path : Folder that contains train, test and validation 
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
dataloader.py - File to load data for the HypE models
manifolds

base.py - Base file with function definitions for the manifolds
euclidean.py - Implementation of the functions for Euclidean manifold
lorentz.py - Implementation of the functions for Lorentz manifold
poincare.py - Implementation of the functions for Poincaré manifold
model.py - File with the main model class (Query2Manifold)
optimizers

radam.py - Reimannian Optimizer for Hyperbolic gradient descent
run.py - File to run the model for different experiments
utils

hyperbolicity.py - Math functions
math_utils.py - Math functions
```
# License
This project is licensed under the Creative Commons Attribution 4.0 International (CC-BY 4.0) license.

