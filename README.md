# Hybrid Feature Selection to Boost the Balance between Performance and Fairness of Machine Learning Software


This repository stores our experimental codes for the paper “Hybrid Feature Selection to Boost the Balance between Performance and Fairness of Machine Learning Software”，PFFS is short for the method we propose in this paper: *Performance-Fairness Fisher Score*.



## Datasets

We use 5 datasets, all of which are widely used in fairness research: **Adult, COMPAS, German Credit, Bank Marketing, MEPS**. All these datasets can be loaded through python's aif360 package, for example:

```python
from aif360.datasets import AdultDataset
```



## Codes for Empirical Study

Our empirical research on removal of protected features and the performance and fairness trends of enlarging features without protected features is placed in the "RQ1-2" folder.

## Codes for PFFS

You can easily reproduce our method, we provide its code in RQ3 and RQ4. 

The filter.py in the two folders records the implementation process of PFFS, in which Fisher's calculation applies the fisher_score function in the skfeature library. 

We conduct experiments on eight scenarios on five datasets, where adult, compas and german all contain two protected features.



## Baseline method

We compare PFFS to random selection and observe how many features each method needs to select to reach a balance between performance and fairness.

## Acknowledgment
Many thanks to the code contribution of [*"Ignorance and Prejudice" in Software Fairness*](https://ieeexplore.ieee.org/document/9402057), our experiments are based on the work of this paper.
