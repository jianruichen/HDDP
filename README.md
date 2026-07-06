# A Dynamics-GCN Hybrid Framework for Feature Learning in Disease-Related Association Prediction
## Datasets
- data/narna/c-d.csv is the circRNA_disease dataset, which contain 6437 associations between 212 circRNAs and 90 diseases.

- data/narna/l-d.csv is the lncRNA_disease dataset, which contain 139329 associations between 5176 lncRNAs and 598 diseases.

- data/didr/DDil.csv is the lncRNA_disease dataset, which contain 6677 associations between 597 drugs and 1229 diseases.

- data/didr/didr.csv is the lncRNA_disease dataset, which contain 18416 associations between 1519 drugs and 268 diseases.

- data/didr/dige.csv is the lncRNA_disease dataset, which contain 139329 associations between 2099 genes and 5858 diseases.

## Code
### Environment Requirement
The code has been tested running under Python 3.7.13 The required packages are as follows:
- numpy == 1.19.2
- scipy == 1.7.3
- tensorflow == 1.15.2
- torch == 1.13.1
- torch-cluster == 1.6.0+pt113cpu
- torch-geometric == 2.0.4
- torch-scatter == 2.0.9
- torch-sparse == 0.6.15+pt113cpu
- torch-spline-conv == 1.2.1+pt113cpu
##Running a model
To train the model in the paper, run this command:

```
python main.py 
```

