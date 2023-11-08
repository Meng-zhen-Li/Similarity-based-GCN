# GraphCan

GraphCan is a similarity-based Graph Convolutional Network model for graph analysis tasks. It includes multiple types of similarity measures, and learns a canonical representation of a graph using both network topology and multiple types of similarity measures.

## Datasets
There are preprocessed data and the preprocessing code in the `data` folder. The original data can be found [here](https://github.com/xiangyue9607/BioNEV/tree/master).

## Running GraphCan

Use the following command to run GraphCan for a dataset:
```
python main.py --dataset DrugBank_DDI
```
The basic parameters of GraphCan:
- `--epochs`: The number of epochs to train the model.
- `--learning_rate`: The learning rate of training.
- `--hidden1`: The number of dimensions of the first GCN hidden layer.
- `--hidden2`: The number of dimensions of the second GCN hidden layer, and also the final output dimension.
- `--noise_level`: The percentage of noise added to the input graph.
- `sim_idx`: If you would to try GraphCan with only one similarity measure, use this parameter to indicate which one to use, otherwise all similarity measures are used. The indices of similarities:
  1. Adamic Adar
  2. Common Neighbor
  3. Von Neumann
  4. Random Walk with Restart
  5. Adjacency Matrix

## Citation
@inproceedings{li2023canonical,
  title={Canonical Representation of Biological Networks Using Graph Convolution},
  author={Li, Mengzhen and Co{\c{s}}kun, Mustafa and Koyut{\"u}rk, Mehmet},
  booktitle={Proceedings of the 14th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics},
  pages={1--9},
  year={2023}
}
