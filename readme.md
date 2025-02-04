# [Neural Networks] An end-to-end bi-objective approach to deep graph partitioning

This repository contains the implementation of the approach proposed in the paper "An end-to-end bi-objective approach to deep graph partitioning".


#### Dataset

A random directed graph with some self-loops is provided in the './graph/' directory. The code takes sparse adjacent matrix generated by *scipy.sparse* python package.

#### Enviroment requirements

The code mainly relys on Pytorch 2.1.

#### To run our method

```console
./run.sh
```

Note: when partitioning a graph in too many partitions, empty partitions may happens.


Please cite this paper if you use the model or any code from this repository in your own work:

```bibtex
@article{wei2025end,
  title={An end-to-end bi-objective approach to deep graph partitioning},
  author={Wei, Pengcheng and Fang, Yuan and Wen, Zhihao and Xiao, Zheng and Chen, Binbin},
  journal={Neural Networks},
  volume={181},
  pages={106823},
  year={2025},
  publisher={Elsevier}
}
```
