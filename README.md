# [VoroTO: Multiscale Topology Optimization of Voronoi Structures using Surrogate Neural Networks](https://arxiv.org/abs/2404.18300)

[Rahul Kumar Padhy](https://sites.google.com/view/rahulkp/home), [Krishnan Suresh](https://directory.engr.wisc.edu/me/faculty/suresh_krishnan), [Aaditya Chandrasekhar](https://aadityacs.github.io/)  
University of Wisconsin-Madison


## Abstract

Cellular structures found in nature exhibit remarkable properties such as high strength, high energy absorption, excellent thermal/acoustic insulation, and fluid transfusion. Many of these structures are Voronoi-like; therefore researchers have proposed Voronoi multi-scale designs for a wide variety of engineering applications. However, designing such structures can be computationally prohibitive due to the multi-scale nature of the underlying analysis and optimization. In this work, we propose the use of a neural network (NN) to carry out efficient topology optimization (TO) of multi-scale Voronoi structures. The NN is first trained using Voronoi parameters (cell site locations, thickness, orientation, and anisotropy) to predict the homogenized constitutive properties. This network is then integrated into a conventional TO framework to minimize structural compliance subject to a volume constraint. Special considerations are given for ensuring positive definiteness of the constitutive matrix and promoting macroscale connectivity. Several numerical examples are provided to showcase the proposed method.


## Citation

```
@misc{padhy2024voroto,
      title={VoroTO: Multiscale Topology Optimization of Voronoi Structures using Surrogate Neural Networks}, 
      author={Rahul Kumar Padhy and Krishnan Suresh and Aaditya Chandrasekhar},
      year={2024},
      eprint={2404.18300},
      archivePrefix={arXiv},
      primaryClass={cs.CE}
}
```
