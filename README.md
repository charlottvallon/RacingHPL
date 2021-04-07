Hierarchical Predictive Learning for Racing
===============

We use [Hierarchical Predictive Learning](https://arxiv.org/abs/2005.05948) to race an autonomous [BARC](http://www.barc-project.com/) vehicle around 1/10 scale [Formula 1 tracks](https://github.com/jbelien/F1-Circuits). 
Hierarchical Predictive Learning (HPL) is a data-driven control scheme based on high-level strategies learned from previous task trajectories (i.e. racelines from a training set of Formula 1 tracks). Strategies map a forecast of the environment (i.e. upcoming track curvature) to a target set (i.e. distance from centerline) for which the system should aim across a horizon of N steps. 
These strategies are applied to the new task in real-time using a forecast of the upcoming environment, and the resulting output is used as a terminal region by a low-level MPC controller. 
Here we use Gaussian Processes trained on previously collected task trajectories to represent the learned strategies.

![AE_velcomp](https://user-images.githubusercontent.com/21371184/113793261-b8b92b00-96fc-11eb-8b2a-818a89cb5831.jpg)
The HPL controller is able to closely mimic the raceline of new, unseen tracks despite using a short horizon environment forecast.

Running Code
---------------------------

1. Install numpy, pyomo, gpytorch.
2. `python main.py`


Citing Work
---------------------------
1. Charlott Vallon and Francesco Borrelli. "Data-driven hierarchical predictive learning in unknown environments." In IEEE CASE (2020).
