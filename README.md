# PennWorldEnvironment
A repository for the open source code of PennWorld! An exciting new RL environment 

## Setup
After cloning this repository, it can be setup in 3 easy steps:
First, import the conda environment:
```
conda env create -n pen_world_env --file demo/environment.yml
```
Then install the pen_world gym environment:
```
python setup.py install
```
Finally, enter the `demo` folder and run the demo file
```
cd demo
python demo.py
```
The jupyter notebook `demo.ipynb` can also be ran to an example of training with the environment.
