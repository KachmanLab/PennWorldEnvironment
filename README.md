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

<br/><br/>
Copyright 2023 Shakeeb Majid

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
