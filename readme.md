# CoLight 

#### This repository is provided as-is, further updates are included in [LibSignal](https://darl-libsignal.github.io/), which supports flexible and cross-platform (CityFlow and SUMO) training and testing in PyTorch. We also actively looking for contributions for LibSignal, feel free to reach out if you would love to contribute to the project!

CoLight is a reinforcement learning agent for network-level traffic signal control. 

```
@inproceedings{colight,
 author = {Wei, Hua and Xu, Nan and Zhang, Huichu and Zheng, Guanjie and Zang, Xinshi and Chen, Chacha and Zhang, Weinan and Zhu, Yamin and Xu, Kai and Li, Zhenhui},
 title = {CoLight: Learning Network-level Cooperation for Traffic Signal Control},
 booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '19},
 year = {2019},
 location = {Beijing, China}
} 
```

It shares the similar code structure with PressLight ([PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network](http://personal.psu.edu/hzw77/publications/presslight-kdd19.pdf)) from KDD 2019.

Usage and more information can be found below.

## Usage

How to run the code:

We recommend to run the code through docker. Some brief documentation can be found at https://docs.docker.com/.

1. Please build a docker image using the dockerfile provided.

``sudo docker pull hzw77/colight:v0.1``

2. Pull the codes for CoLight.
``git clone https://github.com/wingsweihua/colight.git``


3. Please run the built docker image to initiate a docker container. Please remember to mount the code directory.

``sudo docker run -it -v /path/to/your/workspace/colight/:/colight/ --shm-size=8gb --name hua_colight hzw77/colight:v0.1 /bin/bash``

``cd colight``

(Alternatively, you can install the packages (included in the dockerfile) on your linux system)

Start an experiment by:

``python -O runexp.py``

Here, ``-O`` option cannot be omitted unless debug is necessary. In the file ``runexp.py``, the args can be changed.

* ``runexp.py``

  Run the pipeline under different traffic flows. Specific traffic flow files as well as basic configuration can be assigned in this file. For details about config, please turn to ``config.py``.


For most cases, you might only modify traffic files and config parameters in ``runexp.py``.

## Dataset

* synthetic data

  Traffic file and road networks can be found in ``data/1_3`` && ``data/3_3`` && ``data/6_6`` && ``data/10_10``.

* real-world data

  Traffic file and road networks of New York City can be found in ``data/NewYork``, it contains two networks at different scale: 196 intersection and 48 intersections. Jinan and Hangzhou dataset are also included.



## Agent

* ``agent.py``

  An abstract class of different agents.

* ``CoLight_agent.py``

  Proposed CoLight agent

## Others

More details about this project are demonstrated in this part.

* ``config.py``

  The whole configuration of this project. Note that some parameters will be replaced in ``runexp.py`` while others can only be changed in this file, please be very careful!!!

* ``pipeline.py``

  The whole pipeline is implemented in this module:

  Start a simulator environment, run a simulation for certain time(one round), construct samples from raw log data, update the model and model pooling.

* ``generator.py``

  A generator to load a model, start a simulator enviroment, conduct a simulation and log the results.

* ``anon_env.py``

  Define a simulator environment to interact with the simulator and obtain needed data like features.

* ``construct_sample.py``

* Construct training samples from original data. Select desired state features in the config and compute the corrsponding average/instant reward with specific measure time.

* ``updater.py``

  Define a class of updater for model updating.

