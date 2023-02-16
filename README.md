# STUCCO Experiments

This is the official experiments code for the paper [Soft Tracking Using Contacts for Cluttered Objects (STUCCO) to Perform Blind Object Retrieval](https://ieeexplore.ieee.org/document/9696372).
If you use it, please cite

```
@article{zhong2022soft,
  title={Soft tracking using contacts for cluttered objects to perform blind object retrieval},
  author={Zhong, Sheng and Fazeli, Nima and Berenson, Dmitry},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={3507--3514},
  year={2022},
  publisher={IEEE}
}
```

## Installation (experiments)

1. install [base experiments](https://github.com/UM-ARM-Lab/base_experiments) by following its readme
2. clone repository locally and `cd` into it
3. `pip install -e .`

## Usage
This is the full experiments to reproduce the results from the paper.
See the [light-weight library repository](https://github.com/UM-ARM-Lab/stucco) for how to use STUCCO
in your projects. 
See the [website](https://johnsonzhong.me/projects/stucco/) for videos and a high level introduction.


## Reproduce Paper

1. collect training data

```shell
python collect_tracking_training_data.py --task SELECT1 --gui
python collect_tracking_training_data.py --task SELECT2 --gui
python collect_tracking_training_data.py --task SELECT3 --gui
python collect_tracking_training_data.py --task SELECT4 --gui
```

2. evaluate all tracking methods on this data; you can find the clustering result and ground truth for each trial
   in `~/experiments/data/cluster_res`

```shell
python evaluate_contact_tracking.py
```

3. plot tracking method performances on the training data

```shell
python plot_contact_tracking_res.py
```

4. run simulated BOR tasks (there is a visual bug after resetting environment 8 times, so we split up the runs for
   different seeds)

```shell
python retrieval_main.py ours --task FB --seed 0 1 2 3 4 5 6 7; python retrieval_main.py ours --task FB --seed 8 9 10 11 12 13 14 15; python retrieval_main.py ours --task FB --seed 16 17 18 19
python retrieval_main.py ours --task BC --seed 0 1 2 3 4 5 6 7; python retrieval_main.py ours --task BC --seed 8 9 10 11 12 13 14 15; python retrieval_main.py ours --task BC --seed 16 17 18 19
python retrieval_main.py ours --task IB --seed 0 1 2 3 4 5 6 7; python retrieval_main.py ours --task IB --seed 8 9 10 11 12 13 14 15; python retrieval_main.py ours --task IB --seed 16 17 18 19
python retrieval_main.py ours --task TC --seed 0 1 2 3 4 5 6 7; python retrieval_main.py ours --task TC --seed 8 9 10 11 12 13 14 15; python retrieval_main.py ours --task TC --seed 16 17 18 19
```

repeat with baselines by replacing `ours` with `online-birch` and other baselines
