This repository considers the implementation of the paper

# FoX: Formation-aware exploration in multi-agent reinforcement learning

Which has been accepted to AAAI 2024, and is available in (https://ojs.aaai.org/index.php/AAAI/article/view/29196)


## Running an experiment 

```shell
python3 main.py --config=FOX_QMIX --env-config=academy_3_vs_1_with_keeper with beta1=0.01 beta2=0.02 round=3
```

or

```shell
python3 src/main.py --config=FOX_QMIX --env-config=sc2 with env_args.map_name=3m beta1=0.02 beta2=0.02 round=1
```

The config files act as defaults for an algorithm or environment. 

They are all located in `config`.

where `--config` refers to the configuration of the algorithm in `config/algs`, 

where `--env-config` refers to the configuration of the environment in `config/envs`

The required packages can be found in "requirements.txt", but installing with the file is not recommended.

## Publication

If you find this repository useful, please cite our paper:
```
@inproceedings{jo2024fox,
  title={FoX: Formation-aware exploration in multi-agent reinforcement learning},
  author={Jo, Yonghyeon and Lee, Sunwoo and Yeom, Junghyuk and Han, Seungyul},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={12},
  pages={12985--12994},
  year={2024}
}
```
