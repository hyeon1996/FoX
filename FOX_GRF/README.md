
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
