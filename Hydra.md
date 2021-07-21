# Hydra

**[Hydra官网](https://hydra.cc/docs/next/intro/#installation)** 文档已经写得很好了，去看官方文档！

### Installation

```python
pip install hydra-core --upgrade
```

### Hydra on Command-Line

Your config file should have the '.yaml' file extension.

```yaml
config.yaml
-----------------
dataset:
	data: CIFAR10
	batchsize: 128
optimizer:
	optim: SGD
	lr: 1.0e-6
	momentum: 0.9
```

Passing  your `config_name` as a parameter to  the `@hydra.main()` decorator. Hydra also needs to konw where to find your config.(you need to pass your `config_path` to `@hydra.main()`)

```python
my_py.py
------------------
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    # print(OmegaConf.to_yaml(cfg))
```

Run my_py.py :

```bash
$ python my_py.py
{'dataset': {'data': 'CIFAR10', 'batchsize': 128}, 'optimizer':{'optim': 'SGD', 'lr': 1e-06, 'momentum': 0.9}}
```

For the comfortable reading :

```python
my_py.py
------------------
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # print(cfg)
    print(OmegaConf.to_yaml(cfg))
```

Run my_py.py:

```bash
$ python my_py.py
dataset:
	data: CIFAR10
	batchsize: 128
optimizer:
	opim: SGD
	lr: 1e-06
	momentum: 0.9
```

You can **override** values **in command line**.

```bash
$ python my_py.py dataset.batchsize=256
dataset:
	data: CIFAR10
	batchsize: 256
optimizer:
	opim: SGD
	lr: 1e-06
	momentum: 0.9
```

**Use the lack of the `+` prefix.**

```bash
$ python my_py.py +model.backbone=resnet18 +model.ftrs_dim=128
dataset:
	data: CIFAR10
	batchsize: 128
optimizer:
	opim: SGD
	lr: 1e-06
	momentum: 0.9
model:
	backbone: resnet18
	ftrs_dim: 128
```

Use **`++`** to **override** config value if it's already in the config, or **add it** otherwise.

```bash
$ python my_py.py ++dataset.data=CIFAR100 ++version=-1
dataset:
	data: CIFAR100
	batchsize: 128
optimizer:
	opim: SGD
	lr: 1e-06
	momentum: 0.9
version: -1
```

### Hydra : Using Config Object

```yaml
config.yaml
------------------
node:                         # Config is hierarchical
  loompa: 10                  # Simple value
  zippity: ${node.loompa}     # Value interpolation
  do: "oompa ${node.loompa}"  # String interpolation
  waldo: ???                  # Missing value, must be populated prior to access
```

```python
my_py.py
------------------
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    assert cfg.node.loompa == 10          # attribute style access
    assert cfg["node"]["loompa"] == 10    # dictionary style access

    assert cfg.node.zippity == 10         # Value interpolation
    assert isinstance(cfg.node.zippity, int)  # Value interpolation type
    assert cfg.node.do == "oompa 10"      # string interpolation

    cfg.node.waldo                        # raises an exception
```

Run my_py.py

```bash
$ python my_py.py
Traceback (most recent call last):
  File "my_py.py", line 13, in my_py
    cfg.node.waldo
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: node.waldo
    full_key: node.waldo
    object_type=dict
```

### Hydra on Grouping Config Files

Suppose you have **TWO** config files: `cifar.yaml` and `imagenet.yaml`

Your Directory layout:

```bash
├─ cfg
│  └─ db
│      ├─ cifar.yaml
│      └─ imagenet.yaml
└── my_app.py
```

```yaml
db/cifar.yaml
----------------
dataset: CIFAR10
model: resnet50
```

```yaml
db/imagenet.yaml
----------------
datapath: /path/to/imagenet
model: vgg13
```

Since we moved all the configs into the `conf` directory, we need to tell Hydra where to find them using the `config_path` parameter. **`config_path` is a directory relative to `my_py.py`**.

```python
@hydra.main(config_path="conf")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
```

Running `my_app.py` without requesting a configuration will print an empty config.

```bash
$ python my_app.py
{}
```

Select an item from a config group with `+GROUP=OPTION`, e.g:

```bash
$ python my_py.py +db=cifar
db:
  dataset: CIFAR10
  model: resnet50
```

### Hydra on Grouping Config Files BUT Selecting Default Config

Suppose you want to use cifar.yaml but no longer want to type +db=cifar every time you run your .py.

You can add a **Default List** to your config file. A **Defaults List** is a list telling Hydra how to compose the final config object. By convention, it is the first item in the config.

```bash
├─ cfg
│  └─ db
│  │   ├─ cifar.yaml
│  │   └─ imagenet.yaml
│  └─ config.yaml
└── my_app.py
```

And your config.yaml **(Note that there is a `-` before `db`)**

```yaml
config.yaml
----------------
defaults:
  -  db: cifar.yaml
```

Run my_py.py

```bash
$ python my_py.py
db:
  dataset: CIFAR10
  model: resnet50
```

You can still override or add value as before.

```bash
$ python my_py.py db=imagenet +db.batchsize=128
db:
  dataset: imagenet
  model: vgg
  batchsize: 128
```

You can remove a default entry from the defaults list by prefixing it with ~

```bash
$ python my_app.py ~db
{}
```

Sometimes a config file does not belong in any config group. You can still load it by default but cannot be overridden.

```yaml
defaults:
  - somfile
```

### Hydra on Grouping Config Files WITH all together

Suppose we have complex config files like this :

```bash
├── conf
│   ├── config.yaml
│   ├── db
│   │   ├── mysql.yaml
│   │   └── postgresql.yaml
│   ├── schema
│   │   ├── school.yaml
│   │   ├── support.yaml
│   │   └── warehouse.yaml
│   └── ui
│       ├── full.yaml
│       └── view.yaml
└── my_app.py
```

and your config.yaml :

```yaml
config.yaml
-------------
defaults:
  - db: mysql
  - ui: full
  - schema: school
```

So you get :

```bash
$ python my_app.py
db:
  driver: mysql
  user: omry
  pass: secret
ui:
  windows:
    create_db: true
    view: true
schema:
  database: school
  tables:
  - name: students
    fields:
    - name: string
    - class: int
  - name: exams
    fields:
    - profession: string
    - time: data
    - class: int
```

You can also override value as before `db.user=Admin123` etc.

 ### Set New Key-Value in Python Code:

Suppose your cfg.yaml as 

```yaml
a:
	aa: 1
	bb: 2
```

if you want to set a new key-value like:

```python
import hydra

@hydra.main(config_path='.', config_name='cfg.yaml')
def fun(args):
    args.cc = 3
```

raise a Error

```bash
omegaconf.errors.ConfigKeyError: Key 'cc' is not in struct
	full_key: cc
	object_type=dict
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

you can achive it when your code like this

```python
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path='.', config_name='cfg.yaml')
def fun(args):
    args = OmegaConf.structured(OmegaConf.to_yaml(args))
    args.cc = 3
```

or:

```python
import hydra
from omegaconf import OmegaConf,open_dict

@hydra.main(config_path='.', config_name='cfg.yaml')
def fun(args):
    OmegaConf.set_struct(args, True)
    with open_dict(args):
        args.cc = 3
```

