在Facebook的Detectron2框架中，广泛使用了注册机制Registry。

#### 1. 注册机制Registry的使用

```python
registry_machine = Registry('registry_machine')

@registry_machine.register()
def print_hello_world(word):
    print('hello {}'.format(word))


@registry_machine.register()
def print_hi_world(word):
    print('hi {}'.format(word))

if __name__ == '__main__':

    cfg1 = 'print_hello_world'
    registry_machine.get(cfg1)('world')

    cfg2 = 'print_hi_word'
    registry_machine.get(cfg2)('world')
```

可以看到，**如果创建了一个Registry的对象，并在方法/类定义的时候用装饰器装饰它，则可以通过 registry_machine.get(方法名)的 办法来间接的调用被注册的函数**

#### 2. 为什么使用Registry

对于detectron2这种，需要支持许多不同的模型的大型框架，理想情况下所有的模型的参数都希望写在配置文件中，那问题来了，如果我希望根据我的配置文件，决定我是需要用VGG还是用ResNet ，我要怎么写呢？

菜鸡如我，必写出这种代码：

```python
if class_name == 'VGG':
    model = build_VGG(args)
elif class_name == 'ResNet':
    model = build_ResNet(args)
```

但是如果用了注册类，代码就是这样的：

```python
class_name = 'VGG' # 'ResNet'
model = model_registry.get(class_name)(args)
```

可以看到代码的可扩展性变得非常强了

#### 3. Registry 源码

```python
class Registry():
    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret
```

完整版的`Registry`的源码见 [facebook: Registry](https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py)

#### Ref

[[Detectron2] 01-注册机制 Registry 实现](https://zhuanlan.zhihu.com/p/93835858)

