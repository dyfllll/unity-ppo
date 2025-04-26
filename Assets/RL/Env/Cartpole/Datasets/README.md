-init -init1 是从Pytorch保存出的初始化权重
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
```



-py 是Pytorch训练出的结果，训练了不到200轮，需要取消勾选normalizeState

-normalizeState和-normalizeState2 是在normalize State情况下在Unity中训练的结果，大概500轮就没有失败的情况，需要勾选normalizeState，对于CartPole可以一直持续下去

