**torch.chunk(input: Tensor, chunk: int, dim: int =0) -> List of Tensors** 

**Tensor.chunk(chunks: Tensor, dim: int=0) -> List of Tensors** 

Splits a tensor into a specific number of chunks.(Last chunk will be smaller if the tensor size along the given dimension `dim` is not divisible by `chunks`, would be null.)

```python
>>>torch.chunk(torch.arange(8), chunks=2, dim=-1)
>>>(tensor([0, 1, 2, 3]), tensor([4, 5, 6, 7]))
```
```python
>>>torch.chunk(torch.arange(8), chunks=3, dim=-1)
>>>(tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7]))
```
```python
>>>a = torch.arange(8).view(2, 4)
>>>tensor([[0, 1, 2, 3],
          [4, 5, 6, 7]])
>>>torch.chunk(a, chunks=3, dim=0)
>>>(tensor([[0, 1],
          [4, 5]]),
   tensor([[2, 3],
          [6, 7]]))
>>>torch.chunk(a, chunks=2, dim=0)
>>>(tensor([[0, 1],
          [4, 5]]),
   tensor([[2, 3],
          [6, 7]]))
'''
LAST CHUNK COULD BE NULL
'''
```

**torch.split(tensor: Tensor, split_size_or_sections: int/List[int], dim: int=0)**

**Tensor.split(split_size, dim: int=0)**

Splits the tensor into chunks. 

If `split_size_or_sections` is an integer type, then [`tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor) will be split into equally sized chunks (if possible) the same as torch.chunk().

If `split_size_or_sections` is a list, then [`tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor) will be split into `len(split_size_or_sections)` chunks with sizes in `dim` according to `split_size_or_sections`.

```python
>>> a = torch.arange(10).reshape(5,2)
>>> a
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
>>> torch.split(a, 2)
(tensor([[0, 1],
         [2, 3]]),
 tensor([[4, 5],
         [6, 7]]),
 tensor([[8, 9]]))
>>> torch.split(a, [1,4])
(tensor([[0, 1]]),
 tensor([[2, 3],
         [4, 5],
         [6, 7],
         [8, 9]]))
```

**torch.unbind(input: Tensor,dim: int=0) -> seq** 

**Tensor.unbind(input: Tensor, dim: int=0) -> seq** 

Remove a tensor dimension and return as a list of Tensors.

```python
>>>torch.unbind(torch.tensor([[1,2,3],
                             [4,5,6],
                             [7,8,9]]))
>>>(tensor([1,2,3]),tensor([4,5,6]),tensor([7,8,9]))
```

```python
>>>a = torch.arange(8).view(2,2,2)
>>>tensor([
    	  [[0, 1],
           [2, 3]],
          [[4, 5],
           [6, 7]]
		 ])
>>>torch.unbind(a,dim=0)
>>>(tensor([[0, 1],
           [2, 3]]), 
   tensor([[4, 5],
           [6, 7]]))
```

**torch.Tensor.repeat(\*size) -> Tensor**

repeat this tensor along each dimension

```python
>>>x = torch.tensor([1,2,3])
>>>x.repeat(4,2)
tensor([[1,2,3,1,2,3],
        [1,2,3,1,2,3],
        [1,2,3,1,2,3],
        [1,2,3,1,2,3]])
```

**torch.scatter(input,dim,index,src) -> Tensor**

**torch.Tensor.scatter_(dim, index, src) -> Tensor**

```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

```python
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> src
tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2, 0]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
tensor([[1, 0, 0, 4, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 3, 0, 0]])
>>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
tensor([[1, 2, 3, 0, 0],
        [6, 7, 0, 0, 8],
        [0, 0, 0, 0, 0]])
```
```python
>>> x = torch.rand(2, 5)
>>> tensor([[0.1940, 0.3340, 0.8184, 0.4269, 0.5945],
            [0.2078, 0.5978, 0.0074, 0.0943, 0.0266]])

>>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)

>>> tensor([[0.1940, 0.5978, 0.0074, 0.4269, 0.5945],
            [0.0000, 0.3340, 0.0000, 0.0943, 0.0000],
            [0.2078, 0.0000, 0.8184, 0.0000, 0.0266]])
```

**torch.gather()**

**torch.Tensor.gather_()**