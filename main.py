import string_sum
from torch.utils.dlpack import from_dlpack

a, b = 100, 1000
print(f'{a} + {b} = {string_sum.sum_as_string(a, b)}')

xp = string_sum.eye(10)

print('Pointer', xp)

x = from_dlpack(xp)

print('Shape:', x.shape)
print('Stride:', x.stride())
print(x)