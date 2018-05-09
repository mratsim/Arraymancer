import numpy as np

# native endian native int
a = np.array([1, 2, 3, 4])
np.save('int.npy', a)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
np.save('int_2D_c.npy', a)

a = np.asfortranarray(a)
np.save('int_2D_f.npy', a)

# little endian float32
a = np.array([1, 2, 3, 4], dtype = '<f4')
np.save('f32LE.npy', a)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype = '<f4')
np.save('f32LE_2D_c.npy', a)

a = np.asfortranarray(a)
np.save('f32LE_2D_f.npy', a)

# big endian uint64
a = np.array([1, 2, 3, 4], dtype = '>u8')
np.save('u64BE.npy', a)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype = '>u8')
np.save('u64BE_2D_c.npy', a)

a = np.asfortranarray(a)
np.save('u64BE_2D_f.npy', a)
