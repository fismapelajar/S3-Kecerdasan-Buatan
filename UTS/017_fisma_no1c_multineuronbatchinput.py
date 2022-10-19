# Nama  : Fisma Meividianugraha Subani
# NIM   : 21091397017
# Kelas : 2021 A

# No 1c. Multi Neuron Batch Input
#   i.   Input layer fature 10
#   ii.  Per batchnya 6 input 
#   iii. Neuron 5

import numpy as np
# inisialisasi variabel inputs berdasarkan jumlah batch = 6 dengan panjang sesuai input layer = 10 [matriks 6x10]
inputs  = [[1.3, 0.5, 2.0, 0.3, 3.1, 5.0, 0.2, 3.0, -1.3, -0.5],
           [3.0, 0.9, 2.0, 0.3, 0.3, 9.0, 0.2, 3.0, -3.0, -0.9],
           [8.0, 0.3, 2.0, 0.3, 0.8, 3.0, 0.2, 3.0, -8.0, -0.3],
           [1.6, 1.2, 2.0, 2.0, 6.1, 2.1, 0.2, 0.2, -1.6, -1.2],
           [6.0, 7.0, 2.0, 0.3, 0.6, 0.7, 0.2, 3.0, -6.0, -7.0],
           [1.9, 1.0, 2.0, 2.2, 9.1, 0.1, 0.2, 2.2, -1.9, -1.0]]

# inisialisasi variabel weights berdasarkan jumlah neuron = 5 dengan panjang sesuai input layer = 10 [matriks 5x10]
weights = [[1.0, 0.1, 1.9, 9.5, 0.1, 1.0, 9.1, 5.9, -1.0, -0.1],
           [1.9, 0.5, 1.9, 6.7, 9.1, 5.0, 9.1, 7.6, -1.9, -0.5],
           [1.3, 1.1, 1.9, 6.7, 3.1, 1.1, 9.1, 7.6, -1.3, -1.1],
           [1.0, 0.5, 2.0, 0.1, 0.1, 5.0, 0.2, 1.0, -1.0, -0.5],
           [2.8, 1.1, 2.0, 0.2, 8.2, 1.1, 0.2, 2.0, -2.8, -1.1]]

# inisialisasi variabel biases berdasarkan panjang neuron = 5
biases  = [5.0, 9.0, 3.0, 2.1, 0.7]

# Rumus dot product vector [inputs batch * hasil transpose weights] + [biases]
#                          [(matriks 6x10) * (matriks 10x5)] + [biases1, biases2, biases3, biases4, biases5]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases

# Menampilkan hasil output
print(layer_outputs)
