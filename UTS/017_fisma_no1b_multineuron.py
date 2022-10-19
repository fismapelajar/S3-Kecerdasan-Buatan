# Nama  : Fisma Meividianugraha Subani
# NIM   : 21091397017
# Kelas : 2021 A

# No 1b. Multi Neuron
#   i.   Input layer fature 10
#   ii.  Neuron 5

import numpy as np
# inisialisasi variabel inputs dengan panjang input layer = 10
inputs  = [1.3, 0.5, 2.0, 0.3, 3.1, 5.0, 0.2, 3.0, -1.3, -0.5]

# inisialisasi variabel weights berdasarkan jumlah neuron = 5 dengan panjang sesuai input layer = 10 
weights = [[1.0, 0.1, 1.9, 9.5, 0.1, 1.0, 9.1, 5.9, -1.0, -0.1],
           [1.9, 0.5, 1.9, 6.7, 9.1, 5.0, 9.1, 7.6, -1.9, -0.5],
           [1.3, 1.1, 1.9, 6.7, 3.1, 1.1, 9.1, 7.6, -1.3, -1.1],
           [1.0, 0.5, 2.0, 0.1, 0.1, 5.0, 0.2, 1.0, -1.0, -0.5],
           [2.8, 1.1, 2.0, 0.2, 8.2, 1.1, 0.2, 2.0, -2.8, -1.1]]

# inisialisasi variabel biases berdasarkan panjang neuron = 5
biases  = [1.0, 5.0, 1.1, 5.0, 1.1]

# Rumus dot product vector [np.dot(weight[0], input), np.dot(weight[1], input), np.dot(weight[2], input),
#                           np.dot(weight[3], input), np.dot(weight[4], input)] + biases
layer_outputs = np.dot(weights, inputs) + biases

# Menampilkan hasil output
print(layer_outputs)
