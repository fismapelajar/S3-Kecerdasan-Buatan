# Nama  : Fisma Meividianugraha Subani
# NIM   : 21091397017
# Kelas : 2021 A

# No 1a. Multi Neuron Batch Input
#   i.   Input layer fature 10
#   ii.  Per batchnya 6 input 
#   iii. Hidden layer 1, 5 neuron
#   iv.  Hidden layer 2, 3 neuron

# inisialisasi library numpy
import numpy as np
# inisialisasi variabel inputs berdasarkan jumlah batch = 6 dan input layer feature = 10 (matriks 6x10)
inputs   = [[1.3, 0.5, 2.0, 0.3, 3.1, 5.0, 0.2, 3.0, -1.3, -0.5],
            [3.0, 0.9, 2.0, 0.3, 0.3, 9.0, 0.2, 3.0, -3.0, -0.9],
            [8.0, 0.3, 2.0, 0.3, 0.8, 3.0, 0.2, 3.0, -8.0, -0.3],
            [1.6, 1.2, 2.0, 2.0, 6.1, 2.1, 0.2, 0.2, -1.6, -1.2],
            [6.0, 7.0, 2.0, 0.3, 0.6, 0.7, 0.2, 3.0, -6.0, -7.0],
            [1.9, 1.0, 2.0, 2.2, 9.1, 0.1, 0.2, 2.2, -1.9, -1.0]]

# inisialisasi variabel weights1 [Hidden layer 1] berdasarkan jumlah neuron = 5 dan input layer feature = 10 (matriks 5x10)
weights1 = [[1.0, 0.1, 1.9, 9.5, 0.1, 1.0, 9.1, 5.9, -1.0, -0.1],
            [1.9, 0.5, 1.9, 6.7, 9.1, 5.0, 9.1, 7.6, -1.9, -0.5],
            [1.3, 1.1, 1.9, 6.7, 3.1, 1.1, 9.1, 7.6, -1.3, -1.1],
            [1.0, 0.5, 2.0, 0.1, 0.1, 5.0, 0.2, 1.0, -1.0, -0.5],
            [2.8, 1.1, 2.0, 0.2, 8.2, 1.1, 0.2, 2.0, -2.8, -1.1]]

# inisialisasi variabel biases1 berdasarkan panjang neuron = 5 [Hidden layer 1]
biases1  = [5.0, 9.0, 3.0, 2.1, 0.7]

# inisialisasi variabel weights2 [Hidden layer 2] berdasarkan jumlah neuron = 3 dan input layer feature = 5 (matriks 3x5)
weights2 = [[2.4, 1.0, 2.0, 2.2, 4.2],
            [1.7, 0.8, 1.9, 4.5, 7.1],
            [1.0, 1.1, 1.9, 4.5, 0.1]]

# inisialisasi variabel biases2 berdasarkan panjang neuron = 3 [Hidden layer 2]
biases2  = [0.1, 8.0, 1.1]

# Rumus dot product vector [inputs batch * hasil transpose weights1] + [biases]
#                          [(matriks 6x10) * (matriks 10x5)] + [biases1[1], biases1[2], biases1[3], biases1[4], biases1[5]]
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# Rumus dot product vector [layer_outputs1 * hasil transpose weights2] + [biases]
#                          [(matriks 6x5) * (matriks 5x3)] + [biases2[1], biases2[2], biases2[3]]
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# Menampilkan hasil output
print(layer2_outputs)