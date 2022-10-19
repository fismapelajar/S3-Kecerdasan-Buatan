# Nama  : Fisma Meividianugraha Subani
# NIM   : 21091397017
# Kelas : 2021 A

# No 1a. Single Neuron
# 	i.   Input layer fature 10
#   ii.  Neuron 1

import numpy as np
# inisialisasi variabel inputs dengan panjang input layer = 10
inputs  = [1.3, 0.5, 2.0, 0.3, 3.1, 5.0, 0.2, 3.0, -1.3, -0.5]

# inisialisasi variabel weights berdasarkan jumlah neuron = 1 dengan panjang sesuai input layer = 10 
weights = [1.0, 0.1, 1.9, 9.5, 0.1, 1.0, 9.1, 5.9, -1.0, -0.1]

# inisialisasi variabel bias berdasarkan panjang neuron = 1
bias    = 4.0

# Rumus dot product vector [inputs[0-9]*weights[0-9]] + [bias]
outputs = np.dot(weights, inputs) + bias

# Menampilkan hasil output
print(outputs)
