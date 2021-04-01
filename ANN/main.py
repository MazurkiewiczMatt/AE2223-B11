import preprocessing
import aircraft_recognition

nn = aircraft_recognition.train_nn(dataset, hidden_layers = 1)
print(nn(test_image))

# process:
# 1. upload images
# 2. make them same size and make them grayscale 
# 3. turn images into numpy arrays
# 4. 