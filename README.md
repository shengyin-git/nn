process_data_mask.py is used to get the masked image, transform the picking position to the center, rotate to the picking angle, and cut to a pre-defined size.

copy process_data_mask.py to the method_random_full directory and run "python process_data_mask.py" to get the data set for training.

move the generated dataset and replace the data dir.

run "python test_nn_pytorch_bce_data" to train the neural network. 

test_nn_pytorch_bce.py is the script for testing with cats and dogs.

test_nn_pytorch_bce_data.py is the script for testing with data_224 using sigmoid and binary cross entropy loss. 

test_nn_pytorch_softmax_data.py is the script for testing with data_224 using softmax and cross entropy loss.

test_nn_pytorch_bce_data_balance.py is the script for testing with data_224 (equal number of label 0 and label 1) using sigmoid and binary cross entropy loss. 

test_nn_pytorch_softmax_feature.py is the script for testing with data_224 with resnet extracted features already using softmax and cross entropy loss.

reload_nn.py is for testing the trained neural network. 

test_nn_pytorch_bce_feature.py is the one recommended.

