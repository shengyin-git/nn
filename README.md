process_data_mask.py is used to get the masked image, transform the picking position to the center, rotate to the picking angle, and cut to a pre-defined size.

copy process_data_mask.py to the method_random_full directory and run "python process_data_mask.py" to get the data set for training.

move the generated dataset and replace the data dir.

run "python test_nn_pytorch_bce_data" to train the neural network. 
