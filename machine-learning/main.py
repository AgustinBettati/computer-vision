from hu_moments_generation import generate_hu_moments_file
from testing_model import load_and_test_svm
from training_svm_model import train_svm_model

generate_hu_moments_file()
train_svm_model()
load_and_test_svm()
