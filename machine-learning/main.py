from hu_moments_generation import generate_hu_moments_file
from testing_model import load_and_test
from training_model import train_model

show_contours = False
generate_hu_moments_file(show_contours)
train_model()
load_and_test(show_contours)
