import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    ### YOUR CODE HERE
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    result = cosine_similarity(vector1, vector2)

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    expected_result = dot_product / (magnitude1 * magnitude2)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    ### YOUR CODE HEREh

    vector = np.array([[1, 3], [2, 1], [1, 2]])  
    target_vector = np.array([1, 2])
    
    result = nearest_neighbor(target_vector, vector)
    
    expected_index = 2 
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
