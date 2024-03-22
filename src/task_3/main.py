from src.task_1_2.distance_measure import *
from src.task_1_2.model_classification import *
from task_3.preprocess import *
def evaluate_model(db_path, option='distance measure', resolution=resolution, is_correct_lighting=is_correct_lighting, is_blur=is_blur, is_extract_face=is_extract_face, scale=scale): 
    if option not in ["distance measure", "model classification", "neural network"]:
        raise ValueError("option should be either 'distance measure' or 'model classification'")
    data = preprocess(db_path, resolution=resolution, is_correct_lighting=is_correct_lighting, is_blur=is_blur, is_extract_face=is_extract_face, scale=scale)

    if option == "distance measure":
        evaluate_distance_measure(data["vectors"], data["labels"], )
    elif option == "model classification": 
        evaluate_model_classification(data["vectors"], data["labels"], )
    elif option == "neural network":
        
   