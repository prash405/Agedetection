import cv2
from deepface import DeepFace

image_path = "F:\devops\pic.jpeg"


def detect_gender_age(image_path):
   
    analysis = DeepFace.analyze(image_path, actions=['age', 'gender'])
    
    
    gender = analysis[0]['gender']
    age = analysis[0]['age']
    
    print(f"Detected Gender: {gender}")
    print(f"Detected Age: {age}")

    return gender, age

if __name__ == "__main__":
    detect_gender_age(image_path)
