import argparse
import json

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

def load_labels(labels_path):
  with open(labels_path, 'r') as f:
    class_names = json.load(f)
  return class_names

def load_model(model_path):
  model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
  return model

def process_image(image_path):
  image = Image.open(image_path)
  image = np.asarray(image)
  image = tf.convert_to_tensor(image)
  image = tf.image.resize(image, (224, 224))
  image = tf.cast(image, tf.float32)
  image /= 255
  image = image.numpy()
  return image

def predict(image_path, model, top_k=5):
  image = process_image(image_path)
  image = np.expand_dims(image, axis=0)

  preds = model.predict(image)[0]
  ordered_indices = np.argsort(-preds)[:top_k]
  probs = preds[ordered_indices]
  classes = (ordered_indices+1).astype('str').tolist()
  return probs, classes

def create_parser():
  parser = argparse.ArgumentParser(prog="Image Classifier", description="Recognize different species of flowers")
  parser.add_argument("image_path", type=str, help="The path for the image file")
  parser.add_argument("model_path", type=str, help="The path for the classification model")
  parser.add_argument("--top_k", type=int, help="The number of top predictions to show")
  parser.add_argument("--category_names", type=str, help="The path to the file containing the class names for the labels")
  return parser
                       

def main():
  parser = create_parser()
  args = parser.parse_args()

  image_path = args.image_path
  model_path = args.model_path
  model = load_model(model_path)
  top_k = args.top_k if args.top_k else 5


  probs, classes = predict(image_path, model, top_k=top_k)

  if args.category_names:
    class_names = load_labels(args.category_names)
    labeled_classes = [class_names[label] for label in classes]
    
    flwr = "flowers" if top_k > 1 else "flower"
    prb = "Probabilities" if top_k > 1 else "Probability"
    print(f"Top {top_k} predicted {flwr}: ", labeled_classes)
    print(f"{prb} for the predicted {flwr}:", probs)
  else:
    clss = "classess" if top_k > 1 else "class"
    prb = "Probabilities" if top_k > 1 else "Probability"
    print(f"Top {top_k} predicted {clss}: ", classes)
    print(f"{prb} for the predicted {clss}:", probs)

if __name__ == "__main__":
  main()