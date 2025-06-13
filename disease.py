import gradio as gr
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define class names in the order your model was trained on
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Define the prediction function
def predict(image):
    image = tf.image.resize(image, (256, 256))
    image = tf.expand_dims(image, 0)  # No division by 255
    prediction = model.predict(image)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return f"{predicted_class} ({confidence*100:.2f}%)"

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="label",
    title="Plant Disease Detector",
    description="Upload a leaf image to predict if it's Early Blight, Late Blight, or Healthy"
)

interface.launch()
