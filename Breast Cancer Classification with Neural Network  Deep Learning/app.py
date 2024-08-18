# Import necessary modules
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np

# Create a Flask application instance
app = Flask(__name__)

# Load the trained model using TensorFlow
model = tf.keras.models.load_model("breast_cancer_classification_model.h5")

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    input_data = [float(request.form.get(field)) for field in
                  ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                   "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
                   "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                   "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
                   "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                   "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
                   "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]]
    
    # Combine inputs into a single list
    input_data = np.asarray([input_data])

    # Make a prediction using the model
    prediction = model.predict(input_data)
    prediction_class = (prediction > 0.5).astype(int)

    # Determine the output message
    if prediction_class[0] == 0:
        prediction_text = 'The person does not have breast cancer.'
    else:
        prediction_text = 'The person has breast cancer.'

    # Pass the prediction value to the template
    return render_template("index.html", prediction=prediction_text)

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True)
