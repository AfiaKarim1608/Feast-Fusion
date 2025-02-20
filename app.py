from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model = joblib.load("RecipeIngredients.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df = joblib.load("recipes_df.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    recipe_name = request.form["recipe_name"] 
    input_vector = vectorizer.transform([recipe_name])
    _, indices = model.kneighbors(input_vector)
    matched_index = indices[0][0]
    matched_recipe = df.iloc[matched_index]
    ingredients = matched_recipe["ingredients"].split(",")  

    return render_template("result.html", recipe_name=recipe_name, ingredients=ingredients)

if __name__ == "__main__":
    app.run(debug=True)


