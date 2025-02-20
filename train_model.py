import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

df = pd.read_csv("recipes.csv")

df["ingredients"] = df["ingredients"].apply(lambda x: x if isinstance(x, str) else "")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["recipe_name"])  

model = NearestNeighbors(n_neighbors=1, metric="cosine")  
model.fit(X)

joblib.dump(model, "RecipeIngredients.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(df, "recipes_df.pkl") 

print("Model training completed and saved successfully!")
