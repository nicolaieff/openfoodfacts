generator = pipeline("text-generation", model="OpenLLM-France/Lucie-7B-Instruct-v1.1") 


def generate_category(tags):
    prompt = (
    f"Here is a list of food-related tags: {', '.join(tags)}.\n"
    f"Based on this list, what is the most appropriate high-level food category?\n"
    f"Answer with only one category:"
    )
    result = generator(prompt, max_new_tokens=10, do_sample=True, temperature=0.7)[0]['generated_text']
    return result.split("Suggest a high-level category:")[-1].strip().split("\n")[0]

# d_cat = d_cat.with_columns(
#     pl.col("product_name").apply(generate_category).alias("llm_tag")
#     )
from tqdm import tqdm

# Appliquer à la colonne Polars (ex: cleaned_tags)
for tags in df["categories_tags"].to_list():
    print(tags, generate_category(tags))

# Ajouter la nouvelle colonne
# df = df.with_columns(pl.Series("predicted_category", results))


import pandas as pd
from transformers import pipeline
import torch

# Exemple de DataFrame
df = pd.DataFrame({
    "tags": [
        ["en:snacks", "en:salty-snacks", "de:sojabohnen"],
        ["en:cheese", "fr:fromage", "it:formaggio"],
        ["en:meat", "en:poultry", "es:pollo"],
        ["fr:jus", "en:fruit-juice", "de:fruchtsaft"]
    ]
})

# Étape 1 : Nettoyer les tags (enlever les codes langues)
def clean_tags(tag_list):
    return [tag.split(":")[1].replace("-", " ") for tag in tag_list if ":" in tag]

df["cleaned_tags"] = df["tags"].apply(clean_tags)

# Étape 2 : Utiliser un modèle pour générer une catégorie (résumé ou label)
# On prend un modèle de génération ou classification légère
generator = pipeline("text-generation", model="bigscience/bloomz-560m")  # Peut être remplacé par "tiiuae/falcon-rw-1b", etc.

def generate_category(tags):
    prompt = f"Given the following tags: {', '.join(tags)}. Suggest a high-level category:"
    result = generator(prompt, max_new_tokens=10, do_sample=True, temperature=0.7)[0]['generated_text']
    # Extraire juste la réponse après le prompt
    return result.split("Suggest a high-level category:")[-1].strip().split("\n")[0]

df["predicted_category"] = df["cleaned_tags"].apply(generate_category)

print(df[["tags", "predicted_category"]])
