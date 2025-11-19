from pyvis.network import Network

groups = {
    "Additifs": ["additives_n", "additives_tags"],
    "Allergènes": ["allergens_tags"],
    "Marques": ["brands", "brands_tags"],
    "Catégories": ["categories", "categories_tags", "categories_properties"],
    "Scores": ["nutriscore_score", "nutriscore_grade", "nutriscore_tags", "ecoscore_score", "ecoscore_grade", "ecoscore_tags"],
    "Ingrédients": [
        "ingredients", "ingredients_tags", "ingredients_text", "ingredients_n", "ingredients_with_specified_percent_n",
        "ingredients_with_unspecified_percent_n", "ingredients_percent_analysis", "ingredients_from_palm_oil_n",
        "ingredients_without_ciqual_codes", "ingredients_without_ciqual_codes_n", "known_ingredients_n",
        "new_additives_n", "unknown_ingredients_n", "ingredients_original_tags"
    ],
    "Nutrition": ["nutriments", "nutrition_data_per", "no_nutrition_data", "nutrient_levels_tags", "unknown_nutrients_tags"],
    "Emballage": ["packaging", "packaging_tags", "packaging_text", "packagings", "packagings_complete", 
                  "packaging_shapes_tags", "packaging_recycling_tags"],
    "Produit": ["product_name", "generic_name", "quantity", "product_quantity", "product_quantity_unit", "serving_size", "serving_quantity"],
    "Origine": ["origins", "origins_tags", "manufacturing_places", "manufacturing_places_tags", "countries_tags"],
    "Dates": ["created_t", "last_modified_t", "last_image_t", "last_updated_t", "entry_dates_tags", "last_edit_dates_tags"],
    "Utilisateurs": ["creator", "editors", "correctors_tags", "informers_tags", "last_editor", "last_modified_by"],
}

net = Network(notebook=True, height="700px", width="100%", bgcolor="#ffffff", font_color="black")

# Ajouter les nœuds et liens
for group, features in groups.items():
    net.add_node(group, label=group, color="orange", shape="box")
    for feat in features:
        net.add_node(feat, label=feat, color="lightblue")
        net.add_edge(group, feat)

# Générer le fichier HTML
net.show("../output/openfoodfacts_groups.html")