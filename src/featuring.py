import polars as pl
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import CONFIG, CAT_CUSTOM, CAT2_CUSTOM
from src.country_map import COUNTRY_GROUP

file_clean = CONFIG['file_clean']
nutri = CONFIG['selected_nutrient']
fats_nuts_seeds = CONFIG['fats_nuts_seeds']
seed = CONFIG['seed']
feats = CONFIG['feats_model']
target = CONFIG['target']
beverage_tag = CONFIG['beverage_tag']


def dict_cat(d):
    return {
    tag: category
    for category, tags in d.items()
    for tag in tags
    }


def custom_tags(df, col_init, col_cust, nb, d):

    tag_to_category = dict_cat(d)

    df = df.with_columns([
        pl.col(col_init).map_elements(
            lambda tag_list: list(
                {tag_to_category[tag] for tag in tag_list 
                if tag in tag_to_category}
            ),
            return_dtype=pl.List(pl.String)
        ).alias(col_cust)
    ])

    df = df.with_columns([
        pl.when(pl.col(col_cust).list.len() < nb)
        .then(
            pl.col(col_cust).list.concat([None] * nb)
        )
        .otherwise(pl.col(col_cust))
        .alias(col_cust)
    ])
    df = df.with_columns([
        pl.col(col_cust).list.get(i).alias(f"{col_cust}_{i}") for i in range(nb)
        ]).drop(col_cust)

    return df


def filter_cats(df, name, set_cats, drop=False):
    df = df.with_columns([
        pl.col("food_groups_tags").map_elements(
        lambda cats: int(bool(set(cats) & set_cats)),
        return_dtype=pl.Int8
        ).alias(f"is_{name}")
        ])
    if drop:
        df = df.filter(pl.col(f"is_{name}") == 0).drop(f"is_{name}")
    return df


def kcal_revise(df):
    df = df.with_columns(
    (pl.col("energy-kj") / 4.184).alias("energy-kj/4.184")
    )

    df = df.with_columns(
        pl.when(pl.col("energy-kcal") == 0)
        .then(pl.col("energy-kj/4.184"))
        .otherwise(pl.col("energy-kcal"))
        .alias("energy-kcal-revise")
    )
    return df


def num_round(df):
    df = df.with_columns([
        pl.col(col).cast(pl.Float32)
        for col in df.select(pl.col(pl.Float64)).columns
        ])
    df = df.with_columns(
        pl.col("energy-kcal-revise").round().cast(pl.Int64)
        )
    return df


def beverage(df, drop=False):
    df = filter_cats(df, 'beverage', beverage_tag)
    df = df.with_columns(
        pl.col("categories_tags")
        .list.eval(pl.element().str.contains(r"en:beverage"))
        .list.any()
        .cast(pl.Int8)
        .alias("categories_tags_beverage")
        )
    df = df.with_columns(
        ((pl.col("is_beverage") == 1) | (pl.col("categories_tags_beverage") == 1))
        .cast(pl.Int8)
        .alias("contient_beverage")
        )
    if drop:
        df = df.filter((pl.col("contient_beverage") == 0))
        df = df.drop(["is_beverage", "categories_tags_beverage", "contient_beverage"])
    return df


def build_feats(df):
    param1 = {
    # 'df': df,
    'col_init': 'food_groups_tags',
    'col_cust': 'tag',
    'nb': 2,
    'd': CAT_CUSTOM
    }
    param2 = {
        'col_init': 'categories_tags',
        'col_cust': 'tag2',
        'nb': 7,
        'd': CAT2_CUSTOM
    }
    param3 = {
        'col_init': 'countries_tags',
        'col_cust': 'country',
        'nb': 3,
        'd': COUNTRY_GROUP
    }
    # Tags group aliment
    df = custom_tags(df, **param1)
    df = custom_tags(df, **param2)
    # Pays
    df = custom_tags(df, **param3)
    df = filter_cats(df, 'fats_nuts_seeds',
                     fats_nuts_seeds)
    df = kcal_revise(df)
    df = num_round(df)
    # df = df.with_columns(
    #     pl.col('nutriscore_grade').str.to_uppercase()
    #     )
    return df


# # c = 'tag_0'
# def encode_cat(df, c='tag_0'):
#     le = LabelEncoder()
#     df[c] = le.fit_transform(df[c])
#     # save model
#     with open(f"./model/{c}_encoder.pkl", "wb") as f:
#         pickle.dump(le, f)

#     print(dict(zip(le.classes_, le.transform(le.classes_))))
#     df = pd.get_dummies(df, columns=[c], drop_first=True)
#     return df

def encode_category(df, c='tag_0', new_data=False):
    if new_data:
        with open(f'./model/{c}_encode.txt', 'r') as f:
            cats = [line.strip() for line in f]
    else:
        cats = df[c].drop_nulls().unique().to_list()
        # save
        with open(f'./model/{c}_encode.txt', 'w') as f:
            for cat in cats:
                f.write(f"{cat}\n")

    encoded_columns = [
        (pl.col(c) == cat).cast(pl.UInt8()).alias(cat) for cat in cats
        ]

    # df_encoded = df.select(df.columns+encoded_columns)
    df_encoded = df.with_columns(encoded_columns)
    df_encoded = df_encoded.drop(c)
    return df_encoded


def pl_train_test_split(df, test_size, seed):

    n = df.height
    shuffled_idx = (
        pl.DataFrame({"idx": pl.Series("idx", list(range(n)))})
        .with_columns(pl.col("idx").shuffle(seed=seed).alias("shuf"))
        .sort("shuf")["idx"]
    )
    df = df.with_columns(
        pl.Series(name="order", values=shuffled_idx)
    )
    train = df.filter(pl.col("order") >= test_size).drop("order")
    test  = df.filter(pl.col("order") <  test_size).drop("order")
    X, y = train.drop(target), train.select(target)
    X_out, y_out = test.drop(target), test.select(target)

    return X, X_out, y, y_out


def build_dataset(to_viz=False):
    df = pl.read_parquet(file_clean)
    float_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Float64]
    df = df.with_columns([
        pl.col(float_cols).cast(pl.Float32)])

    # on garde les features
    df = df.select(feats+[target])
    cols = ['tag_0', 'tag2_0']
    for col in cols:
        if col in df.columns:
            df = encode_category(df, c=col)
            # df = encode_category(df, c='tag2_0')

    X, X_out, y, y_out = pl_train_test_split(df,
                                             test_size=0.2,
                                             seed=seed
                                             )

    X_all, y_all = df.drop(target), df.select(target)
    y = y.to_numpy().ravel()
    y_out = y_out.to_numpy().ravel()
    y_all = y_all.to_numpy().ravel()
    return X, X_out, y, y_out, X_all, y_all


def nutrisc_algo(df, c):
    df = df.with_columns(
        pl.when(pl.col('is_fats_nuts_seeds') == 1)
        .then(
            pl.when(pl.col(c) < -5).then(pl.lit('A'))
            .when(pl.col(c).is_between(-5, 2)).then(pl.lit('B'))
            .when(pl.col(c).is_between(3, 10)).then(pl.lit('C'))
            .when(pl.col(c).is_between(11, 18)).then(pl.lit('D'))
            .when(pl.col(c) >= 19).then(pl.lit('E'))
            .otherwise(None)
        )
        .otherwise(  # Si ≠ 1
            pl.when(pl.col(c) <= 0).then(pl.lit('A'))
            .when(pl.col(c).is_between(1, 2)).then(pl.lit('B'))
            .when(pl.col(c).is_between(3, 10)).then(pl.lit('C'))
            .when(pl.col(c).is_between(11, 18)).then(pl.lit('D'))
            .when(pl.col(c) >= 19).then(pl.lit('E'))
            .otherwise(None)
        )
        .alias(f'{c}_grade')
        )
    return df
