import polars as pl

from src.config import CONFIG
from src.tools import clean
from src.featuring import beverage

nutrient_limit = CONFIG['nutrient_limit']
nutri = CONFIG['selected_nutrient']


def classic_clean(df):
    nutri_clean = [e for e in nutri if e not in 'energy-kj']
    df = clean(df, nutri_clean)
    df = df.drop_nulls(subset=['nutriscore_score'])
    return df


def replace_null(df):
    return df.with_columns([
        pl.col(col).fill_null(0) for col in nutri
        ])


def outliers_clean(df):
    # strict cleanup of outliers
    for col, (min_val, max_val) in nutrient_limit.items():
        df = df.filter(
            (pl.col(col) >= min_val) & (pl.col(col) <= max_val)
        )
    return df


def nutri_null(df):
    ''' Parfois tous les nutri sont = 0
    ~ 1 000 valeurs
    '''
    c = ['fat', 'saturated-fat', 'sugars', 'proteins',
         'salt', 'fiber', 'fruits-vegetables-nuts']

    df = df.with_columns(
        pl.sum_horizontal(*[pl.col(col) for col in c]).alias("sum_nutri")
    )

    df = df.filter(
        ~ ((pl.col('sum_nutri') == 0) &
           (pl.col('nutriscore_score') > 2))
           )
    return df

def process_cleanup(df):
    df = classic_clean(df)
    df = replace_null(df)
    df = outliers_clean(df)
    df = beverage(df, drop=True)
    # df = nutri_null(df)
    return df
