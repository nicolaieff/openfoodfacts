import polars as pl
from tqdm import tqdm

from src.config import CONFIG

out_folder = CONFIG['folder_output']

def clean(df: pl.DataFrame,
          cols: list
          ) -> pl.DataFrame:
    # -> supprime ttes les lignes avec au moins 1 val manquante
    # df = df.drop_nulls()  +subset=[]
    # -> Supprimer les lignes où toutes les [cols] sont null
    df = df.filter(
            ~pl.fold(acc=pl.lit(True),
                    function=lambda acc, s: acc & s.is_null(), 
                    exprs=[pl.col(c) for c in cols])  # pl.all()
                    )
    # -> Supprime les doublons exacte
    df = df.unique()
    return df


def load_data(path_parque):
     dfs = [pl.read_parquet(f) for f in tqdm(path_parque)]
     df = pl.concat(dfs, how='vertical')
     return df


def anomaly_counts(df: pl.DataFrame, nutri_limit: dict) -> pl.DataFrame:
    anomalies = []

    for col, (min_val, max_val) in nutri_limit.items():
        too_low = df.filter(pl.col(col) < min_val).height
        too_high = df.filter(pl.col(col) > max_val).height
        total_anomalies = too_low + too_high

        anomalies.append({
            "feature": col,
            "too_low": too_low,
            "too_high": too_high,
            "total": total_anomalies
        })

    return pl.DataFrame(anomalies)


def val_cnt(df, c, save=False):
    df_stat = df.select(pl.col(c).value_counts()
                        ).unnest(c
                                 ).sort("count").with_columns(
            (pl.col("count") / df.height * 100).round(2).alias("percentage"
                                                               )
        )
    # stats = df.select(pl.col(c)).describe()
    if save:
        df_stat.write_csv(f"{out_folder}/stat_{c}.txt")
    return df_stat


def len_list(df, c):
    """
    Calcule les longueurs des listes dans une colonne `list[str]`
    et retourne les value counts de ces longueurs.
    """
    len_col = f"{c}_len"

    return (
        df.with_columns(
            pl.col(c).list.len().alias(len_col)
        )
        .select(pl.col(len_col).value_counts())
        .unnest(len_col)
        .sort(len_col)
    )


# df.filter(
#     (pl.col("food_groups_tags").is_not_null()) & 
#     (pl.col("food_groups_tags").list.len() > e)
# )

def compare_df(name1, name2):
    df1 = pl.read_parquet(name1)
    df2 = pl.read_parquet(name2)

    same_columns = df1.columns == df2.columns
    same_shape = df1.shape == df2.shape
    df1_sorted = df1.sort(df1.columns)
    df2_sorted = df2.sort(df2.columns)
    same_data = df1_sorted.rows() == df2_sorted.rows()

    are_equal = same_columns and same_shape and same_data

    if are_equal:
        print("✅ same file")
    else:
        print("❌ not same file")


