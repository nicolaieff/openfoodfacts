import polars as pl


def read_row_group(parquet_file, 
                   group_index: int, 
                   colonnes: list
                   ) -> pl.DataFrame:
    table = parquet_file.read_row_group(group_index, 
                                        columns=colonnes)
    return pl.from_arrow(table)


def extract_nutrient_value(nutri_list, 
                           key):
    # stop when find nutrient
    if not nutri_list:
        return None
    nutrient = next(
        (nut for nut in nutri_list 
         if key.lower() in nut.get("name", "").lower()),
        None
        )
    return nutrient.get("value") if nutrient else None


def add_nutrients(df: pl.DataFrame, 
                  nutri_keys: list
                  ) -> pl.DataFrame:
    return df.with_columns([
        pl.struct(["nutriments"])
        .map_elements(lambda x, key=key: extract_nutrient_value(x["nutriments"], key),
                      return_dtype=pl.Float64
                      ).alias(key)
        for key in nutri_keys
    ]).drop("nutriments")


def extract_product_name(df: pl.DataFrame
                         ) -> pl.DataFrame:
    return df.with_columns([
        pl.col("product_name").list.eval(
            pl.element().struct.field("text")
        ).list.first()
    ])


def process_row_group(parquet_file, 
                      group_index: int, 
                      colonnes: list, 
                      nutri_keys: list) -> pl.DataFrame:
    df = read_row_group(parquet_file, group_index, colonnes)
    df = add_nutrients(df, nutri_keys)
    df = extract_product_name(df)
    return df

