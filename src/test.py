import polars as pl

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
