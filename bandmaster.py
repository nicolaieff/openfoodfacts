import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path

from src.config import CONFIG
from src.processdata import process_row_group
from src.tools import clean, load_data
from src.cleanup import process_cleanup
from src.model import model_knn, model_tree, model_lgb
from src.featuring import build_feats, build_dataset
from src.optiparams import optuna_search_lgb

folder_process = CONFIG['folder_process']
file_clean = CONFIG['file_clean']
nutri = CONFIG['selected_nutrient']
seed = CONFIG['seed']
knn_param_grid = CONFIG['knn_gridsrch']

def extract_data():
    data_path = CONFIG['data_path']
    variables = CONFIG['selected_var']

    parquet_file = pq.ParquetFile(data_path)
    
    batch_group = []
    i_file = 0

    for i in tqdm(range(parquet_file.num_row_groups)):
        df = process_row_group(parquet_file, i, variables, nutri)
        batch_group.append(df)

        if (i + 1) % 1000 == 0:
            df_concat = pl.concat(batch_group)
            df_concat = clean(df_concat, nutri)
            df_concat.write_parquet(f'{folder_process}/food_transforme_{i_file}.parquet')
            batch_group = []
            i_file += 1

    if batch_group:
        df_concat = pl.concat(batch_group)
        df_concat = clean(df_concat, nutri)
        df_concat.write_parquet(f'{folder_process}/food_transforme_{i_file}.parquet')

def cleanup_data(path_parque):
    df = load_data(path_parque)
    print('Before clean : ', df.shape)
    df = process_cleanup(df)
    print('After clean : ', df.shape)
    df.write_parquet(file_clean)


def compute_model(X, y):
    res_tree = model_tree(X, y)
    print('TREE ', res_tree)

    res_knn = model_knn(X, y)
    print('KNN ', res_knn)

    res_lgb = model_lgb(X, y)
    print('LGBM ', res_lgb)

    return {
        'TREE': res_tree,
        'KNN': res_knn,
        'LGBM': res_lgb
    }


def main():
    # ETAPE A : create food_transforme_*.parquet
    folder = Path(folder_process)
    files = list(folder.glob('food_transforme_*.parquet'))
    if files:
        print(f'A. ✅ {len(files)} file(s) found')
    else:
        print('A. Extract data with selected feature and nutrient...')
        extract_data()

    # ETAPE B : create nutrient_cleaned.parquet
    clean_file = list(Path().glob(file_clean))
    if clean_file:
        print('B. 🧹 file clean')
    else:
        print('B. Cleaning...')
        files = list(folder.glob('food_transforme_*.parquet'))
        cleanup_data(files)

    # ETAPE C : add features à nutrient_cleaned.parquet
    schema = pl.read_parquet_schema(file_clean)
    if 'tag_0' in list(schema.keys()):
        print('C. 🥐 feature ok')
    else:
        df = pl.read_parquet(file_clean)
        df = build_feats(df)
        df.write_parquet(file_clean)
        print('C. Feature added...')


    # ETAPE D : build dataset
    X, X_out, y, y_out, X_all, y_all = build_dataset()
    print('Data size ', X_all.shape)
    print('Trainset size : ', X.shape)
    print('Trainset features : ', X.columns)

    # # ETAPE E : model
    print('E. 🏓 Compare model...')
    if False:
        res = compute_model(X, y)

    # ETAPE F : recherche de meilleurs parametres
    try:
        lgb_best_params = CONFIG['lgb_best_params']
        print('F. 🤖 best params find')
    except:
        lgb_best_params = optuna_search_lgb(X, y, X_out, y_out)
        print('F. LGBM parameter research...', lgb_best_params)
        # last : {'cv_sc': array([0.923, 0.923]), 'sc_mean': np.float64(0.923)}

    # ETAPE G : save model
    if True:
        print('G.', model_lgb(X_all, y_all, lgb_best_params, save=True))

    # analyse des erreurs du modèle

if __name__ == '__main__':
    main()