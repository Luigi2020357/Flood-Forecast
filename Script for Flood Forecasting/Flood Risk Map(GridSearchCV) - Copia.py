import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from skimage.transform import resize
import pandas as pd
import rasterio 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from joblib import Parallel, delayed, Memory 
from sklearn.metrics import mean_squared_error, r2_score
import os 


cache_dir = 'C:/Users/dell1/Desktop/LIDAR/cache_dir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


memory = Memory(location=cache_dir, verbose=0)

# Funzione per la divisione sicura
def safe_divide(a, b):
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = 0  # Imposta a zero i valori non validi
    return result

# Caricamento dei dati pluviometrici con caching
@memory.cache 
def load_rainfall_file(file):
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file, encoding='latin1')
        except Exception as e:
            print(f"Errore nel caricamento del file {file}: {e}")
            return None
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def main():
    # Ridurre i dati pluviometrici agli ultimi 11 anni
    rainfall_files = ['2014.csv','2015.csv','2016.csv','2017.csv','2018.csv','2019.csv', '2020.csv', '2021.csv', '2022.csv', '2023.csv', '2024.csv']
    rainfall_data = Parallel(n_jobs=-1)(delayed(load_rainfall_file)(file) for file in rainfall_files)
    rainfall_data = [df for df in rainfall_data if df is not None]
    rainfall_data = pd.concat(rainfall_data).dropna()
    rainfall_mean = rainfall_data.mean().mean() 
    
    def load_lidar_data(files):
        with rasterio.open(files[0]) as src:
            dem = src.read(1)
        return dem 

    lidar_files = ['calabria_155_D38121574_0101_DSMFirst.tiff', 'calabria_155_D38121574_0101_DSMLast.tiff', 'calabria_155_D38121574_0101_DTM.tiff']
    dem = load_lidar_data(lidar_files)
    dem_reduced = resize(dem, (dem.shape[0], dem.shape[1]), anti_aliasing=True)  # Aumento della risoluzione 
    
    def load_image(file):
        return np.array(Image.open(file))

    land_use_map = load_image('Land Use map.png')
    land_use_map_reduced = resize(land_use_map, (land_use_map.shape[0], land_use_map.shape[1]), anti_aliasing=True)  # Aumento della risoluzione

    lithology_map = load_image('lithology_ULF.png')
    lithology_map_reduced = resize(lithology_map, (lithology_map.shape[0], lithology_map.shape[1]), anti_aliasing=True)  # Aumento della risoluzione

    def extract_features(dem, land_use_map, lithology_map, rainfall_mean):
        min_length = min(dem.size, land_use_map.size, lithology_map.size)
        dem_flat = dem.flatten()[:min_length]
        land_use_flat = land_use_map.flatten()[:min_length]
        lithology_flat = lithology_map.flatten()[:min_length]
        rainfall_mean_array = np.full(min_length, rainfall_mean)
        return np.column_stack((dem_flat, land_use_flat, lithology_flat, rainfall_mean_array))

    features = extract_features(dem_reduced, land_use_map_reduced, lithology_map_reduced, rainfall_mean)
    features_df = pd.DataFrame(features).dropna(axis=1, how='all')
    features = features_df.values

    imputer = SimpleImputer(strategy='mean') 
    features = imputer.fit_transform(features)
    
    if not np.isfinite(features).all():
        features = np.nan_to_num(features)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    if not np.isfinite(features).all():
        features = np.nan_to_num(features)

    pca = PCA(n_components=0.95)
    X = pca.fit_transform(features)
    
    labels = np.random.randint(0, 4, size=features.shape[0])
    if len(np.unique(labels)) < 2:
        raise ValueError("Le etichette devono contenere almeno due classi distinte.")

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    param_grid_rf = {
        'n_estimators': [100, 200, 300,],
        'max_depth': [10, 20, 30,None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid_rf,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_

    print(grid_search_rf.best_params_)
    
    def markov_chain_prediction(model, X_test):
        predictions = model.predict(X_test)
        num_classes = len(np.unique(predictions))
        
        if num_classes < 2:
            raise ValueError("Il numero di classi previste è inferiore a 2")
        
        transition_matrix = np.zeros((num_classes, num_classes))
        
        for i in range(len(predictions) - 1):
            current_state = predictions[i]
            next_state = predictions[i + 1]
            if current_state >= num_classes or next_state >= num_classes:
                continue 
            transition_matrix[current_state][next_state] += 1

        transition_matrix = safe_divide(transition_matrix, transition_matrix.sum(axis=1, keepdims=True))
        
        markov_predictions = []
        current_state = predictions[0]

        for _ in range(len(predictions)):
            next_state_probabilities = transition_matrix[current_state]
            if not np.isfinite(next_state_probabilities).all():
                next_state_probabilities = np.ones(num_classes) / num_classes
            next_state = np.random.choice(range(num_classes), p=next_state_probabilities)
            markov_predictions.append(next_state)
            current_state = next_state

        return markov_predictions, transition_matrix

    markov_predictions_rf, transition_matrix_rf = markov_chain_prediction(best_rf_model, X_test)

   
    mse_rf = mean_squared_error(y_test, markov_predictions_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, markov_predictions_rf)

    print(f'MSE RF: {mse_rf}')
    print(f'RMSE RF: {rmse_rf}')
    print(f'R2 RF: {r2_rf}')

    def generate_flood_maps(predictions, dem_shape, land_use_map):
        flood_map_size = dem_shape[0] * dem_shape[1]

        if len(predictions) < flood_map_size:
            predictions = np.pad(predictions, (0, flood_map_size - len(predictions)), mode='constant', constant_values=0)

        flood_map = np.array(predictions[:flood_map_size]).reshape(dem_shape)
        risk_levels = ['Basso', 'Moderato', 'Alto', 'Molto Alto']
        
        # Definire una mappa di colori con maggiore contrasto
        cmap = mcolors.ListedColormap(['green', 'yellow', 'orange', 'red'])
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N) 

        plt.figure(figsize=(10, 8))
        plt.imshow(land_use_map, cmap='gray', alpha=0.3)  # Aumentare la trasparenza della mappa dell'uso del suolo
        plt.imshow(flood_map, cmap=cmap, norm=norm, alpha=0.7)
        cbar = plt.colorbar(ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(risk_levels)
        plt.title('Mappa del Rischio di Inondazione Sovrapposta alla Mappa dell\'Uso del Suolo', fontsize=16)
        plt.xlabel('Longitudine', fontsize=14) 
        plt.ylabel('Latitudine', fontsize=14)
        plt.contour(flood_map, colors='black', linewidths=0.5)  # Smussamento dei contorni
        plt.savefig('flood_risk_map_with_land_use_improved.png', dpi=300)
        plt.show()

    generate_flood_maps(markov_predictions_rf, dem.shape, land_use_map_reduced)
    

    # Stampa della matrice di transizione 
    print("Matrice di transizione RF:") 
    print(transition_matrix_rf)

if __name__ == "__main__":
    main()
