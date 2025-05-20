import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import *
import fastnanquantile as fnq
from itertools import product
import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from user_encoding_lib import *
from utils import *


def get_data(data_path, metadata_path):
    df = pd.read_csv(data_path)
    metadata = pd.read_csv(metadata_path, delimiter=";")

    num_users = df["ID"].unique().size
    num_days = df.shape[0]//num_users

    X = df.values[:,:-2].astype(float)
    X[X<0] = 0.0
    X[X>0] = np.clip(X[X>0], 1e-3, np.inf)
    X_user = np.reshape(X, (num_users, -1))
    return X_user, metadata, num_users, num_days

def preprocess_data(X_user, metadata):
    X_user_scaled = (X_user)/metadata["estimated_ac_capacity"].values[:,None]*1000
    return X_user_scaled

def main():
    
    data_folder = "data/"
    data_file = "X_daily_15min_example.csv"
    results_folder = "results/"

    RANDOM_SEED = 2112
    NUMBER_OF_CLUSTERS = [11, 12, 13, 14, 15]
    NUMBER_OF_LDA_TOPICS = [4, 5, 6, 7, 8, 9, 10]
    NUMBER_OF_LDA_CLUSTERS = [100, 250, 500, 1000]
    NUMBER_OF_LOWER_DIMS = [96]
    DISTANCE_MEASURES = ["Symmetric KL Div", "Bhattacharyya"]
    LINKAGE = ["average", "complete"]
    quantiles = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95]

    data_path = os.path.join(data_folder, data_file)
    metadata_path = os.path.join(data_folder, "metadata.csv")

    print(f"Loading data from {data_path}...")
    X_user, metadata, num_users, num_days = get_data(data_path, metadata_path)
    X_train = preprocess_data(X_user, metadata).reshape(num_users, num_days, -1)

    n_rows = len(quantiles)*num_users * (1 + len(NUMBER_OF_CLUSTERS) * (1 + len(NUMBER_OF_LDA_TOPICS)*len(NUMBER_OF_LDA_CLUSTERS)*len(NUMBER_OF_LOWER_DIMS)*len(DISTANCE_MEASURES)*len(LINKAGE)))
    results_dict = {
                        "Number of Clusters":      np.empty(n_rows, dtype=int),
                        "Number of Topics":        np.empty(n_rows, dtype=int),
                        "Wording Granularity":     np.empty(n_rows, dtype=int),
                        "Number of Lower Dims":    np.empty(n_rows, dtype=int),
                        "Distance Measure":        np.empty(n_rows, dtype=object),  # strings → object
                        "Linkage":                 np.empty(n_rows, dtype=object),
                        "Method":                  np.empty(n_rows, dtype=object),
                        "Metric":                  np.empty(n_rows, dtype=object),
                        "Quantile":                np.empty(n_rows, dtype=float),
                        "ID":                      np.empty(n_rows, dtype=int),
                        "Value":                   np.empty(n_rows, dtype=float),
                    }
    acc = 0

    print("Getting the results of the pooled baseline...")
    # quantile_levels_pooled = np.zeros((len(quantiles), num_users, *X_train.shape[-2:]))
    quantile_levels_pooled = fnq.nanquantile(X_train, quantiles, axis=0)[:,None,...]
    quantile_levels_pooled = np.repeat(quantile_levels_pooled, num_users, axis=1)
    
    _, quantile_losses = calculate_quantile_loss(targets=X_train, quantile_predictions=quantile_levels_pooled, quantiles=quantiles, nonzero=True)

    for i, q in enumerate(quantiles): 
        for u in range(num_users): 
            results_dict["Number of Clusters"][acc] = 1
            results_dict["Number of Topics"][acc] = -1
            results_dict["Wording Granularity"][acc] = -1
            results_dict["Number of Lower Dims"][acc] = -1
            results_dict["Distance Measure"][acc] = "N/A"
            results_dict["Linkage"][acc] = "N/A"
            results_dict["Method"][acc] = "Pooled"
            results_dict["Metric"][acc] = "Quantile Loss"
            results_dict["Quantile"][acc] = q
            results_dict["ID"][acc] = u
            results_dict["Value"][acc] = np.nanmean(quantile_losses[i,u])
            acc += 1

    pbar_phy = tqdm.tqdm(total=len(NUMBER_OF_CLUSTERS), dynamic_ncols=True)

    for num_clusters in NUMBER_OF_CLUSTERS:
        pbar_phy.write(f"Getting the results of the physics-based clustering with {num_clusters} clusters...")
        phy_features = metadata[["tilt", "azimuth"]].values
        kmeans_phy = KMeans(n_clusters=num_clusters, random_state=RANDOM_SEED).fit(phy_features)
        labels_phy = kmeans_phy.labels_

        # Check for clusters with fewer than 2 members
        invalid_clusters = [i for i in np.unique(labels_phy) if np.sum(labels_phy == i) < 2]
        if invalid_clusters:
            pbar_phy.write(f"Invalid setting: Clusters with fewer than 2 members found: {invalid_clusters}")
            continue

        quantile_levels_phy_cluster = np.zeros((len(quantiles), num_clusters, *X_train.shape[-2:]))
        quantile_levels_phy_user = np.zeros((len(quantiles), num_users, *X_train.shape[-2:]))
        
        for c in range(num_clusters):
            cluster_members = np.where(labels_phy==c)[0]
            quantile_levels_phy_cluster[:,c] = fnq.nanquantile(X_train[cluster_members], quantiles, axis=0)
        
        for u in range(num_users): quantile_levels_phy_user[:,u] = quantile_levels_phy_cluster[:,labels_phy[u]]
        
        _, quantile_losses = calculate_quantile_loss(targets=X_train, quantile_predictions=quantile_levels_phy_user, quantiles=quantiles, nonzero=True)

        for i, q in enumerate(quantiles): 
            for u in range(num_users): 
                results_dict["Number of Clusters"][acc] = num_clusters
                results_dict["Number of Topics"][acc] = -1
                results_dict["Wording Granularity"][acc] = -1
                results_dict["Number of Lower Dims"][acc] = -1
                results_dict["Distance Measure"][acc] = "N/A"
                results_dict["Linkage"][acc] = "N/A"
                results_dict["Method"][acc] = "Physics-based"
                results_dict["Metric"][acc] = "Quantile Loss"
                results_dict["Quantile"][acc] = q
                results_dict["ID"][acc] = u
                results_dict["Value"][acc] = np.nanmean(quantile_losses[i,u])
                acc += 1
        pbar_phy.update(1)
    pbar_phy.close()

    print("Getting the results of the entity-based clustering...")

    pbar_lda = tqdm.tqdm(total=len(NUMBER_OF_LDA_TOPICS)*len(NUMBER_OF_LDA_CLUSTERS)*len(NUMBER_OF_LOWER_DIMS)*len(DISTANCE_MEASURES)*len(LINKAGE), dynamic_ncols=True)
    pbar_clusters = tqdm.tqdm(total=len(NUMBER_OF_CLUSTERS), dynamic_ncols=True)
                
    for num_topics, num_lda_clusters, num_lower_dims, distance_measure_name, linkage in product(NUMBER_OF_LDA_TOPICS, NUMBER_OF_LDA_CLUSTERS, NUMBER_OF_LOWER_DIMS, DISTANCE_MEASURES, LINKAGE):
        pbar_lda.write(f"Experimenting with {num_topics} topics, {num_lda_clusters} clusters, {num_lower_dims} lower dims, {distance_measure_name} distance measure and {linkage} linkage...")

        if num_lower_dims == X_train.shape[-1]: entity_model = UserEncoder(num_topics=num_topics, num_clusters=num_lda_clusters, random_state=RANDOM_SEED, reduce_dim=False)
        else: entity_model = UserEncoder(num_topics=num_topics, num_clusters=num_lda_clusters, random_state=RANDOM_SEED, reduce_dim=True, num_lower_dims=num_lower_dims)

        pbar_lda.write(f"Fitting the entity model...")
        entity_model.fit(X_train,
                    fit_kwargs={"lda": {
                                "perp_tol": 0.1,
                                "max_iter": 1000,
                                "batch_size": 64,
                                "verbose": False,
                                "learning_method": "online",
                                "evaluate_every": 5,
                                "n_jobs": None,
                                "doc_topic_prior": 1/entity_model.num_topics,
                                "topic_word_prior": 1/entity_model.num_clusters,
                            }})

        pbar_lda.write(f"Transforming the data...")
        gamma_matrix = entity_model.transform(X_train)

        if distance_measure_name == "Symmetric KL Div": distance_measure = sym_kl_div_dirichlet
        elif distance_measure_name == "Bhattacharyya": distance_measure = bhattacharyya_distance_dirichlet
        else: raise ValueError(f"Distance measure {distance_measure_name} not supported.")

        distances = distance_measure(gamma_matrix[:,:,None], gamma_matrix.T[None,:,:])

        for num_clusters in NUMBER_OF_CLUSTERS:
            pbar_lda.write(f"Clustering the data with {num_clusters} clusters...")
            clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", linkage=linkage).fit(distances)
            labels = clustering.labels_

            # Check for clusters with fewer than 2 members
            invalid_clusters = [i for i in np.unique(labels) if np.sum(labels == i) < 2]
            if invalid_clusters:
                pbar_lda.write(f"Invalid setting: Clusters with fewer than 2 members found: {invalid_clusters}")
                continue
            
            pbar_clusters.write(f"Getting the quantile results...")
            quantile_levels_entity_cluster = np.zeros((len(quantiles), num_clusters, *X_train.shape[-2:]))
            quantile_levels_entity_user = np.zeros((len(quantiles), num_users, *X_train.shape[-2:]))
            
            for c in range(num_clusters):
                cluster_members = np.where(labels==c)[0]
                quantile_levels_entity_cluster[:,c] = fnq.nanquantile(X_train[cluster_members], quantiles, axis=0)
            for u in range(num_users): quantile_levels_entity_user[:,u] = quantile_levels_entity_cluster[:,labels[u]]
            
            _, quantile_losses = calculate_quantile_loss(targets=X_train, quantile_predictions=quantile_levels_entity_user, quantiles=quantiles, nonzero=True)

            for i, q in enumerate(quantiles): 
                for u in range(num_users):
                    results_dict["Number of Clusters"][acc] = num_clusters
                    results_dict["Number of Topics"][acc] = num_topics
                    results_dict["Wording Granularity"][acc] = num_lda_clusters
                    results_dict["Number of Lower Dims"][acc] = num_lower_dims
                    results_dict["Distance Measure"][acc] = distance_measure_name
                    results_dict["Linkage"][acc] = linkage
                    results_dict["Method"][acc] = "Entity"
                    results_dict["Metric"][acc] = "Quantile Loss"
                    results_dict["Quantile"][acc] = q
                    results_dict["ID"][acc] = u
                    results_dict["Value"][acc] = np.nanmean(quantile_losses[i,u])
                    acc += 1
            
            pbar_clusters.update(1)
        pbar_clusters.reset()
        pbar_lda.update(1)
            
    pbar_lda.close()
    pbar_clusters.close()
    print("Saving the results...")
    results_path = os.path.join(results_folder, "results_vanilla.csv")
    if not os.path.exists(results_folder): os.makedirs(results_folder)
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
