import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis, cityblock, correlation, minkowski
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
import argparse
from sklearn.decomposition import KernelPCA, IncrementalPCA, SparsePCA
from utils import *
import os 

def cal_distances(method, X, data):
    if method == 'euclidean':
      distances = np.linalg.norm(data - X, axis=1)

    elif method == 'angle-based':
      data = data.reshape((1, -1))
      distances = (-1) * cosine_similarity(X, data)
      distances = np.array(distances).squeeze()

    elif method == 'modified-sse':
      data = data.reshape((1, -1))
      distances = np.array([np.sum((data - x)**2) / (np.sum(x**2)*np.sum(data**2)) for x in X])

    elif method == 'mahalanobis':
      inv_cov = np.linalg.inv(X.T@X)
      distances = np.array([mahalanobis(x, data,inv_cov) for x in X])

    elif method == 'manhattan':
      distances =  np.array([cityblock(x, data) for x in X])

    elif method == 'sse':
      distances =  np.array([np.sum((x-data)**2) for x in X])

    elif method == 'chi square':
      distances =  np.array([np.sum((x-data)**2 / (x - data)) for x in X])

    elif method == 'correlation coefficient-based':
      distances =  np.array([correlation(x, data) for x in X])

    elif method == 'minkowski':
      distances =  np.array([minkowski(x, data) for x in X])
    return distances

class Distance_Classifier:
    # constructor
    def __init__(self, method):
        self.X = []     # the face vectors
        self.y = []     # the labels for those face vectors
        if method not in ['euclidean', 'angle-based', 'modified-sse', 'mahalanobis', 'manhattan', 'sse', 'chi square', 'correlation coefficient-based', 'minkowski' ]:
          raise Exception("Name of method should be in the list ['euclidean', 'angle-based', 'modified-sse', 'mahalanobis', 'manhattan', 'sse', 'chi square', 'correlation coefficient-based', 'minkowski ]")
        self.method = method

    # train the model
    def fit(self, X, y):
        self.X = X
        self.y = y

    # give the model a single face and get back a single prediction
    # if only_prediction is False, then more information will be returned as a dictionary (the index, and the distance)
    def predict_single(self, data, only_prediction=True):
        # find the distances of the sample from the training faces
        distances = cal_distances(self.method, self.X, data)

        index = np.argmin(distances)
        # print(index)
        # output the prediction based on the labels of the training faces and return a result

        prediction = self.y[index]

        if only_prediction:
            return prediction
        else:
            return {"prediction": prediction, "index": index, "distance": distances[index]}

    # predict a batch of faces at the same time, returns an array of predictions
    def predict(self, data):
        predictions = [self.predict_single(sample) for sample in data]
        return predictions

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        return accuracy_score(y_pred, y_test)

    def plot_best_match(self, X_train_origin, X_test_origin, X_test, y_test):
        incorrect_indices = []
        print('VISUALIZE ALL THE WRONGLY CLASSIFIED IMAGES')
     
        for i, data in enumerate(X_test):
            prediction_result = self.predict_single(data, only_prediction=False)
            best_match_index = prediction_result["index"]
            prediction = prediction_result["prediction"]

            if int(prediction) != int(y_test[i]):

                incorrect_indices.append(i)
                best_match_image = X_train_origin[best_match_index].reshape((112, 92))

                # Create a new figure for each subplot
                plt.figure(figsize=(10, 5))

                # Plot the original image on the left
                plt.subplot(1, 2, 1)
                plt.imshow(X_test_origin[i].reshape((112, 92)), cmap='gray')
                plt.title(f'Actual: {y_test[i]}')
                plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')

                # Plot the best match image on the right
                plt.subplot(1, 2, 2)
                plt.imshow(best_match_image, cmap='gray')
                plt.title(f'Best Match: {self.y[best_match_index]}')
                plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')

        plt.show()

class K_Nearest_Neighbors(Distance_Classifier):
      def __init__(self, method, neighbors=3):
          super().__init__(method)
          self.neighbors = neighbors

      def predict_single(self, data, only_prediction=True):
          # find the distances of the sample from the training faces
          distances = cal_distances(self.method, self.X, data)

          # find the index of the smallest distance
          sorted = np.argsort(distances)
          index = sorted[:self.neighbors]
          neighbors = np.array(self.y)[sorted][:self.neighbors]
          if len(np.unique(neighbors)) == len(neighbors):
            index = index[0]
            prediction = neighbors[0]
          else:
            prediction, counts = mode(neighbors)
            index = index[np.where(neighbors == prediction)[0][0]]

          if only_prediction:
              return prediction
          else:
              return {"prediction": prediction, "index": index, "distance": distances[index]}

def evaluate_similarity(data_dir, test_size, stratify = True, type_pca="pca", n_components=50, is_scaler=True, is_show_metrics=True, using_knn=True, n_knn=5, classifier="euclidean"):
    
    X, y = load_dataset(data_dir)
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    predictions = {"pred":[], "true":[]}
    if stratify:
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, stratify=y, random_state=1511)
    else:
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=1511)

    if is_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

  
    if type_pca == "pca":
        pca = PCA(n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    elif type_pca == "kernel":
        pca = KernelPCA(n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    elif type_pca == "incremental":
        pca = IncrementalPCA(n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    elif type_pca == "sparse":
        pca = SparsePCA(n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    else: 
      raise Exception("Name of PCA should be in the list ['pca', 'kernel', 'incremental', 'sparse']")

    if using_knn:
        model = K_Nearest_Neighbors(method=classifier, neighbors=n_knn)
    else:
        model = Distance_Classifier(classifier)


    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    predictions["pred"].extend(y_pred)
    predictions["true"].extend(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Average accuracy for {classifier} classifier: {accuracy}%")
    accuracy, f1, precision, recall = performance_report(y_test, y_pred, classifier)

    if is_show_metrics:
        model.plot_best_match(X_train, X_test, X_test_pca, y_test)
        ax = plt.axes()
        plt.suptitle(f"Confusion matrix for PCA / {classifier}")
        plt.title(f"Accuracy: {accuracy*100}%", fontsize=10)
        ConfusionMatrixDisplay.from_predictions(
            predictions["true"],
            predictions["pred"],
            xticks_rotation="vertical",
            include_values=False,
            normalize="true",
            display_labels=["" for i in range(len(set(predictions["true"])))],
            ax=ax)
        ax.tick_params(axis='both', which='both', length=0)
        plt.show()
    return accuracy, f1, precision, recall
def report_options(data_dir, test_size, stratify = True, type_pca="pca", n_components=50, is_scaler=True, is_show_metrics=True, using_knn=True, n_knn=5): 
    print(f'''Split the data into train and test with {test_size} as test size. Choose whether to stratify or not: {stratify}
            Choose the type of PCA: {type_pca}, number of components: {n_components}, whether to scale the data or not: {is_scaler}, 
            whether to show performance metrics or not: {is_show_metrics}, using KNN or not: {using_knn}, number of neighbors (if using knn): {n_knn}
            ''')
    metrics =  ['euclidean', 'angle-based', 'modified-sse', 'mahalanobis', 'manhattan', 'sse', 'chi square', 'correlation coefficient-based', 'minkowski']
    # Create a dataframe for index is metrics and columns are classifiers
    df = pd.DataFrame(columns=metrics, index=['accuracy', 'precision', 'recall', 'f1'])
    for metric in metrics:
        # evaluate_models(data_dir, test_size, stratify, type_pca, n_components, is_scaler, is_show_metrics, using_knn, n_knn, metric)
        accuracy, f1, precision, recall = evaluate_similarity(data_dir, test_size, stratify, type_pca, n_components, is_scaler, is_show_metrics, using_knn, n_knn, metric)  
        df[metric] = [accuracy, precision, recall, f1]
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate models with various parameters.")
    parser.add_argument('--data_dir', type=str, default=r"G:\My Drive\Data Mining\Project 1\ATT images", help='Path to the dataset')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test size for train_test_split')
    parser.add_argument('--stratify', type=bool, default=True, help='Whether to stratify split or not')
    parser.add_argument('--type_pca', type=str, default='kernel', choices=['pca', 'kernel', 'incremental', 'sparse'], help='Type of PCA')
    parser.add_argument('--n_components', type=int, default=50, help='Number of components for PCA')
    parser.add_argument('--is_scaler', type=bool, default=False, help='Whether to scale the data or not')
    parser.add_argument('--is_show_metrics', type=bool, default=False, help='Whether to show performance metrics or not')
    parser.add_argument('--using_knn', type=bool, default=True, help='Whether to use KNN or not')
    parser.add_argument('--n_knn', type=int, default=3, help='Number of neighbors for KNN')
    # parser.add_argument('--classifier', type=str, default=classifier, choices=['euclidean', 'angle-based', 'modified-sse', 'mahalanobis', 'manhattan', 'sse', 'chi square', 'correlation coefficient-based', 'minkowski'], help='Classifier type')
    
    args = parser.parse_args()
    data_dir = args.data_dir
    test_size = args.test_size
    stratify = args.stratify
    type_pca = args.type_pca
    n_components = args.n_components
    is_scaler = args.is_scaler
    is_show_metrics = args.is_show_metrics
    using_knn = args.using_knn
    n_knn = args.n_knn
    # classifier = args.classifier
    report_options(data_dir, test_size, stratify, type_pca, n_components, is_scaler, is_show_metrics, using_knn, n_knn)
# def main(data_dir = )
