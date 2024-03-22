import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_dataset(data_dir = '/content/drive/MyDrive/Face recognition/ATT images'):
    data=[]
    target=[]
    for i in range(1,41):
        images = os.listdir(f'{data_dir}/s'+str(i))
        for image in images:

            img = cv2.imread(f'{data_dir}/s'+str(i)+"/"+image,0)
            height1, width1 = img.shape[:2]
            # img_col = np.array(img, dtype='float64').flatten()
            img_col = np.array(img, dtype='float64')
            subject = int(i)
            data.append(img_col)
            target.append(subject)
    return np.array(data), np.array(target) - 1

def show_40_distinct_people(images, unique_ids):
    #Creating 4X10 subplots in  18x9 figure size
    fig, axarr=plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
    #For easy iteration flattened 4X10 subplots matrix to 40 array
    axarr=axarr.flatten()

    #iterating over user ids
    for unique_id in unique_ids:
        image_index=unique_id*10
        axarr[unique_id].imshow(images[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")

def show_10_faces_of_n_subject(images, subject_ids):
    # each subject has 10 distinct face images
    cols=10
    rows=(len(subject_ids)*10)/cols
    rows=int(rows)
    # rowsx10 dimensions
    # print('{} x {}'.format(rows, cols))

    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(18,9))
    # axarr=axarr.flatten()

    for i, subject_id in enumerate(subject_ids):
        for j in range(cols):
            image_index=subject_id*10 + j
            axarr[i,j].imshow(images[image_index], cmap="gray")
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            axarr[i,j].set_title("face id:{}".format(subject_id))

class PCA:
	# constructor
    def __init__(self, n):
        self.n = n                                      # The number of principal components to keep
        self.mean_data = None                           # The average face
        self.weights = None                             # The weights of each face
        self.eigenvalues = None                         # The eigenvalues of the covariance matrix
        self.components_ = None                # The unit eigenvectors of the covariance matrix

    def fit(self, data):

        # Find the average face
        self.mean_data = np.mean(data, axis=0)

        # Subtract all the faces with the average face to center it
        data_adj = data - self.mean_data

        # Calculate the covariance matrix
        # Computing A*A.T speeds up calculation compared to A.T*A [transpose trick]
        C = 1/(len(data_adj) - 1) * np.matmul(data_adj, data_adj.T)

        # Find the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(C)

        # Sort them in descending order
        sorted_indices = np.argsort(eigenvalues.T)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]

        # Recovering the eigenfaces
        new_eigenvectors = np.matmul(data_adj.T, eigenvectors).T[sorted_indices]

        # Normalizing the eigenfaces
        self.components_ = new_eigenvectors / np.linalg.norm(new_eigenvectors)

        # Save the weights
        self.weights = np.matmul(data_adj, self.components_.T)

    # transform the data according to the fitted model
    def transform(self, data):
        # center the data
        data_adj = data - self.mean_data

        components = self.components_[0:self.n]

        bool_arr = np.ones(len(components), dtype=bool)
        weights = np.matmul(data_adj, components[bool_arr].T)

        # return the weights that make up the face
        return weights

    # perform fitting and transforming at the same time
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    # show the mean face
    def show_mean(self):
        plt.title("Mean Image")
        plt.imshow(self.mean_data.reshape((112, 92)), cmap="gray")
        plt.show()

    # show the cumulative explained variance
    def show_explained_variance(self):
        total = sum(self.eigenvalues)
        cumulative_sum = [sum(self.eigenvalues[:i]) / total for i in range(self.n)]
        plt.plot(range(1, self.n + 1), cumulative_sum, linewidth=2)
        plt.title("Explained Variance")
        plt.xlabel("Number of Principle Components")
        plt.show()

    # show the eigenfaces
    def show_components(self):
    #   print(self.principal_components.shape)
      number_of_eigenfaces=len(self.components_)
      eigen_faces=self.components_.reshape((number_of_eigenfaces, 112, 92))[:self.n]
      cols=10
      rows=5
      fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))
      axarr=axarr.flatten()
      for i in range(50):
          axarr[i].imshow(eigen_faces[i],cmap="gray")
          axarr[i].set_xticks([])
          axarr[i].set_yticks([])
          axarr[i].set_title("eigen id:{}".format(i))
      plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))

    # show all the metrics at the same time
    def show_metrics(self):
        self.show_mean()
        self.show_explained_variance()
        self.show_components()



def performance_report(y_test, y_pred, cls):
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    precision = precision_score(y_test, y_pred, average='macro') * 100
    recall = recall_score(y_test, y_pred, average='macro') * 100
    print("#"*80)
    print(f"Evaluate metrics for {cls} classifier:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-score: {f1:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print("#"*80)
    return accuracy, f1, precision, recall
