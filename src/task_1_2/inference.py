import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
def inference(image_path, test_size, stratify = True, type_pca="pca", n_components=50, is_scaler=True, using_knn=True, n_knn=5, classifier="euclidean" ):
    predict_image = cv2.imread(image_path)
    predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2GRAY)
    predict_image = cv2.resize(predict_image, (102, 92))
    predict_image = np.array([predict_image.flatten()])

    
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
    predict = model.predict(predict_image)

    # plot the best match



