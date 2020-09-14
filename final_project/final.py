import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import os

class Visual_BOW():
    def __init__(self, k=20, dictionary_size=30):
        self.k = k  # number of SIFT features to extract from every image
        self.dictionary_size = dictionary_size  # size of your "visual dictionary" (k in k-means)
        self.n_tests = 10  # how many times to re-run the same algorithm (to obtain average accuracy)

    def extract_sift_features(self):
        '''
        To Do:
            - load/read the Caltech-101 dataset
            - go through all the images and extract "k" SIFT features from every image
            - divide the data into training/testing (70% of images should go to the training set, 30% to testing)
        Useful:
            k: number of SIFT features to extract from every image
        Output:
            train_features: list/array of size n_images_train x k x feature_dim
            train_labels: list/array of size n_images_train
            test_features: list/array of size n_images_test x k x feature_dim
            test_labels: list/array of size n_images_test
        '''
        
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        
        random.seed()
        
        filepath = "./101_ObjectCategories"
        
        i = 0
        for category in os.listdir(filepath)[:self.dictionary_size]: 
            if(category != ".DS_Store"):
                for jpg in os.listdir(filepath + "/" + category):



                    image_path = filepath + "/" + category + "/" + jpg
                    img = cv2.imread(image_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.k)
                    kp, des = sift.detectAndCompute(gray,None)

                    kp = kp[:self.k]
                    des = des[:self.k]

                    des_float_a = []
                    for element in des:
                        des_float_b = []
                        for d in element:
                            des_float_b.append(float(d))
                        des_float_a.append(des_float_b)


                    div = random.randint(0,9)
                    if(div <= 6):
                        train_features.append(des_float_a)
                        train_labels.append(category)
                    else:
                        test_features.append(des_float_a)
                        test_labels.append(category)
           
        
        return train_features, train_labels, test_features, test_labels
                
    
    def create_dictionary(self, features):
        '''
        To Do:
            - go through the list of features
            - flatten it to be of size (n_images x k) x feature_dim (from 3D to 2D)
            - use k-means algorithm to group features into "dictionary_size" groups
        Useful:
            dictionary_size: size of your "visual dictionary" (k in k-means)
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            kmeans: trained k-means object (algorithm trained on the flattened feature list)
        '''
        
        flat_features = []
        
        for i in range(len(features)):
            for kp in features[i]:
                flat_features.append(kp)
                
        kmeans = KMeans(n_clusters = self.dictionary_size).fit(flat_features)
        
        return kmeans

    def convert_features_using_dictionary(self, kmeans, features):
        '''
        To Do:
            - go through the list of features (images)
            - for every image go through "k" SIFT features that describes it
            - every image will be described by a single vector of length "dictionary_size"
            and every entry in that vector will indicate how many times a SIFT feature from a particular
            "visual group" (one of "dictionary_size") appears in the image. Non-appearing features are set to zeros.
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            features_new: list/array of size n_images x dictionary_size
        '''
        
        features_new = []
        
            
        for image in features:
            
            cluster_loc = []
            for i in range(self.dictionary_size):
                cluster_loc.append(0)
                
            cluster = kmeans.predict(image)
            for index in cluster:
                cluster_loc[index] = cluster_loc[index] + 1
            features_new.append(cluster_loc)
            
        return features_new

    def train_svm(self, inputs, labels):
        '''
        To Do:
            - train an SVM classifier using the data
            - return the trained object
        Input:
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
        '''
        
        clf = svm.SVC()
        clf.fit(inputs, labels)
        
        return clf

    def test_svm(self, clf, inputs, labels):
        '''
        To Do:
            - test the previously trained SVM classifier using the data
            - calculate the accuracy of your model
        Input:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            accuracy: percent of correctly predicted samples
        '''
        
        correct = 0
        
        predictions = clf.predict(inputs)
        for i in range(len(predictions)):
            if(predictions[i] == labels[i]):
                correct += 1
                
        accuracy = correct / len(labels)
        
        
        return accuracy

    def save_plot(self, features, labels):
        '''
        To Do:
            - perform PCA on your features
            - use only 2 first Principle Components to visualize the data (scatter plot)
            - color-code the data according to the ground truth label
            - save the plot
        Input:
            features: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        '''
        
        pca = PCA(n_components=2)
        PC = pca.fit_transform(features)
        
        
        plot_dict = {}
        
        for i in range(len(labels)):
            label = labels[i]
            
            if(label in plot_dict):
                plot_dict[label][0].append(PC[i][0])
                plot_dict[label][1].append(PC[i][1])
            else:
                plot_dict[label] = [[],[]]

        plt.figure(figsize=(15,8))
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        
        for label in plot_dict:
            
            color = (random.random(), random.random(), random.random())
            
            plt.scatter(plot_dict[label][0], plot_dict[label][1], c = [color])
            
            plt.savefig("results_k" + str(self.k) + "_dictSize" + str(self.dictionary_size) + ".png")
        
    
        
        
        
############################################################################
################## DO NOT MODIFY ANYTHING BELOW THIS LINE ##################
############################################################################

    def algorithm(self):
        # This is the main function used to run the program
        # DO NOT MODIFY THIS FUNCTION
        accuracy = 0.0
        for i in range(self.n_tests):
            train_features, train_labels, test_features, test_labels = self.extract_sift_features()
            kmeans = self.create_dictionary(train_features)
            train_features_new = self.convert_features_using_dictionary(kmeans, train_features)
            classifier = self.train_svm(train_features_new, train_labels)
            test_features_new = self.convert_features_using_dictionary(kmeans, test_features)
            accuracy += self.test_svm(classifier, test_features_new, test_labels)
        self.save_plot(test_features_new, test_labels)
        accuracy /= self.n_tests
        return accuracy

if __name__ == "__main__":
    alg = Visual_BOW(k=20, dictionary_size=30)
    accuracy = alg.algorithm()
    print("Final accuracy of the model is:", accuracy)