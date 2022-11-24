# Brain Tumor Classifier and Comparison Between Degraded and Restored versions of the Dataset.  


## Objective

This project is part of my graduation thesis whose objective was to test whether the application of [Uformer](https://github.com/ZhendongWang6/Uformer) was capable of mitigating the negative effects that MRI artifacts (Gaussian noise, Contrast, Blurring, Ringing, and Ghosting) have on the accuracy of a neural network designed to classify three types of brain tumor (meningiomas, gliomas, and pituitary tumors). This part of the project focuses on the classification and comparison parts.

For each of the five artifacts (Gaussian noise, Contrast, Blurring, Ringing, and Ghosting) and their 10 respective degradation levels, an accuracy score was produced by testing the respective dataset in a CNN created to classify MRI images into the three types of tumors mentioned above. The same is done to the restored version of these test datasets. My hypothesis was that the datasets where the images were restored using Uformer had a better accuracy than the version where they were not restored. 

## Dataset

The dataset used in this project is the [Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427). In case you are working with the _Portainer_ service from the GPDS Research group from the University of Brasília (UnB), the path to the dataset is already put into the files as "/mnt/nas/GianlucasLopes/NeuralBlack/patientImages/splits". In case you are not in this research group or for some reason the directory is no longer available, download the dataset and add its path where it is needed in the code.


## Instructions

* You must first complete the steps outlined in [Uformer-for-Artifact-Removal](https://github.com/tuliotrefzger/Uformer-for-Artifact-Removal).
* If for some reason you cannot access the dataset through the "/mnt/nas/GianlucasLopes/NeuralBlack/patientImages/splits" path, or you do not have access to the GPDS GPU server, download the dataset from the link above and replace the lines containing this old path for whatever path you decide to give to this dataset.
* Run main.py to obtain the weights and biases of the CNN.
* Run test_dataset.py to get the accuracy plot for a specific degradation. Repeat this process for each of the 5 artifacts.

## Acknowledgement

This works borrows heavily from [brain_tumor_classifier](https://github.com/gianlopes/brain_tumor_classifier).
