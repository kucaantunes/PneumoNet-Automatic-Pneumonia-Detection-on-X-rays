# PneumoNet-Automatic-Pneumonia-Detection-on-X-rays

Full source code at: https://drive.google.com/file/d/1bF83IhYM9-LV0vkdStr5LF6l1DctLwiI/view?usp=sharing

6.	Automatic Pneumonia Detection on X-rays


 ![image](https://github.com/user-attachments/assets/195f0fa4-15b0-4e83-8149-3a00ab3c35bf)

Figure 103. Representation of the process to detect pneumonia on X-rays, wgere the X-ray datasets are analyzed, the developed prototype is applied, XAI is used visualizing the area on which the model focuses when making predictions in the form of a heat map and the performance results obtained.










![image](https://github.com/user-attachments/assets/81b79153-843d-4e72-bd19-7c69208c6f62)

 

Non pneumonia	Pneumonia
 
 

Figure 104. Number of images of the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) and x-rays with pneumonia and without (Prashant, 2020).



 ![image](https://github.com/user-attachments/assets/9b6f4966-55f8-4395-85c3-7b3d7dea5d64)

Figure 187. Training progress of applying the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).
Figure 188 shows the confusion matrix of using the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7). A confusion matrix is a tabular representation used in deep learning and classification tasks to evaluate the performance of a predictive model. It summarizes the performance of a classification algorithm by presenting the counts of true positive, true negative, false positive, and false negative predictions for each class in a tabular format. The TP was 1096, the FP 239, the FN 170 and the TN 3179.
 
Figure 188. The confusion matrix of applying the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).
The confusion matrix shows an accuracy of 91.3%, a precision of 82.1% and a recall of 86.6% from applying the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).
Figure 189 shows the ROC curve of using the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7). The AUC was 96.62%. The AUC is a metric used to evaluate the performance of a classification model, particularly in binary classification tasks. The ROC (Receiver Operating Characteristic) curve plots the true positive rate against the false positive rate at various threshold settings. The AUC metric summarizes the performance of a binary classification model across various classification thresholds, providing a consolidated measure of its ability to distinguish between the two classes. It's a widely used metric for evaluating the overall performance of classifiers, particularly in scenarios where balanced classification is crucial.
 ![image](https://github.com/user-attachments/assets/48de0364-18d6-4cd1-b1eb-b7c597b64c78)
 ![image](https://github.com/user-attachments/assets/e1851435-3915-429c-ac86-ac34e0698108)


Figure 189. The ROC curve of applying the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).
Figure 190 shows some of the performance metrics calculated via MatLab of applying the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).

 


Figure 190. The F1-score, precision, overall precision, AUC, recall and overall recall of applying the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).

T

 
 ![image](https://github.com/user-attachments/assets/c1d2a1e1-2b53-4775-af1b-c5fbfc2d0633)




![image](https://github.com/user-attachments/assets/e0efa003-60b5-4110-86cb-236952d170a5)





![image](https://github.com/user-attachments/assets/be1b6273-ddcf-44e2-b75c-4d3bdb4e4c6d)









![image](https://github.com/user-attachments/assets/3e5b43c9-249f-4e31-aad7-f5f1ab949402)




![image](https://github.com/user-attachments/assets/66ef10a4-4128-4df0-9b1c-f0bd4d9729fd)




![image](https://github.com/user-attachments/assets/e1ea23c3-65fb-4de0-aea6-12aa43425d1f)


