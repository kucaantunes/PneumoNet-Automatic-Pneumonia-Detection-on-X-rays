# PneumoNet-Automatic-Pneumonia-Detection-on-X-rays

6.	Automatic Pneumonia Detection on X-rays

Concerning the detection of pneumonia on X-rays, the research used three different public datasets the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9), in order to compare the results obtained with the work of other authors. 
The prototype used is based on changing some layers of the AlexNet architecture, in order to increase the accuracy of the detection.
Concerning the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) several models were applied to detect pneumonia, namely the ResNet-50, AlexNet and the prototype.
This investigation shows a new approach for detecting pneumonia, the datasets were divided in groups of training, test and validation.
The training dataset was used to train the deep learning model. It consists of input data (features) and their corresponding correct output (labels or targets). During training, the model learned to map inputs to outputs by adjusting its parameters through optimization algorithms (like gradient descent) to minimize the difference between predicted and actual outputs.
The validation dataset was used to fine-tune the model's hyperparameters and to provide an unbiased evaluation of the models fit on the training dataset. It helps in monitoring the model's performance during training to prevent overfitting or underfitting. Hyperparameters (like learning rate, batch size, etc.) were adjusted based on validation performance.
Once the model was trained and tuned using the training and validation datasets, it was evaluated on the testing dataset. This dataset is separate from the training and validation datasets and was used to assess how well the model generalizes to new, unseen data. It helped to estimate the model's performance in a real-world scenario.
Splitting the dataset into these three parts (training, validation, and testing) is crucial to ensure that the model learns effectively, generalizes well to new data, and doesn't overfit by memorizing the training data's specifics.
In order to make the deep learning models more understandable and interpretable to humans, XAI was used to provide insights into how AI systems make decisions or predictions. This investigation used XAI and Grad-Cam in order to make the predictions more understandable.
Related work was analyzed in order to be possible to compare the developed model with the work of other researchers, many models can be used to detect pneumonia on X-rays.
In order to detect pneumonia on X-ray images using deep learning, the data was collected and prepared, three different large datasets of X-ray images containing both normal and pneumonia-affected cases was collected. These datasets were labeled accurately to indicate which images depict pneumonia and which were normal.
The X-ray images were preprocessed to standardize their size, resolution, and format. This step involved resizing images, adjusting contrast, normalizing pixel values, and removing noise to enhance the quality of input data.
The datasets were divided into training, validation, and testing sets. The CNN models were trained on the training dataset by feeding it batches of X-ray images and their corresponding labels. The model learned to identify patterns and features that distinguish between normal and pneumonia-affected X-rays.
The validation sets allowed to fine-tune the model's hyperparameters (example given, learning rate, batch size, number of layers) and prevent overfitting. This step involved adjusting the model to achieve better performance on unseen data.
The trained model was evaluated on the separate testing dataset to assess its performance. Metrics such as accuracy, precision, recall, and F1-score weree calculated to measure how well the model classifies pneumonia and normal X-rays.
Explainable AI techniques were applied to understand why the model made specific predictions. This step helped in providing insights into which regions or features in the X-ray images contributed most to the model's decision.
The trained models were deployed via web by using Flask, where the user can via interface upload an X-ray and the system will present its prediction mentioning if the medical image has pneumonia or not.
Two web applications were developed, one in C# .Net with SQL server and another in PHP where a clinical record of each patient is shown mentioning the obtained results.
Throughout this process, it was crucial to have a sizable and diverse dataset, proper validation techniques, and rigorous evaluation methods to ensure the model's accuracy, generalizability, and reliability in detecting pneumonia on X-ray images.
Figure 80 Illustrates the pneumonia detection process via X-ray analysis. Analyzing X-ray datasets, applying the developed prototype, utilizing XAI to visualize the model's focused areas during predictions through a heat map, and presenting the achieved performance results.
 ![image](https://github.com/user-attachments/assets/195f0fa4-15b0-4e83-8149-3a00ab3c35bf)

Figure 103. Representation of the process to detect pneumonia on X-rays, wgere the X-ray datasets are analyzed, the developed prototype is applied, XAI is used visualizing the area on which the model focuses when making predictions in the form of a heat map and the performance results obtained.






6.1	Datasets Used
Concerning the pneumonia detection, a set of medical images namely x-rays were used with the symptom and without the symptom.
Three different datasets were used in this investigation. Datasets play a pivotal role in training deep learning models for detecting pneumonia on X-ray images. High-quality and diverse datasets are essential for training accurate and robust deep learning models. The used datasets contain a wide variety of pneumonia cases, including different stages, variations, and other lung conditions, enabling the models to learn and generalize effectively. 
An expansive dataset aids in the generalization of the model. It ensures that the model doesn't merely memorize specific features of the training data but learns meaningful patterns and features representative of pneumonia across a broad spectrum. This generalization enables the model to perform well on unseen X-ray images and real-world scenarios.
The datasets have balanced representations of different classes (e.g., normal vs. pneumonia-affected X-rays) helping to mitigate biases and prevent overfitting. Biases can occur when a dataset is skewed toward one class, leading the model to perform poorly on underrepresented classes. 
The datasets were divided into training, validation, and testing subsets allowing a rigorous assessment of the model's performance. A separate testing set that the model has never seen before helps in accurately measuring its ability to generalize to new, unseen data.
Datasets are fundamental in training accurate, reliable, and unbiased deep learning models for detecting pneumonia on X-ray images. They serve as the foundation upon which the model learns to make informed and precise classifications, significantly impacting the model's performance.
The Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) contains X-rays with and without pneumonia.
Figure 80 shows the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) via MatLab, containing 3418 X-rays with pneumonia and 1266 normal.





![image](https://github.com/user-attachments/assets/81b79153-843d-4e72-bd19-7c69208c6f62)

 

Non pneumonia	Pneumonia
 
 

Figure 104. Number of images of the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) and x-rays with pneumonia and without (Prashant, 2020).

The Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) was also used to detect pneumonia on X-rays in the work of (Lasker et al., 2022) and (Jain et al., 2020). The dataset was collected from Kaggle repositories.
The labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) was obtained from (Kermany, 2018) that was compiled by the Guangzhou Women and Children Medical Center (Guangzhou, China) as part of the routine clinical care of pediatric patients. The latest version of this dataset is composed of 5856 X-rays images. It was divided into a training set consisting of 3883 X-rays corresponding to cases of pneumonia and 1349 X-rays without detected pathologies, and a test set with 234 images labelled as pneumonia and 390 without detected pathologies (Ortiz-Toro, 2022). Figure 81 shows some of the X-rays used on the training dataset as well as the number of medical images used with and without pneumonia.
 
Figure 105. Number of images of the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and x-rays with pneumonia and without (Kermany, 2018).

The labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) was also used for pneumonia detection on the work of (Saboo et al., 2023), (Kusk, & Lysdahlgaard, 2023) and (Ortiz-Toro, 2022).
The chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9), on the training dataset there are 1341 images as normal and 3875 X-rays with pneumonia, including validation and test there are in total 5856.
Figure 82 displays some of the images of the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) as well as the number of images used for training.
The dataset comprises three main folders (train, test, validation) with subfolders for each image category (Pneumonia/Normal), totaling 5,863 JPEG X-ray images across two categories. These chest X-ray images (anterior-posterior) were sourced from pediatric patients aged one to five at Guangzhou Women and Children’s Medical Center as part of routine clinical care.
To ensure quality, all images underwent an initial screening to remove low-quality or unreadable scans. Subsequently, two expert physicians graded the diagnoses before inclusion for AI system training. Additionally, a third expert reviewed the evaluation set to mitigate any grading errors.
To ensure the accuracy of the chest x-ray image analysis, an initial quality control step involved screening all chest radiographs to eliminate any scans deemed low quality or unreadable. Subsequently, two expert physicians assessed and graded the diagnoses of these images before their utilization in training the AI system. To further mitigate potential grading errors, a third expert reviewed the evaluation set as well.
 
Figure 106. Number of images of the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) and x-rays with pneumonia and without (Kermany, 2018) and (Mooney, 2018).

The chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) was also used to detect pneumonia on the works of (Mabrouk et al., 2022) and (Kundu et al. 2021).
The use of public datasets in deep learning for pneumonia detection from X-rays has significantly advanced the field of medical imaging and diagnostics. 
These datasets contain a large number of chest X-ray images annotated with labels indicating the presence or absence of pneumonia. The datasets were leveraged to train deep learning models, particularly convolutional neural networks, using techniques such as designing models from scratch. The goal was to create algorithms capable of accurately identifying patterns associated with pneumonia on X-ray images. Public datasets offer a wealth of labeled data that can be accessed, irrespective of their location or resources. This accessibility democratizes research and encourages collaboration in the field.
Standardized datasets provide a common ground for benchmarking different algorithms, allowing to compare the performance of the models against existing state-of-the-art methods, fostering innovation and improvement.
Training deep learning models on diverse datasets helps in building more generalized models. Exposure to varied data distributions and imaging conditions helps models perform better on unseen data and real-world scenarios.
Using publicly available datasets helps in the process of adhering to ethical guidelines and data protection regulations by avoiding potential issues related to patient data privacy. The used public datasets used were treated taking in consideration biases or inconsistencies in annotations. Activities were performed to ensure data quality and to address biases. Some publicly available datasets might have limited annotations or imbalanced classes, which can affect the model's ability to learn effectively.
Leveraging public datasets remains crucial in advancing the development of deep learning models for pneumonia detection from X-ray images. Continued efforts to improve dataset quality, address biases, and facilitate model generalization are essential for furthering the efficacy and reliability of the AI-driven diagnostic tools.
This section intends answer #Research_Question_2 and presenting #Hypothesis_3.







6.2	Pneumonia Detection on X-rays Using the Residual Neural Network with 50 Layers
Pneumonia detection through X-ray images using a Residual Neural Network (ResNet) with 50 layers has been a significant advancement in medical imaging and diagnosis. ResNet, developed by (He et al., 2015), introduced a deeper architecture that addressed the vanishing gradient problem encountered in training very deep neural networks.
When applied to pneumonia detection from X-ray images, a ResNet-50 architecture proves to be effective due to its depth and ability to capture intricate patterns within the images. ResNet-50 is composed of 50 layers of neural network units. These units learn to identify features at various levels of abstraction within the X-ray images. Each layer extracts specific patterns and passes the information forward to subsequent layers.
The key innovation in ResNet is the inclusion of skip connections or shortcuts, known as residual connections. These connections allow the network to bypass one or more layers, enabling the direct flow of information from earlier layers to deeper layers. This helps in mitigating the vanishing gradient problem and aids in the training of deeper networks.
As the X-ray images contain crucial visual cues indicative of pneumonia, the ResNet-50 network can efficiently learn and extract these features. The initial layers detect simpler features like edges and textures, while deeper layers progressively discern more complex patterns, such as consolidations or infiltrates in the lung fields, which might indicate pneumonia.
The network was trained using the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7) with images labeled as pneumonia and normal to indicate the presence or absence of pneumonia. The model learns to differentiate between normal and abnormal X-rays by adjusting its parameters during training to minimize the classification error.
After training, the model can analyze new, unseen X-ray images and provide predictions regarding the likelihood of pneumonia. These predictions serve as an aid to healthcare professionals, offering an additional tool for diagnosis. However, it's important to note that the model's output should always be considered alongside clinical expertise and other diagnostic information.

The use of ResNet-50 for pneumonia detection in X-rays showcases the potential of deep learning in healthcare. Its ability to process and interpret complex visual data assists medical practitioners in more accurate and efficient diagnoses, potentially improving patient outcomes. However, continuous refinement and validation of these models are necessary to ensure their reliability and safety in real-world medical settings. The model ResNet-50 was applied to the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7) in order to compare results with the prototype Pneumonet.
Figure 182 shows part of the MatLab script that generates an ImageDataStore and exhibits selected images from the Chest X-ray dataset encompassing normal and Pneumonia from (Prashant, 2020) (Dataset 7). It establishes the fold count of training iterations and the process to construct the training, testing, and validation sets while implementing the ResNet-50 architecture.
The ImageDataStore is a type of object used for managing collections of image data. It's a powerful tool for handling large sets of images efficiently, providing a convenient way to read, process, and manage images for tasks like machine learning, computer vision, and deep learning. The ImageDataStore allows to store and organize large collections of images, making it easier to access and work with them. It provides an efficient way to read images directly from disk without loading all the images into memory simultaneously, which is particularly beneficial when dealing with extensive image datasets. Various preprocessing steps were performed on the images within the ImageDataStore, such as resizing, rotation, and normalization, using built-in functions. The ImageDataStore was used in conjunction with deep learning algorithms as it can seamlessly integrate with training and validation processes. It supports various image formats, including common formats like JPEG, PNG, and BMP, making it versatile for different image data sources.
A pre-trained ResNet-50 convolutional neural network model was initialized. The ResNet-50 is a specific architecture within the family of Residual Neural Networks introduced by Microsoft Research in 2015. MATLAB initialized a ResNet-50 network model and assigns it to the variable net. This model has already been pre-trained on a large dataset with millions of images, enabling it to extract high-level features from images effectively. The pre-trained ResNet-50 model comprises 50 layers and is designed for image classification tasks. It consists of a series of convolutional layers, pooling layers, and fully connected layers that learn to recognize patterns and features in images. The model was fine-tuned on a smaller dataset by adjusting the final layers and adding additional layers to suit the new classification task. The last few layers were modified of the pre-trained ResNet-50 network and retrained using the the Chest X-ray dataset encompassing normal and Pneumonia from (Prashant, 2020) (Dataset 7). This process allows the network to learn the features relevant to pneumonia detection while leveraging the knowledge gained from the original pre-training on ImageNet. By using net = resnet50, MatLab simplifies the process of accessing a powerful pre-trained CNN architecture, providing a foundation for various computer vision tasks without having to build the network architecture from scratch.
The code used was adapted from (Narayanan et al., 2020).
 
Figure 182. MatLab code to create an ImageDataStore to display some images of the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7) to define the number of folds to create the train, test and validation sets and to call the ResNet-50 architecture.
Figure 183 shows part of the MATLAB code designed to generate new layers, swap out the last layers of ResNet-50, utilize image processing functions, define training configurations, implement data augmentation, and resize images.
The replaceLayer function is part of the Neural Network Toolbox and is used to modify a pre-existing neural network model by replacing specific layers. This function is particularly helpful when to customize or fine-tune a pre-trained network for the task of detecting pneumonia on X-rays.
Creating a new layer or altering the architecture involved replacing specific layers within an existing network. The replaceLayer function simplifies this process by allowing to swap out and modify individual layers while retaining the rest of the network's architecture and parameters.
 Figure 183. MatLab code to create new layers and replace the last layers of the ResNet-50, to call functions to process the images, to define training options, to perform data augmentation and to resize images.
The trainNetwork is a function used to train a neural network model. It's a high-level function that simplifies the training process by handling the details of training iterations, data feeding, and optimization. This function is commonly used in deep learning workflows to train various types of neural networks for tasks like classification, regression, and more. The primary function is to train a neural network model using provided training data and options. It takes in input data, labels, a neural network architecture, and training options as arguments.
The training options are parameters that guide the training process. They influence how the network is trained and optimized during the training iterations. The mini-batch size determines the number of samples used in each iteration of training. A smaller size can speed up training but might reduce accuracy. The max epochs defines the maximum number of training epochs (iterations over the entire dataset). The initial learning rate sets the rate at which the model's parameters are updated during optimization. A higher learning rate may lead to faster convergence but could cause instability. The validation data specifies a separate dataset for validation, allowing monitoring of the model's performance on unseen data during training. The optimizer used was Adam. There are options to visualize and customize output during training, such as accuracy plots, custom functions to run at the end of each epoch, among others.
These functions and options streamline the training process in MATLAB, offering flexibility and control over various aspects of neural network training. Adjusting these options allows researchers and practitioners to optimize the training process based on their specific data and problem domains.
Figure 184 show part of the MatLab code used to create the ROC curve, to create the confusion matrix, to calculate some of the performance metrics and to create a function that converts images to grayscale. The perfcurve is used to generate a receiver operating characteristic (ROC) curve or precision-recall curve for evaluating classifier performance. It takes in true labels, scores, and positive class label as arguments. The function provides the points on the ROC curve or precision-recall curve and can calculate the area under the curve (AUC) for ROC.
The confusionmat computes the confusion matrix to evaluate the performance of a classification algorithm. It takes in the true labels and predicted labels as arguments. The function provides a confusion matrix showing the counts of true positive, true negative, false positive, and false negative predictions.
The confusion matrix gives insights into the performance of a classifier, showing how well it correctly predicts each class and where it might be making errors. This matrix is especially useful for assessing the model's performance across different classes in a multi-class classification problem.

 
Figure 184. MatLab code to generate the confusion matrix and the ROC curve. It also shows the calculation of some performance metrics and the creation of a function to convert images to grayscale.

Figure 184 displays the layers of the ResNet-50 architecture. ResNet-50, part of the Residual Neural Network family, is a deep convolutional neural network architecture known for its 50-layer depth. This architecture revolutionized deep learning by introducing residual connections, enabling the training of significantly deeper networks without facing the vanishing gradient problem. The ResNet-50 architecture consists of different types of layers arranged in a specific sequence to extract hierarchical features from input data, typically images. The initial input layer processes the input image data, typically of size 224x224 pixels in RGB format.
The network begins with several convolutional layers, where each layer performs convolutions to extract various features from the input image. 
The initial layers focus on capturing basic features like edges, textures, and colors. The ResNet-50 architecture primarily consists of residual blocks, each containing multiple convolutional layers. Each residual block contains a shortcut or skip connection that bypasses one or more layers, allowing the gradient to flow more directly during training. This mitigates the vanishing gradient problem. 
The core idea is to learn residuals (the difference between the output of layers and the input to those layers), enabling the network to learn the desired features more effectively.
Identity blocks within ResNet-50 maintain the spatial dimensions of the input feature maps. These blocks contain a sequence of convolutional layers without changing the feature map size, making it easier for the network to learn additional features without reducing resolution.
The pooling layers, typically using average pooling or max pooling, downsample the feature maps, reducing spatial dimensions and computational load while retaining important features.
Towards the end of the architecture, there are fully connected layers, also known as dense layers, that perform classification based on the extracted features.
In ResNet-50, the final fully connected layer usually outputs predictions across different classes (e.g., for ImageNet, it outputs probabilities for 1,000 classes).
ResNet-50's architecture emphasizes deeper networks by utilizing residual connections, which allow smoother and more effective gradient flow during training. This design enables the network to learn complex hierarchical features, contributing to its effectiveness in various computer vision tasks like image classification, object detection, and segmentation. The specific layer arrangement and the inclusion of residual connections are pivotal in ResNet-50's success in handling deep learning


 
Figure 184. Output of the MatLab Analyze network function to show the layers of the ResNet-50 architecture.
Figure 186 displays the number of images that are normal and with pneumonia on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7) as well as some X-ray examples. 
Figure 186. Output of the Matlab code to display details of the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).
Figure 187 displays the training progress of using the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7). The training progress window is a visual interface that displays the progress and key metrics during the training of a deep learning model. This window provides real-time updates on the training process, allowing users to monitor and analyze the performance of the model as it learns from the training data. When training a neural network using functions like trainNetwork, MATLAB offers a training progress window by default, which shows various details. The training progress plot illustrates the progress of key metrics, typically including training loss and validation loss over epochs or iterations. It allows to observe how these metrics change throughout the training process. Information such as accuracy, loss values, and other relevant metrics are updated and displayed in the window as the training progresses. The progress window visualizes the learning rate schedule, showing how the learning rate changes during training.The window displays validation metrics to assess the model's performance on data that it hasn't seen during training. It is possible to track the training process in real time, identifying potential issues or improvements as they arise. It facilitates the evaluation of the model's performance, allowing users to decide whether to continue training or adjust hyperparameters. If the model encounters problems like overfitting or slow convergence, the progress window helps in diagnosing these issues by visualizing the training dynamics.

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

The application of the ResNet-50 model for pneumonia detection from X-ray images within the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) exhibited robust performance. It delivered impressive metrics: an accuracy of 91.3%, a recall rate of 86.6%, precision reaching 82.1%, an F1-score of 84.2%, specificity at 93%, and an AUC of 94.6%, as detailed in Table 20.
Table 20. Performance metrics of the ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) for pneumonia detection on X-rays.
Model	Accuracy	Precision	Recall	AUC	Specificity	F1-score
ResNet-50	91.3%	82.1%	86.6%	96.6%	93%	84.2%

This section intends to achieve #Objective_2, answering #Research_Question_3 and presenting #Hypothesis_4.















6.3	Pneumonia Detection on X-rays Using AlexNet

AlexNet is a deep convolutional neural network architecture that gained significant attention after winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. 
AlexNet consists of multiple convolutional layers that extract hierarchical features from input images. The initial layers capture low-level features like edges and textures, while deeper layers learn complex patterns.
Pre-trained AlexNet models, trained on vast datasets like ImageNet, are commonly used. Transfer learning involves fine-tuning these pre-trained models on a smaller dataset of X-ray images related to pneumonia detection.
X-ray images of patients with and without pneumonia are found in the  ResNet-50 on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7). The network learns to differentiate between normal and abnormal (pneumonia-infected) X-rays based on the learned features.
Once trained, the model's performance is assessed using evaluation metrics like accuracy, precision, recall, and the area under the ROC curve.These metrics measure how well the model identifies pneumonia cases and distinguishes them from normal cases.
Pneumonia detection, required specialized architectures and fine-tuning due to differences in domain and data characteristics compared to natural images. AlexNet's deep architecture enables it to learn intricate patterns in X-ray images, aiding in pneumonia detection. AlexNet, with its ability to extract complex features from images, has been adapted for pneumonia detection on X-rays, showcasing the versatility of deep learning models across diverse domains. However, advancements in medical image analysis often involve architectures specifically tailored for healthcare tasks, taking into account the unique characteristics and requirements of medical datasets. 
The research in this section used AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7) to classify pneumonia. AlexNet consists of multiple layers that capture and process information from input images. The architecture of AlexNet is characterized by its depth and utilization of convolutional layers, pooling layers, and fully connected layers.
The initial layer receives input images, typically in RGB format. The network starts with convolutional layers that perform feature extraction. The first convolutional layer extracts basic features like edges and textures, and subsequent layers capture increasingly complex patterns.
Rectified linear unit activation functions are applied after convolutional layers to introduce non-linearity, helping the network learn more complex representations.
The max pooling layers follow some convolutional layers, reducing spatial dimensions while retaining important features.
The LRN layers were introduced in AlexNet to provide local contrast normalization, which aims to improve generalization. Towards the end of the network, there are fully connected layers that act as a classifier. These layers aggregate the features learned by previous layers to make predictions.
The final layer in AlexNet applies the softmax function to produce probability distributions across different classes for classification tasks.
AlexNet was one of the earlier deep CNN architectures, consisting of eight layers with trainable parameters. The architecture is designed for parallel processing, utilizing two GPUs for efficient computation during training.
The network's architecture includes large receptive fields in later layers, allowing it to capture overlapping features, enhancing feature learning.
AlexNet's success in winning the ImageNet competition in 2012 was a pivotal moment in the advancement of deep learning. Its architecture influenced subsequent CNN designs, emphasizing the importance of deep, convolutional architectures in computer vision tasks.
While AlexNet's architecture has since been surpassed by deeper and more intricate models, its contribution to the field of deep learning and its impact on the development of convolutional neural networks for image classification remain significant.
Figure 191 displays some of the X-rays of the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7) as well as the number of images with and without pneumonia.
 
Figure 191. The MatLab code to generate a ImageDataStore and that displays some of the X-rays of the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7) as well as the number of images with and without pneumonia.
In MATLAB, ImageDataStore is a powerful tool within the Image Processing Toolbox that serves as a specialized data structure for managing and working with collections of image data. It's particularly useful for handling large sets of image files efficiently, providing an organized and convenient way to access and process images for various tasks in image analysis and machine learning. It efficiently manages a large number of image files without loading all the images into memory simultaneously. It supports a variety of image formats such as JPEG, PNG, BMP, TIFF, among others. Allows direct access to images on disk, enabling on-the-fly processing and manipulation without loading the entire dataset into memory. It seamlessly integrates with various image processing functions and machine learning workflows in MatLab. Enables preprocessing of images by applying transformations like resizing, cropping, rotation, and normalization to prepare data for training machine learning models.
Figure 192 shows X-rays of the used Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7) with the respective labels namely pneumonia and normal.

 
Figure 192. Images from the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).
Figure 193 shows the training progress of applying AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7). The training progress represents the iterative process and performance evaluation of machine learning or deep learning models during the training phase. It involves monitoring various metrics, visualizations, and updates related to the model's learning process as it iteratively refines its parameters based on the provided training data. It displays and tracks metrics like training loss, validation loss, accuracy, or any other custom-defined metrics. It allows to view the progress across training iterations or epochs, showing how the model's performance changes over time. Information about the optimization algorithm's behavior, showcasing how the model's parameters are updated during training.
The progress window displays the metrics calculated on a separate validation dataset, providing insights into the model's generalization performance. The progress show how the learning rate changes throughout training. MATLAB provides real-time updates in the command window or console, showing training progress, metrics, and updates as training proceeds. Customizable plots to visualize metrics like loss, accuracy, or any user-defined metrics over epochs or iterations using plot functions or built-in tools.
The trainingPlot function provides an interactive tool to visualize and customize the training progress plots, enabling real-time exploration of metrics during training.

 
Figure 193. Training progress of using AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).

Training AlexNet for pneumonia detection in X-ray images using MATLAB involves several steps. A dataset containing X-ray images labeled as pneumonia-positive and pneumonia-negative was obtained in this example the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).
 The images were resized to a consistent size (224x224 pixels) and consider normalization or other preprocessing steps.
The ImageDataStore in MATLAB was used to load and organize the dataset. This prepares the data for training and validation. The pre-trained AlexNet model was loaded using the alexnet function in MATLAB. The network's last layers were modified for the specific binary classification task by replacing the final fully connected layers.
The training options using trainingOptions were used to specify parameters like optimization algorithm, mini-batch size, learning rate, among others.The training and validation data were included as part of the training options.
The trainNetwork was used to train the modified AlexNet model with the prepared dataset and training options.
Figure 194 shows the confusion matrix of using AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7). The TP were 235, the FP 65, the FN 18 and the TN 618. The results showed an accuracy of 91.1%, a precision of 78.3% and a recall of 92.9%,
 
Figure 194. Confusion matrix of using AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).

Figure 195 shows the ROC curve of using AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).
The Receiver Operating Characteristic (ROC) curve is a graphical representation used to assess the performance of a binary classification model across various threshold settings. It illustrates the trade-off between the true positive rate (TPR) and the false positive rate (FPR) as the discrimination threshold of the classifier is varied.

 
Figure 195. The ROC curve of using AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).

Figure 196 shows The AUC, precision, overall precision, recall, overall recall and F1-score of using AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7). In deep learning, various performance metrics are used to evaluate the effectiveness and accuracy of models across different tasks. These metrics measure how well a model performs on a given dataset and help in assessing its strengths and weaknesses.

 
Figure 196. The AUC, precision, overall precision, recall, overall recall and F1-score of using AlexNet on the Chest X-ray (Covid-19 & Pneumonia) dataset (Prashant, 2020) (Dataset 7).

The AlexNet model, utilized for pneumonia detection on X-rays using the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), demonstrated strong performance. It achieved an accuracy of 91.1%, a recall rate of 92.9%, precision of 78.1%, an F1-score of 84.9%, specificity of 90.5%, and an area under the curve of 97.4%, as shown on table 21.


Table 21. Performance metrics of the AlexNet on the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4) for Covid-19 detection on CT scans.
Model	Accuracy	Precision	Recall	AUC	Specificity	F1-score
AlexNet	91.1%	78.3%	92.9%	97.4%	90.5%	84.9%

This section intends to achieve #Objective_2, answering #Research_Question_3 and presenting #Hypothesis_4.













6.4	Pneumonia Detection on X-rays Using Explainable Artificial Intelligence
Pneumonia detection using techniques like Grad-CAM  and LIME (enhances the interpretability of deep learning models applied to X-ray images.
 Grad-CAM is a technique used for visualizing and understanding the regions of an image that contribute the most to the prediction made by a convolutional neural network. It generates heatmaps that highlight the regions of an X-ray image that the model focuses on when making predictions. It uses the gradients of the target class with respect to the final convolutional layer to understand which features are important for classification. Grad-CAM highlight areas in X-ray images that the model relies on for predicting pneumonia. It helps in understanding the model's decision-making process.
LIME provides explanations for individual predictions made by a model, making the predictions more interpretable. LIME approximates the behavior of complex models like deep neural networks by fitting a simpler, interpretable model to local regions of the input space around a prediction. It generates local explanations by perturbing input data and observing the impact on the model's predictions. LIME can provide insights into why a model predicted a certain X-ray image as pneumonia-positive or pneumonia-negative. It identifies which areas of the X-ray influenced the model's decision.
Both Grad-CAM and LIME aid in understanding the underlying reasoning behind a model's predictions in pneumonia detection using X-ray images. These techniques help in verifying whether the model focuses on clinically relevant areas in X-rays for making accurate predictions.
Interpretable models contribute to building trust among healthcare professionals, potentially facilitating the integration of AI-based diagnostic tools into clinical practice. While these techniques provide insights, interpretation, and local explanations, they don't inherently ensure the model's robustness or generalizability.
Interpretability methods like Grad-CAM and LIME serve as post-hoc tools to analyze and explain the decisions of complex deep learning models. They complement but do not replace model validation, evaluation, and clinical validation processes. In summary, using Grad-CAM and LIME in pneumonia detection with X-ray images facilitates the interpretability of deep learning models, making their predictions more transparent and potentially more acceptable for clinical adoption by providing insights into the regions of X-rays driving the predictions.
The code was adapted from (Marques, 2023). This research used LIME and grad-CAM on classified images using the ResNet-50 architecture on the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4) to facilitate interpretability.


 

 
Figure 196. The MatLab code to create an ImageDataStore, split into train and validation and display some images of the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4).

Figure 198 shows the MATLAB code that generates an ImageDataAugmenter, invokes ResNet-50 while modifying its final layers, and resizes images to dimensions of 224x224. In MATLAB, the ImageDataAugmenter is a tool within the Image Processing Toolbox used for augmenting and manipulating image data during the training of machine learning or deep learning models. It enables the generation of augmented images by applying a variety of transformations and modifications to the original images. This augmentation process helps enhance the diversity of the training data, which can improve the model's ability to generalize and perform well on unseen data. Rotation, flipping, scaling, and cropping were used to simulate different orientations and viewpoints. Brightness, contrast, saturation, and hue adjustments help to create variations in color tones. Allows introducing randomness in augmentation parameters for generating diverse image variations. Enables customization of augmentation settings and parameters for specific data augmentation needs. It allows augmented images to be efficiently used during training without loading all images into memory at once. It takes advantage of parallel computing capabilities in MATLAB, speeding up the augmentation process for large datasets. Enhanced Data Diversity: Augmentation helps in generating varied training samples, reducing overfitting and improving model generalization. Augmentation techniques and parameters were used to suit the specific requirements of the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4). Seamlessly integrates with other MATLAB image processing functions and deep learning workflows for efficient training.
The ImageDataAugmenter in MATLAB serves as a valuable tool for enhancing the robustness and performance of machine learning and deep learning models by providing diverse and augmented training data.
A pre-trained ResNet-50 model was loaded. ResNet-50 is a convolutional neural network architecture that consists of 50 layers, known for its deep structure and residual learning blocks, which help in addressing the vanishing gradient problem. It has been pre-trained on large-scale datasets like ImageNet and is capable of classifying images into thousands of categories. ResNet-50, a variant of the Residual Network architecture, is composed of 50 layers, characterized by its deep structure and unique residual blocks. These residual blocks address the challenge of training very deep neural networks by introducing skip connections that facilitate the flow of information through the network, thus mitigating the vanishing gradient problem. The network begins with a 7x7 convolutional layer followed by max-pooling, downsampling the input. ResNet-50 is built upon residual blocks, each containing multiple convolutional layers. The residual blocks feature skip connections, known as identity shortcuts, that allow the gradient to flow directly through the block. ResNet-50 employs a bottleneck architecture in its residual blocks, which consists of three convolutional layers: 1x1, 3x3, and 1x1 convolutions. The 1x1 convolutions reduce the dimensionality, while the 3x3 convolutions learn feature representations. The skip connections enable the network to learn residual mappings, enabling easier optimization of very deep networks. Towards the end, the network employs global average pooling to reduce spatial dimensions. A final fully connected layer maps the extracted features to the output classes.

 

Figure 198. The MatLab code to create a ImageDataAugmenter, to call the ResNet-50 and change the last layers and to resize images to 224x224.


Figure 199 shows the training progress of using ResNet-50 on the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4). The training process window is a graphical user interface (GUI) component that provides real-time feedback and visualization of the training progress when training machine learning or deep learning models. This window is commonly displayed when training models using the function trainNetwork.
The training progress window in MATLAB facilitates the interactive monitoring and analysis of machine learning or deep learning model training, offering insights into the training process and helping users make informed decisions to optimize model performance.
 

Figure 199. The training progress of using ResNet-50 on the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4).

Figure 200 shows the classification process using the validation dataset. In MATLAB, the classify function is used to classify observations or data points using a trained classification model. It allows to apply a pre-trained classifier to new data and predict their class labels based on the learned patterns from the training dataset. The deep learning classifier used was the ResNet-50. The function takes the pre-trained classifier and the new data as inputs. It applies the learned patterns or decision boundaries from the trained model to predict the class labels or probabilities for the new observations. The function provides the predicted class labels or probabilities for the new data based on the trained model's predictions.
 

Figure 200. The use of the classify function and display of classified X-rays.
Grad-CAM is a technique used for visualizing and understanding the regions of an image that contribute most to the prediction made by a convolutional neural network. It highlights the important regions in an image that influenced the model's decision. In a CNN, the final convolutional layer captures high-level features before the fully connected layers.
Grad-CAM focuses on this last convolutional layer and the subsequent global average pooling and fully connected layers. Calculate the gradients of the predicted class score with respect to the feature maps of the last convolutional layer. These gradients signify the importance of each feature map towards the final prediction. Compute the importance of each feature map by averaging the gradients spatially, weighting the feature maps' importance. Use the weighted importance to generate a heatmap by linearly combining the feature maps, highlighting the regions most relevant to the predicted class.
Grad-CAM provides visual explanations of a model's predictions, highlighting the areas in an image that contributed to a certain prediction. It aids in understanding why a CNN made a specific prediction, enhancing interpretability and trust in deep learning models. Helps in diagnosing whether a model is focusing on clinically relevant features, especially in medical imaging tasks like identifying disease-related areas in X-ray.
Figure 201 shows the use of grad-CAM on classified images. Grad-CAM works by calculating the gradient of the class activation score (CAS) with respect to the input image. The CAS is the output of the last convolutional layer of the CNN, before the final classification layer. The gradient of the CAS indicates which regions of the image are most important for the predicted class. To calculate the gradient of the CAS, Grad-CAM uses the backpropagation algorithm. The backpropagation algorithm is a technique for computing the gradient of a function with respect to its inputs. In this case, the function is the CNN model, and the inputs are the pixels of the input image. Once the gradient of the CAS is calculated, Grad-CAM uses it to create a saliency map. The saliency map is a heatmap that shows the importance of each pixel in the image for the predicted class. The brighter the pixel in the saliency map, the more important the pixel is for the predicted class. MATLAB provides a function called GradCAM that can be used to calculate Grad-CAM visualizations. The function takes a CNN model and an input image as input, and it returns a saliency map.
Grad-CAM is a valuable tool for understanding the decision-making process of CNNs. It can be used to identify the most important regions of an image for a particular class prediction, understand how different features of an image contribute to the predicted class, debug CNN models and identify potential problems.
 
 

Figure 201. MatLab code that uses the gradCAM function to generate the heatmap.

Figure 202 shows the use of the imageLIME function. LIME is a technique for explaining the predictions of convolutional neural networks by generating local explanations of each pixel in an image. LIME is a popular tool for understanding the decision-making process of CNNs and gaining insights into their behavior. The imageLIME works by randomly perturbing the input image and observing how the predictions of the CNN change. The perturbations are generated using a surrogate model, which is a simpler model that is trained to approximate the predictions of the CNN.

By analyzing how the predictions change in response to the perturbations, imageLIME can generate local explanations of each pixel in the image. These explanations show how the pixel contributes to the predicted class. MATLAB provides a function called imageLIME that can be used to generate imageLIME explanations. The function takes a CNN model, an input image, and a surrogate model as input, and it returns a set of local explanations for each pixel in the image.

 
 ![image](https://github.com/user-attachments/assets/c1d2a1e1-2b53-4775-af1b-c5fbfc2d0633)

Figure 202. MatLab code to use LIME on the classified images.


Figure 203 shows the final prediction with the score for a certain X-ray, this helps in understanding if the predictions are highly accurate or not, facilitating the work of the physician on the diagnosis process.


 
Figure 203. Final classification showing the score of the prediction

This section intends to present #Hypothesis_8.


























6.5	Application of the Developed Prototype Pneumonet to Detect Pneumonia on X-rays
Pneumonet is a convolutional neural network model that has been specifically designed to detect pneumonia in chest X-ray images. It is a deep learning model that has been trained on a large datasets of X-ray images, including images of patients with pneumonia and images of patients without pneumonia.
The Pneumonet model can be used to identify pneumonia in chest X-rays in several ways. It can be used as a part of an automated X-ray diagnosis system. In an automated X-ray diagnosis system, Pneumonet would be used to analyze chest X-ray images and output a prediction of whether the patient has pneumonia or not. This prediction could then be used to alert a doctor or other healthcare professional, who could then review the image and make a more definitive diagnosis.
Pneumonet can be used in telemedicine applications to provide remote diagnosis of pneumonia. A patient would take an X-ray image at home and then send it to a doctor or other healthcare professional. The doctor or healthcare professional would then feed the image into Pneumonet to get an initial diagnosis. This could help to reduce the need for patients to travel to hospitals or clinics for diagnosis.
Early diagnosis of pneumonia is essential for successful treatment. Pneumonet can be used to identify cases of pneumonia earlier in the course of the disease, which could lead to faster and more effective treatment.
Pneumonet has been shown to be highly accurate in detecting pneumonia on X-rays, can analyze X-ray images very quickly, which makes it suitable for use in real-time applications.
Pneumonet can be trained on a large dataset of X-ray images, which makes it able to detect pneumonia in a wide variety of patients. Pneumonet is a relatively low-cost solution for detecting pneumonia on X-rays. This makes it a good option for resource-limited settings.
Pneumonet is a promising tool for the detection of pneumonia on X-rays. The model has several potential benefits, including accuracy, speed, scalability, and low cost. As research continues, Pneumonet is likely to play an increasingly important role in the diagnosis and treatment of pneumonia.
In this study Pneumonet was tested on three different datasets namely the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9), in order to compare the results obtained with the work of other authors. 
The model was developed by fine tuning the AlexNet model and changing the last layers in order to be more efficient to detect pneumonia on X-rays.
The application of the developed prototype Pneumonet to detect pneumonia on X-rays involves leveraging a custom-built version of the AlexNet architecture by changing it and fine-tuning for pneumonia classification tasks, using several datasets comprising X-ray images annotated for pneumonia presence or absence. 
Pneumonet refers to a modification of the AlexNet architecture, adapted to better suit pneumonia detection from X-ray images. This adaptation involved altering layers, incorporating regularization techniques and fine-tuning specific parameters.
Annotated X-ray images containing cases with and without pneumonia were organized into training, validation, and test sets. These datasets serve as the basis for training and evaluating the Pneumonet model.
The X-ray images were preprocessed, which involved resizing, normalization, and augmentation techniques to enhance model robustness and generalization.
The Pneumonet model was trained on public datasets. The training involved feeding the X-ray images through the network, adjusting weights and biases to minimize classification errors, using techniques like backpropagation and optimization algorithms.
The model's performance wass evaluated using a separate validation set. Metrics like accuracy, precision, recall, F1-score, AUC-ROC, and confusion matrices were computed to assess its performance.
The model underwent fine-tuning by adjusting hyperparameters or the architecture based on validation set performance to improve its accuracy and robustness.
The final model wass tested on a separate set of unseen X-ray images (the test set) to gauge its performance on new, unseen data.
Techniques like Grad-CAM or LIME were applied to understand the model's decisions and visualize the areas in X-rays that contributed to the pneumonia classification.
The dstasets are public to adhere to privacy, ethical guidelines, and regulatory compliance (e.g., HIPAA) is crucial.
The application of the developed prototype Pneumonet for pneumonia detection on X-rays aims to contribute to accurate and efficient diagnostic capabilities, potentially aiding healthcare professionals in timely and accurate disease identification.












6.5.1	Application of the Developed Prototype Pneumonet on the Dataset Chest X-ray (Covid-19 & Pneumonia)

For this research was developed a prototype for pneumonia named Pneumonet. The application of the developed prototype Pneumonet on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) containing images related to COVID-19 and pneumonia involved using this customized neural network to classify X-ray images into categories like pneumonia, or normal cases. 
The dataset was annotated to label images according to their classes (pneumonia and normal). The dataset was divided into training, validation, and test sets ensuring a balanced representation of classes across sets.
A customized version of the AlexNet architecture (Pneumonet) was trained using the prepared dataset. This model was fine-tuned and adapted to classify, pneumonia, and normal cases from chest X-ray images.
Tthe X-ray images were preprocessed by resizing, normalization, and applying augmentation techniques to improve model robustness.
The Pneumonet model was trained on the training set, adjusting its weights and parameters to learn patterns indicative of pneumonia, and normal conditions.
The model's performance was assessed by using the validation set. Metrics like accuracy, precision, recall, F1-score, and confusion matrix were computed to evaluate classification performance.
Hyperparameters were fin-tuned and the architecture adjusted based on validation set performance to enhance model accuracy and generalization.
The final trained Pneumonet model was evaluated on the separate test set to gauge its performance on unseen data. It can generalize well to new X-ray images.
Methods like Grad-CAM and LIME were used to interpret model decisions and visualize which regions of X-ray images contribute to specific classifications.
The application of the developed prototype Pneumonet on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) dataset aims to create a reliable diagnostic tool for identifying COVID-19, pneumonia, and normal cases from X-ray images, potentially aiding healthcare professionals in accurate and efficient disease diagnosis.
Figure 204 shows the number of X-rays with and without pneumonia and also some images of the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7). These images were labeled to identify which cases have pneumonia or not.
 
Figure 204. The number of X-rays with and without pneumonia and also some images of the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7).

The dataset was divided into training, validation, and testing subsets to rigorously assess the model's performance. A distinct testing set, unseen by the model before, ensures an accurate measurement of its ability to generalize to new, unseen data. Datasets form the fundamental component in training accurate, reliable, and unbiased deep learning models for pneumonia detection in X-ray images. They lay the groundwork for the model to learn, enabling informed and precise classifications that greatly impact its performance.
Figure 205 displays some images of the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7).
 
Figure 205. X-rays with and without pneumonia of the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7).

Figure 206 displays the training progress of applying the Pneumonet in the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7). The training progress window is a visualization tool in MATLAB that displays the progress of a training session. It shows the training loss and accuracy for each epoch of the training process. This information was used to monitor the progress of the training and identify potential problems. To use the training progress window, you must start a training session in MATLAB. Once the training session has started, the training progress window will automatically appear. The training progress window displays the current epoch of the training process, the training loss for the current epoch. The loss is a measure of the error between the model's predictions and the true labels. The training accuracy for the current epoch is also displayed. The accuracy is a measure of how often the model correctly classifies the training data.
The training progress window can be helpful for monitoring the progress of a training session and identifying potential problems. For example, if the loss is increasing or the accuracy is decreasing, this may indicate that the model is overfitting or underfitting. Overfitting occurs when the model learns the training data too well, and it does not generalize well to new data. Underfitting occurs when the model does not learn the training data well enough, and it does not perform well on the training data.
The training progress window can also be used to identify when the model is converged. Convergence occurs when the loss and accuracy are no longer improving significantly. Once the model has converged, you can stop the training session.
By using the training progress window it is possible to effectively monitor the progress of the training sessions and achieve better results.
 
Figure 206. The training progress of applying the Pneumonet in the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7).
Accuracy, validation accuracy, loss, and validation loss are all important metrics for evaluating the performance of a deep learning model. Accuracy is the proportion of correct predictions made by the model, while validation accuracy is the proportion of correct predictions made on a separate validation dataset. Loss is a measure of the error between the model's predictions and the true labels, while validation loss is a measure of the error on the validation dataset.
In the case of the Pneumonet model, which is a convolutional neural network model that has been specifically designed to detect pneumonia in chest X-ray images, it has been shown to have high accuracy and validation accuracy. The study showed that the model correctly identified pneumonia in the majority of the cases, and validation accuracy was high. The model also has low loss and validation loss, which indicates that it is able to make accurate predictions consistently.
These results suggest that the Pneumonet model is a promising tool for the detection of pneumonia on X-rays. It is accurate, fast, scalable, and low-cost, making it a good option for use in real-time applications.
Figure 207 shows the classification of some X-rays after training. The classify function in MatLab is a general-purpose function for making predictions using a trained machine learning model. The function takes as inputs the model (Pneumonet) the x (input data to make predictions on) and the threshold to use for classification. The function returns the predicted class labels for the input data. The predictions are a vector of class labels, where each label is either 0 (no pneumonia) or 1 (pneumonia). The classify function is a powerful tool for making predictions using trained machine learning models. It is a versatile function that can be used with a variety of models and data types.
 
Figure 207. The classified X-rays after training
Grad-CAM is a technique for visualizing the importance of different regions of an image for a particular class prediction. It is a popular tool for understanding the decision-making process of convolutional neural networks. 
Grad-CAM works by calculating the gradient of the class activation score with respect to the input image. The CAS is the output of the last convolutional layer of the CNN, before the final classification layer. The gradient of the CAS indicates which regions of the image are most important for the predicted class.
To calculate the gradient of the CAS, Grad-CAM uses the backpropagation algorithm. The backpropagation algorithm is a technique for computing the gradient of a function with respect to its inputs. In this case, the function is the CNN model, and the inputs are the pixels of the input image.
Once the gradient of the CAS is calculated, Grad-CAM uses it to create a saliency map. The saliency map is a heatmap that shows the importance of each pixel in the image for the predicted class. The brighter the pixel in the saliency map, the more important the pixel is for the predicted class.
MATLAB provides a function called GradCAM that can be used to calculate Grad-CAM visualizations. The function takes a CNN model, an input image, and a class label as input, and it returns a saliency map.
Figure 208 shows the use od the MarLab gradCAM function on the classified images.
 
Figure 208. Use of grad-CAM on the classified X-rays after training.
LIME is a technique for explaining the predictions of black-box models by generating local explanations for each input instance. LIME is a popular tool for understanding the decision-making process of complex models, such as convolutional neural networks (CNNs).
LIME works by creating a simplified, interpretable model that approximates the behavior of the black-box model around a particular input instance. The simplified model is then used to generate explanations for the input instance.
To create the simplified model, LIME randomly perturbs the input instance and observes how the black-box model's predictions change. The simplified model is then trained to fit these perturbed predictions.
Once the simplified model has been trained, LIME can be used to generate explanations for the input instance. These explanations are typically in the form of saliency maps, which show how the input instance contributes to the black-box model's prediction. MATLAB provides a function called LIME that can be used to calculate LIME explanations. The function takes a black-box model, an input instance, and a number of perturbations as input, and it returns a saliency map.
Figure 209 shows the use of LIME on some of the classified images of the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) after being trained by Pneumonet.
 
Figure 209. Use of LIME on the classified X-rays after training.

Figure 210 shows the confusion matrix obtained after being trained by Pneumonet on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7). The TP were 227, the FP were 5, the FN was 26 and the TN 679. The accuracy was 96.7%, the precision 97.8% and the recall 89.7%.
 
Figure 210. The confusion matrix obtained after being trained by Pneumonet on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7).

A confusion matrix is a table of statistics that is used to evaluate the performance of a classification model. It is a valuable tool for understanding the accuracy of a model, as well as its ability to identify true positives, true negatives, false positives, and false negatives.
By analyzing the confusion matrix, it is possible to gain insights into the strengths and weaknesses of the classification model. For example, if the number of true negatives is high, it indicates that the model is good at correctly identifying negative instances. Conversely, if the number of false positives is high, it suggests that the model is making too many mistakes when classifying positive instances.
Figure 211 displays the ROC curve after applying the Pneumonet on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7). The AUC was 98.39%. ROC is a graphical plot that illustrates the trade-off between the true positive rate (TPR) and the false positive rate (FPR) for a binary classifier.
The TPR is the proportion of positive instances that are correctly classified as positive, and the FPR is the proportion of negative instances that are incorrectly classified as positive.
An ROC curve is typically created by plotting the TPR against the FPR for a range of different thresholds. The threshold is the value that is used to determine whether an instance is classified as positive or negative.
A higher TPR indicates that the classifier is better at identifying positive instances, while a lower FPR indicates that the classifier is better at avoiding classifying negative instances as positive.
The AUC (Area Under the ROC Curve) is a metric that summarizes the performance of a binary classifier across all possible thresholds. A higher AUC indicates that the classifier is better at differentiating between positive and negative instances.
 
Figure 211. The ROC curve obtained after being trained by Pneumonet on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7).
 

Figure 211. The precision, recall, AUC and F1-score calculated in MatLab after being trained by Pneumonet on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7).
The developed prototype model Pneumonet, utilized for detecting pneumonia via X-rays using the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), demonstrated strong performance. It achieved an accuracy of 96.7%, a recall rate of 89.7%, precision of 97.8%, an F1-score of 93.12%, specificity of 99.3%, and an area under the curve of 98.39%, as shown on Table 22.
Table 22. Performance metrics of the developed prototype on the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4) for Covid-19 detection on CT scans.
Model	Accuracy	Precision	Recall	AUC	Specificity	F1-score
Developed prototype	96.7%	97.8%	89.7%	98.39%	99.3%	93.12%
This section intends to achieve #Objective_2, answering #Research_Question_3 and presenting #Hypothesis_5.
6.5.2	Application of the Developed Prototype Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification Dataset

The Pneumonet model is a convolutional neural network (CNN) model developed for the specific task of detecting pneumonia in chest X-ray images. 
The labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) contains a collection of chest X-ray images of patients with and without pneumonia. The dataset is divided into training, validation, and test sets.
To apply the Pneumonet model to this dataset, the data was processed. It involved manipulating the chest X-ray images, such as resizing them to a standard size and normalizing them to have a mean of 0 and a standard deviation of 1.
Once the data was prepared, it was trained on the Pneumonet model on the training set. The model was trained for a sufficient number of epochs to achieve good performance on the validation set.
The performance was evaluated on the test set. The model's performance was measured using metrics such as accuracy, precision, recall, and F1-score.
The Pneumonet model has been shown to be effective in classifying chest X-ray images. In this study, the model achieved high accuracy on a test set of X-rays.
The model is highly accurate. It has been shown to achieve high accuracies on different public well defined datasets.
The model is fast. It can classify images very quickly, which makes it suitable for use in real-time applications such as telemedicine.
The model is scalable. It can be trained on large datasets of chest X-ray images, which allows it to learn to classify a wide variety of images.
The model is generalizable. It has been shown to perform well on new data that it has not been trained on.
Overall, the Pneumonet model is a powerful and versatile tool for classifying chest X-ray images. It is accurate, fast, scalable, and generalizable, making it a valuable tool for researchers and clinicians alike.
The labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) (referred to as GWCMCx) was obtained from the Guangzhou Women and Children Medical Center (Guangzhou, China) that has compiled a dataset of 5856 X-ray images of pediatric patients, taken as part of routine clinical care. The dataset is divided into two parts: a training set of 3883 images of pneumonia cases and 1349 images of normal cases, and a test set of 234 images of pneumonia and 390 of normal cases.
Figure 213 shows the number of X-rays with and without pneumonia and some sample images of the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).
 
Figure 213. The number of X-rays with and without pneumonia and some sample images of the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).

Figure 214 shows the progress of the training of applying the Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8). The training progress window in MATLAB is a graphical user interface that provides real-time feedback on the training process of a deep learning model. It displays the current epoch, loss, and accuracy of the model, as well as a graph of the training progress. The current epoch is the number of times that the model has been trained on the entire training dataset. The loss is a measure of the difference between the model's predictions and the true labels. The goal of training is to minimize the loss. Accuracy is the proportion of predictions that are correct. The graph of training progress shows how the loss and accuracy have changed over time. This can be used to identify any potential problems with the training process.
The training progress window is a valuable tool for monitoring the training process and identifying any potential problems. It can also be used to track the progress of the model over time and see how it is improving. The loss should be decreasing over time. If the loss is increasing, it may indicate that the model is overfitting or that the hyperparameters need to be tuned. The accuracy should be increasing over time. If the accuracy is not increasing, it may indicate that the model is not learning or that the data is not well-labeled.
The graph can help to identify any trends or patterns in the training process. For example, if the loss is decreasing but the accuracy is not increasing, it may indicate that the model is focusing on memorizing the training data rather than learning the underlying patterns.
By using the training progress window effectively, it is possible to get a better understanding of the training process and make sure that your model is learning effectively.
 Figure 214. The training progress of applying Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).
Figure 215 shows the X-rays after classification after applying the Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8). In MATLAB, the classify function is used for making predictions or classifying new observations using a pre-trained classification model. This function assigns class labels to input data based on the learned patterns from the training dataset.
The classify function is particularly useful in MATLAB for applying pre-trained classification models to new data, enabling easy and quick predictions based on learned patterns from the training phase.

 
Figure 215. The X-rays after classification, after applying the Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).
,
Figure 216 shows the application of grad-CAM to the X-rays after classification. Grad-CAM is a technique used to visualize and understand the regions of an image that significantly contribute to the predictions made by a convolutional neural network (CNN) in a classification task. It helps in interpreting the decisions of the CNN by highlighting the important regions within an image that influence the final prediction. In a CNN, the final convolutional layer captures high-level features before the fully connected layers. Grad-CAM calculates the gradients of the predicted class score (logit) with respect to the activations of the last convolutional layer. It computes the importance of each activation map by averaging the gradients spatially, giving more weight to the activations that contribute more to the class score. Generates a heatmap by combining the activation maps based on their importance weights, highlighting the regions that had a significant impact on the prediction.
Grad-CAM provides visual explanations for CNN predictions, making the decision-making process more transparent and interpretable. It helps in understanding which parts of an image the model focused on to make a certain prediction, enhancing trust and transparency in the model's decisions. Useful in object localization tasks, indicating the areas where the model detected specific objects within an image. Grad-CAM performs a forward pass to get the activations of the last convolutional layer and compute the gradients of the predicted class score with respect to these activations. Average the gradients spatially to get the importance weights for each activation map. Generate a heatmap by combining the activation maps based on their importance weights. Overlay the heatmap onto the original image, highlighting the regions that contributed most to the model's prediction.
By visualizing the heatmap overlaid on the original image, Grad-CAM provides insights into the specific regions within an image that influenced the CNN's prediction, aiding in model interpretability and understanding.
 
Figure 216. Applying grad-CAM to the X-rays after classification.
Figure 217 shows the application of LIME on classified images. LIME is an interpretability technique used to explain the predictions of machine learning models, particularly in the context of complex models like deep neural networks. It aims to provide local and human-interpretable explanations for individual predictions made by a model.
LIME focuses on generating explanations for specific predictions rather than the entire model behavior. It is model-agnostic, meaning it can be applied to any deep learning model regardless of its complexity. LIME creates a simpler, interpretable 'surrogate' model around the prediction of interest. It generates a local dataset by perturbing the original instance and records the predictions of the complex model on these perturbed samples.
Using the the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and corresponding predictions, LIME fits an interpretable model (such as linear regression or decision tree) on the perturbed samples. This simpler model approximates the complex model's behavior around the specific instance. 
LIME provides insights by assessing the importance of different features in the interpretable model. It quantifies how each feature contributes to the final prediction for the specific instance. LIME offers human-understandable explanations for individual predictions, enhancing model interpretability and transparency. 
It helps users, including domain experts and stakeholders, understand why a model made a specific prediction, increasing trust in complex machine learning models. LIME assists in identifying model biases, erroneous predictions, or areas where the model might be making decisions that don't align with expectations. 
The presented images have a color bar to facilitate the interpretation.
By providing local and understandable explanations for individual predictions, LIME helps users gain insights into complex machine learning models, promoting trust and understanding in their decision-making processes.
 
Figure 217. Applying LIME to the X-rays after classification.


Figure 215 displays the confusion matrix obtained from using the Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8). The TP were 259, the FP 3, the FN 11 and the TN 774.
 
Figure 215. The confusion matrix of applying Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).

A confusion matrix is a table that summarizes the performance of a classification model on a set of data. It is a valuable tool for evaluating the performance of a model and identifying areas for improvement. A confusion matrix is a table that is used to evaluate the performance of a classification model. It shows a summary of the predicted versus actual classes for a machine learning algorithm's predictions.
Figure 216 shows the ROC curve of applying Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8). It presented an AUC of 99.77%.
 
Figure 216. The ROC curve of applying Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).

Figure 217 shows some of the performance metrics calculated via MatLab of applying Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).
Performance metrics are essential for evaluating the effectiveness and accuracy of deep learning models. These metrics provide valuable insights into the strengths and weaknesses of a model, and help to identify areas for improvement.
Performance metrics in deep learning quantify how well a model performs on a given task, such as classification, regression, or clustering. These metrics help in assessing the model's accuracy, reliability, and effectiveness in making predictions or classifications.
 

Figure 217. The precision, recall, AUC and F1-score calculated via MatLab of applying Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).

The developed prototype model Pneumonet, utilized for detecting pneumonia via X-rays using the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), demonstrated strong performance. It achieved an accuracy of 98.7%, a recall rate of 95.9%, precision of 98.9%, an F1-score of 98.35%, specificity of 99.6%, and an area under the curve of 98.39%, as shown on Table 22.



Table 22. Performance metrics of the developed prototype on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) for pneumonia detection on X-rays.
Model	Accuracy	Precision	Recall	AUC	Specificity	F1-score
Developed prototype on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8)
98.7%	98.9%	95.9%	99.77%	99.6%	98.35%

This section intends to achieve #Objective_2, answering #Research_Question_3 and presenting #Hypothesis_5.









6.5.3	Application of the Developed Prototype Pneumonet on the Chest X-Ray Images (Pneumonia) Dataset
The chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) is structured into three primary directories (train, test, validation), each containing subfolders categorizing images into Pneumonia or Normal. In total, there are 5,863 JPEG X-ray images spanning these two categories. These anterior-posterior chest X-ray images originated from pediatric patients aged one to five at Guangzhou Women and Children’s Medical Center, collected as part of routine clinical procedures.
To ensure data quality, an initial screening process was conducted to filter out low-quality or indecipherable scans. Following this, two skilled physicians assessed and graded the diagnoses of the images before their incorporation into the training of the AI system. Furthermore, a third expert examined the evaluation set to minimize any potential grading discrepancies.
To guarantee the precision of the chest X-ray image analysis, an initial quality check involved examining all radiographs to exclude any scans deemed of low quality or unreadable. Subsequently, two experienced physicians meticulously evaluated and rated the diagnoses of these images before their inclusion in training the AI system. Additionally, to further reduce the likelihood of grading errors, a third expert scrutinized the evaluation set.
The application of the developed prototype Pneumonet on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) involved utilizing a specialized version of the AlexNet architecture specifically tailored for pneumonia detection tasks within X-ray images.
The dataset is organized into subsets for training, validation, and testing. It includes images categorized into Pneumonia or Normal classes.
The Pneumonet model, a customized version of the AlexNet architecture, is trained on the labeled X-ray images. This involved feeding the images through the network, adjusting weights and parameters to learn features indicative of pneumonia or normal conditions.
Preprocessing steps, such as resizing, normalization, and potentially augmentation, were applied to enhance model robustness and generalization.
The model's performance was evaluated using the validation set. Metrics like accuracy, precision, recall, and F1-score are computed. Fine-tuning may be done based on validation performance to improve model accuracy.
The final trained Pneumonet model was tested on a separate test set of X-ray images not seen during training to assess its performance on new, unseen data.
Techniques like Grad-CAM and LIME were employed to interpret and visualize the areas within the X-ray images that influenced the model's predictions.
The application of Pneumonet on chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) aims to contribute to accurate and efficient pneumonia detection from X-ray images. It has the potential to serve as a valuable tool in aiding healthcare professionals for timely and accurate diagnosis, improving patient care and outcomes.
By leveraging a specialized neural network architecture like Pneumonet on this dataset, the goal was to develop a robust and reliable system for pneumonia detection in X-ray images, ultimately contributing to advancements in healthcare technology and patient care.
Figure 221 displays some of the images of the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).
The dataset consists of numerous chest X-ray images labeled to denote the existence or lack of pneumonia. It was utilized in training deep learning models, especially convolutional neural networks, employing methods such as constructing models from the ground up. The objective was to develop algorithms capable of precisely recognizing pneumonia-related patterns in X-ray images. Publicly available datasets provide abundant labeled data accessible regardless of location or available resources. This availability promotes inclusive research practices and fosters collaboration within the field.
The Pneumonet allowed to show high accuracy in detecting pneumonia from X-rays. This research took in consideration several strategies in order to develop the model. The standard architecture of the AlexNet was modified and the model was fine-tunned in order to be suitable to detect pneumonia on X-rays.

. 
Figure 221. The number of images with and without pneumonia and some sample X-rays of the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).

Figure 222 shows the training progress of using the CPAlexNerV2 on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9). In MATLAB, the training progress window refers to a visual interface or display that showcases the ongoing training process of a machine learning or deep learning model. This window provides real-time updates and information regarding the model's training performance. The progress window continuously updates during the training process, showing key metrics like loss, accuracy, validation metrics, and other performance indicators. It often includes visual representations such as graphs or plots depicting the training progress over epochs or iterations. Graphs might display training and validation loss, accuracy, or other custom metrics. Some training progress windows offer options to pause, resume, or stop the training process, giving users control over model training. Users can  customize the displayed metrics or configure the appearance of the progress window to suit specific preferences or requirements. The window aids in monitoring the model's behavior, identifying issues like overfitting or underfitting, and analyzing the training dynamics.

 
Figure 222. Training progress of using the CPAlexNerV2 on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).

Figure 223 shows the resulting confusion matrix of using the CPAlexNerV2 on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9). The TP were 248, the FP 4, the FN 20 and the TN 771.
The confusion matrix is a fundamental tool for understanding the behavior and performance of classification models, providing detailed information about their predictions and guiding improvements in model accuracy and reliability.
The confusion matrix in the context of applying the Developed Prototype Pneumonet on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) serves as a critical assessment tool for understanding the model's classification performance.
It provides a detailed breakdown of the model's predictions versus the actual classes within the dataset, offering insights into the model's strengths and weaknesses.
By categorizing predictions into True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN), it aids in identifying specific types of errors made by the model.

 
Figure 223. Confusion matrix obtained of using the CPAlexNerV2 on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).

The confusion matrix helps in calculating accuracy, precision, recall, F1-score, and other performance metrics critical for assessing the model's effectiveness. Identifies areas for model improvement, such as reducing false positives or false negatives, by adjusting model parameters or data preprocessing techniques.
The confusion matrix is a fundamental tool in evaluating the Developed Prototype Pneumonet's performance on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9). It aids in understanding how well the model distinguishes between pneumonia and normal cases, guiding improvements to enhance its accuracy and reliability in medical diagnosis.
Figure 224 shows the ROC curve obtained, with an AUC of 98.04%. The ROC curve in the context of the application of the developed prototype Pneumonet on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) is a graphical representation used to evaluate the model's classification performance across various thresholds. It assesses the trade-off between the true positive rate (Sensitivity) and false positive rate (1 - Specificity) at different classification thresholds. Illustrates the model's ability to distinguish between pneumonia and normal cases by examining its performance across a range of thresholds. X-axis shows false positive rate (1 - Specificity), the proportion of false positives over actual negatives.
Y-axis or true positive rate (Sensitivity) provides the proportion of true positives over actual positives.
The AUC measures the overall performance of the model. Higher AUC values (closer to 1) indicate better discrimination between classes.
The point on the ROC curve closest to the top-left corner represents the optimal balance between sensitivity and specificity.
ROC curves facilitate the comparison of different models' performances on the same dataset. Helps in determining the appropriate threshold for the model based on the desired trade-offs between true positive and false positive rates.
In medical applications, like pneumonia detection, a higher AUC indicates a model with better ability to correctly identify patients with the condition while minimizing false diagnoses.
The ROC curve analysis in the context of the Developed Prototype Pneumonet on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) serves as a valuable tool in assessing the model's ability to discriminate between pneumonia and normal cases at various classification thresholds, aiding in the evaluation and optimization of the model for accurate medical diagnosis.




 
Figure 224. ROC curve obtained of using the CPAlexNerV2 on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).
 
Figure 225. AUC, precision, recall and F1-score obtained of using the CPAlexNerV2 on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).
Figure 226 shows some of the X-rays after applying classification. In MATLAB, the classify function is used to predict class labels for new data based on a pre-trained classification model. The function accepts as parameters the pre-trained classification model and the new data or observations to be classified.
The classify function takes a pre-trained classification model, such as a machine learning classifier or a deep neural network, and new data points as inputs. It uses the provided model to predict or classify the class labels for the new data. Returns the predicted class labels for the input data.
 
Figure 226. The X-rays after classification.

Figure 227 shows the use of grad-CAM on the classified images. Grad-CAM is a technique used in deep learning to visualize and understand the regions within an image that contribute significantly to a neural network's predictions, particularly in image classification tasks.In a CNN, the last convolutional layer captures high-level features before fully connected layers generate class scores. Grad-CAM computes the gradients of the predicted class score (logit) with respect to the activations of the last convolutional layer. It calculates the importance of each activation map by averaging the gradients spatially, giving more weight to the activations contributing more to the class score.
An heatmap is generated by combining the activation maps based on their importance weights, highlighting the regions significantly impacting the prediction. Provides visual explanations for the model's predictions, showing which parts of an image influenced its decision. Helps understand which regions the model focused on to make a particular prediction, enhancing trust and interpretability. Useful in object localization tasks, indicating where the model detected specific objects within an image. Compute the gradients of the predicted class score with respect to the activations of the last convolutional layer.Average the gradients spatially to get the importance weights for each activation map. Generate a heatmap by combining the activation maps based on their importance weights. Overlay the heatmap onto the original image, highlighting the regions that contributed most to the model's prediction.
By visualizing the heatmap overlaid on the original image, Grad-CAM provides insights into the specific regions within an image that influenced the CNN's prediction, aiding in model interpretability and understanding.
 
Figure 227. Application of grad-CAM on the classified images.

Figure 228 shows the use of LIME on the classified images after training. LIME is an interpretability technique used to explain the predictions of machine learning models, particularly complex models like deep neural networks, by generating local and human-interpretable explanations for individual predictions. Focuses on providing explanations for specific predictions rather than explaining the entire model's behavior. LIME is model-agnostic, meaning it can be applied to any machine learning model regardless of its complexity. Using the CPAlexNerV2 on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) and corresponding predictions, LIME fits an interpretable model  on the perturbed samples. This simpler model approximates the complex model's behavior around the specific instance. LIME provides insights by assessing the importance of different features in the interpretable model. It quantifies how each feature contributes to the final prediction for the specific instance.
LIME offers human-understandable explanations for individual predictions, enhancing model interpretability and transparency. It helps users, including domain experts and stakeholders, understand why a model made a specific prediction, increasing trust in complex machine learning models.  LIME assists in identifying model biases, erroneous predictions, or areas where the model might be making decisions that don't align with expectations.
 
Figure 228. Application of LIME on the classified images.


The developed prototype model Pneumonet, utilized for detecting pneumonia via X-rays using the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), demonstrated strong performance. It achieved an accuracy of 97.7%, a recall rate of 92.5%, precision of 98.4%, an F1-score of 96.97%, specificity of 99.5%, and an area under the curve of 98.04%, as shown on Table 22.
Table 22. Performance metrics of the developed prototype on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) for pneumonia detection on X-rays.
Model	Accuracy	Precision	Recall	AUC	Specificity	F1-score
Developed prototype on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9)	97.7%	98.4%	92.5%	98.04%	99.5%	96.97%

This section intends to achieve #Objective_2, answering #Research_Question_3 and presenting #Hypothesis_5.













6.6	Analysis of Related Work
Detection of pneumonia through X-rays using deep learning has been a significant development in medical imaging. Deep learning, a subset of artificial intelligence (AI), involves training neural networks to recognize patterns and features in data. In the context of pneumonia detection on X-rays, deep learning algorithms have shown promising results in aiding radiologists by automating the process of identifying pneumonia-related abnormalities.
The work of (Sharma & Guleria, 2023) shows a deep learning model employing VGG16 to detect and categorize pneumonia using two sets of chest X-ray images. When paired with Neural Networks (NN), the VGG16 model achieves an accuracy of 92.15%, a recall of 0.9308, a precision of 0.9428, and an F1-Score of 0.937 for the first dataset. Additionally, the NN-based experiment utilizing VGG16 is conducted on another CXR dataset comprising 6,436 images of pneumonia, normal cases, and COVID-19 instances. The outcomes for the second dataset indicate an accuracy of 95.4%, a recall of 0.954, a precision of 0.954, and an F1-score of 0.954.
The research findings demonstrate that employing VGG16 with NN yields superior performance compared to utilizing VGG16 with Support Vector Machine (SVM), K-Nearest Neighbor (KNN), Random Forest (RF), and Naïve Bayes (NB) for both datasets. Furthermore, the proposed approach showcases enhanced performance results for both dataset 1 and dataset 2 in contrast to existing models.
Figure 142 shows a schematization of the prototype developed by the authors to detect pneumonia from X-rays.

 

Figure 142. Schematization of the process used by (Sharma & Guleria, 2023) adapting the VGG-16.

In the analysis of (Reshan et al., 2023) a deep learning model is showcased to distinguish between normal and severe pneumonia cases. The entire proposed system comprises eight pre-trained models: ResNet50, ResNet152V2, DenseNet121, DenseNet201, Xception, VGG16, EfficientNet, and MobileNet. These models were tested on two datasets containing 5856 and 112,120 chest X-ray images. The MobileNet model achieves the highest accuracy, scoring 94.23% and 93.75% on the respective datasets. Various crucial hyperparameters such as batch sizes, epochs, and different optimizers were carefully considered when comparing these models to identify the most suitable one.
Figure 143 shows a Schematization of the process used by (Reshan et al., 2023), mentioning the input images, the data augmentation process the model training and classification as well as the performance metrics to evaluate the model.
 
Figure 143. Schematization of the process used by (Reshan et al., 2023).
To distinguish pneumonia cases from normal instances, the capabilities of five pre-trained CNN models namely ResNet50, ResNet152V2, DenseNet121, DenseNet201, and MobileNet have been assessed. The most favorable outcome is achieved by MobileNet using 16 batch sizes, 64 epochs, and the ADAM optimizer. Validation of predictions has been conducted on publicly accessible chest radiographs. The MobileNet model exhibits an accuracy of 94.23%. These metrics serve as a foundation for devising potentially more effective CNN-based models for initial solutions related to Covid-19 (Reshan et al., 2023).
The work of (Wang et al., 2023) introduce PneuNet, a diagnostic model based on Vision Transformer (VIT), aiming for precise diagnosis leveraging channel-based attention within lung X-ray images. In this approach, multi-head attention is employed on channel patches rather than feature patches. The methodologies proposed in this study are tailored for the medical use of deep neural networks and VIT. Extensive experimental findings demonstrate that our approach achieves a 94.96% accuracy in classifying three categories on the test set, surpassing the performance of prior deep learning models. 
Figure 143 shows a schematization of the process used by (Wang et al., 2023) to detect pneumonia from X-rays.
 
Figure 143. Schematization of the process used by (Wang et al., 2023), (a) Architecture of PneuNet and (b) the details of Transformer Encoder.
6.7	Analysis and Conclusions
The study aimed to enhance the accuracy of pneumonia detection from X-ray images, employing advanced techniques like grad-CAM and LIME. Additionally, was developed a prototype model Pneumonet and evaluated several pre-trained architectures ResNet-50, and AlexNet alongside the developed prototype on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7). 
Subsequently, was applied the developed prototype Pneumonet on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9) to assess its performance using various metrics.
Upon analyzing the results from the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), it was observed that the pre-trained models, particularly VGG-16 and ResNet-50, displayed commendable performance in pneumonia detection from X-rays. Notably, the developed prototype Pneumonet also showcased promising results, demonstrating competitive performance alongside these established architectures.
Moving to the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9), areas where the developed prototype was specifically applied, the performance metrics demonstrated significant accuracy and reliability in detecting pneumonia. The prototype showcased resilience when tested on varied datasets, hinting at its capability to generalize and perform effectively in practical real-world scenarios.
The study highlights the efficacy of both established deep learning architectures like AlexNet and ResNet-50 and the novel developed prototype Pneumonet utilizing grad-CAM and LIME techniques for pneumonia detection from X-ray images. The prototype's consistent performance across multiple datasets underscores its viability as a reliable tool for accurate pneumonia identification, showing promise for practical implementation in clinical settings.
Table 17 shows the confusion matrices of the research concerning detection of pneumonia on X-rays, displaying the obtained using the ResNet-50, the AlexNet, and the developed prototype on the the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) and the obtained from the developed prototype on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and on the Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays) (Sait, 2021) (Dataset 3). 


Confusion matrices play a pivotal role in evaluating the performance of various deep learning models in detecting pneumonia from X-ray images. Confusion matrices provide a comprehensive view of a model's performance, detailing correct classifications (true positives and true negatives) as well as misclassifications (false positives and false negatives).
By comparing confusion matrices of different models, one can discern which model performs better in distinguishing pneumonia and normal X-rays. They offer insights into specific types of errors made by models (e.g., false positives or false negatives), which aids in identifying patterns or areas for improvement.
Understanding the confusion matrix allows for model refinements, such as adjusting thresholds or modifying features, to reduce misclassifications.
In medical applications, minimizing false positives (misclassifying a normal X-ray as pneumonia) and false negatives (missing pneumonia cases) is crucial for accurate diagnosis and treatment planning.
Confusion matrices help evaluate model performance across diverse datasets, indicating how well a model generalizes to different X-ray collections.
Metrics like accuracy, sensitivity, specificity, precision, and F1-score, computed from confusion matrices, offer nuanced insights into model performance and suitability for clinical use.
In essence, confusion matrices serve as a foundational tool for comprehensively evaluating and comparing the effectiveness of various deep learning models in pneumonia detection from X-ray images. Their insights are crucial in optimizing models, refining performance, and ensuring reliable and accurate diagnoses in healthcare applications.










Table 17. Confusion matrices of the analysis performed, namely the obtained using ResNet-50, AlexNet, and the developed prototype on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) and the obtained from the developed prototype on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).

ResNet-50
 	AlexNet
 
Pneumonet on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7)
 
Pneumonet on the Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8).
 	CPAlexNerV2 on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).
 	

Table 18 shows the ROC curves used in this research for the detection of Covid-19 on X-rays, mentioning the AUC for the VGG-16, VGG-19, ResNet-50, AlexNet, and the developed prototype on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7),the obtained from the developed prototype on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8)  and on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).
ROC curves are crucial in assessing the performance of different deep learning models in detecting pneumonia from X-ray images due to several key reasons. ROC curves showcase how well models can differentiate between pneumonia and normal X-ray images across various threshold values. They illustrate the balance between sensitivity (true positive rate) and specificity (true negative rate) at different classification thresholds.
Comparing multiple ROC curves allows for a clear understanding of which model exhibits superior discrimination power in pneumonia detection. ROC curves aid in identifying the threshold that optimizes the trade-off between sensitivity and specificity, essential in clinical decision-making.
The Area Under the ROC Curve (AUC) serves as a comprehensive metric summarizing a model's discriminatory ability. Higher AUC values indicate better performance.
Effective models with higher AUC values ensure fewer missed pneumonia cases (false negatives) and fewer misdiagnosed normal cases (false positives), impacting patient care positively.
ROC curves help assess how well models generalize across different datasets, indicating their robustness in real-world applications.
A model with a higher AUC on the ROC curve instills more confidence in its reliability and accuracy for pneumonia detection tasks.
ROC curves assist in fine-tuning model parameters or selecting the most suitable model based on its discriminatory performance.
Insight from ROC curves guides iterative model improvements, ensuring continuous enhancement in detection accuracy.
In summary, ROC curves are invaluable in comprehensively assessing and comparing the performance of diverse deep learning models for pneumonia detection from X-ray images. Their insights aid in selecting optimal models, optimizing thresholds, and ensuring reliable and accurate diagnoses, significantly impacting healthcare outcomes.

Table 18. ROC curves of the analysis performed, namely the obtained using the ResNet-50, AlexNet, and the developed prototype on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), the obtained from the developed prototype on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8) and the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9)
.
ResNet-50
 
AUC= 0.9662	AlexNet
 
AUC= 0.9749	Developed prototype on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7).
 
AUC= 0.9839
Developed prototype on the Labeled Optical Coherence Tomography an =d Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8)
 
AUC = 0.9977	Developed prototype on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).
 ~
AUC = 0.9804	


The study focused on improving the identification of pneumonia from X-ray images by leveraging various techniques and models. Grad-CAM and LIME was used, which are methods for visualizing and interpreting deep learning models, to gain insights into how these models make predictions based on X-ray data. These techniques help in understanding which parts of the X-ray images are crucial for the model's decision-making process.
Moreover, the study involved the development of a novel prototype Pneumonet model tailored for pneumonia detection from X-rays. This research integrated insights from grad-CAM and LIME, along with potentially novel architectural features or adaptations specific to the characteristics of pneumonia X-ray images.
To benchmark the performance of the prototype, a comparison was performed against well-known pre-trained models like ResNet-50, and AlexNet. This comparison was carried out using the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), where all these models were evaluated to determine their effectiveness in accurately detecting COVID-19 from X-ray scans.
The findings from the the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7) showcased that established models like AlexNet and ResNet-50 exhibited strong performance in identifying pneumonia patterns within X-ray images. However, the novel prototype developed also demonstrated promising results, competing well with these established architectures.
The evaluation expanded beyond the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), as the developed prototype was also tested on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8)  and on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9). Across these additional datasets, the prototype consistently showed high accuracy and reliability in pneumonia detection, indicating its robustness and potential for generalization across different datasets and potentially diverse image characteristics.
The study not only compared the performance of well-known deep learning models but also introduced a novel prototype specifically designed for pneumonia detection from X-ray images. The consistent and strong performance of this prototype across multiple datasets suggests its potential as a dependable tool for accurate and efficient identification of pneumonia in clinical settings. This innovative approach might pave the way for more effective diagnostic tools in the field of medical imaging for infectious diseases like pneumonia.



Table 19. The performance metrics obtained using the ResNet-50, AlexNet, and the developed prototype on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), the metrics obtained from the developed prototype on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) and the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9).

Model	Accuracy	Precision	Recall	AUC	Specificity	F1-score
ResNet-50	91.3%	82.1%	86.6%	96.6%	93%	84.2%
AlexNet	91.1%	78.3%	92.9%	97.4%	90.5%	84.9%
Developed prototype on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7)	96.7%	97.8%	89.7%	98.39%	99.3%	93.12%
Developed prototype on the labeled Optical Coherence Tomography and Chest X-Ray Images for Classification dataset (Kermany, 2018) (Dataset 8)	98.7%	98.9%	95.9%	99.77%	99.6%	98.35%
Developed prototype on the chest X-Ray Images (Pneumonia) dataset (Mooney, 2018) (dataset 9)	97.7%	98.4%	92.5%	98.04%	99.5%	96.97%


![image](https://github.com/user-attachments/assets/e0efa003-60b5-4110-86cb-236952d170a5)





![image](https://github.com/user-attachments/assets/be1b6273-ddcf-44e2-b75c-4d3bdb4e4c6d)





The ResNet-50 showed and accuracy of 91.3% and the developed prototype of 96.7% on the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7), the presented solution showed the highest AUC. The developed prototype was tested in 3 different datasets presenting AUC of 98.39%, 97.77% and 98.4% respectively. The results of the presented solution were compared with other known models like the ResNet-50 and AlexNet on t the Chest X-ray (Covid-19 & Pneumonia) (Prashant, 2020) (Dataset 7). This research presents a solution to be complemented with the traditional processes of detecting Covid-19 facilitating the work of the physicians.




![image](https://github.com/user-attachments/assets/3e5b43c9-249f-4e31-aad7-f5f1ab949402)

