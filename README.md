# ProtoTrees: Neural Prototype Trees for Interpretable Fine-grained Image Recognition - Chest X-ray Disease Detection

# Abstract
At present, patients experience a two-day delay in receiving their chest X-ray results, causing anxiety and treatment delays. To expedite this process, we aim to develop a machine learning model for disease identification from chest X-rays. While, most current models lack interpretability, leading to traceability risks in critical medical decisions. Our project focuses on identifying Pleural Effusion, Cardiomegaly, and normal findings from 9,000 chest X-ray images using an interpretable machine learning model, ProtoTree. We evaluate the accuracy and interpretability trade-off by comparing the ProtoTree model with a baseline ResNet50 model. While the ProtoTree model achieves a comparable accuracy to the ResNet50 model (62% vs. 72%), it provides a decision tree that explains the decision making process, enhancing interpretability. Our future work aims to improve accuracy, generalize to more diseases, and address class imbalance challenges.

# Introduction
A chest X-ray is a crucial diagnostic tool used in the early detection of various diseases in the chest region affecting the heart, lungs, and bones. Typically, a radiologist interprets the X-ray images, with results taking one to two days to be processed, leading to heightened patient anxiety. In a healthcare setting, having fast and accurate diagnostic tools is crucial for timely treatment decisions and patient care. While deep learning models like CNNs offer high accuracy in image recognition tasks, they often lack transparency in their decision-making process. This opacity can be a significant limitation in healthcare, where understanding why a model makes a particular diagnosis is as important as the accuracy of the diagnosis itself.

Our project introduces the Neural Prototype Tree, or ProtoTree, as an interpretable method for image recognition in chest X-rays. This model provides interpretable insights into how it arrives at a diagnosis, making it a valuable tool for improving decision-making in healthcare settings. It combines a Convolutional Neural Network (CNN) with a soft decision tree, trained using a cross-entropy loss function. Unlike conventional black box models, the ProtoTree integrates interpretability directly into its structure, addressing the growing need for models that can be easily understood by non-specialists. The ProtoTree aims to assist overburdened radiologists and physicians unfamiliar with X-ray interpretation, either by providing additional support in decision-making or by highlighting details that may have been overlooked.

Our study focuses on the trade-off between accuracy and interpretability by comparing the performance of a baseline ResNet50 model with our interpretable ProtoTree model. By evaluating both models, we aim to determine the optimal approach for healthcare diagnosis, balancing accuracy with timely results.

# Data
The dataset we are utilizing is the NIH Chest X-ray from Kaggle, containing 112,120 chest X-rays with disease labels from 30,805 unique patients. However, for our study, we have narrowed down our training dataset to three classes we are focusing on. We used a 30-70 test-train split on 1093 images for Cardiomegaly, and 3000 for Effusion and No Finding each. This indicates potential challenges in modeling due to class imbalance.

For our study, the dataset comprises three distinct categories: No Findings, Cardiomegaly, and Pleural Effusion. 
- Cardiomegaly, characterized by an enlarged heart, typically arises as a consequence of an underlying pathological process and can manifest in various forms of primary or acquired cardiomyopathies. This condition can affect the right, left, or both ventricles or the atria of the heart. We selected this class to aid in the identification of larger heart diseases, as Cardiomegaly often serves as a primary indicator of other heart conditions.

-  Pleural Effusion is defined as the accumulation of fluid in the pleural cavity, the space between the lungs and the chest wall. This condition can be caused by various underlying diseases or conditions, including congestive heart failure, cancer, pneumonia, and pulmonary embolism. Of these, lung cancer is particularly noteworthy as a common cause of malignant Pleural Effusion.

- The "No Findings" class indicates instances where neither condition nor any other chest disease is detectable in the X-ray image. Our objective was to gain insights into early signs of heart and lung conditions, which could serve as primary indicators of more serious chest illnesses. Identifying these conditions early is crucial for preventing them from developing into more severe problems.

# Experimental Design
The experiments are divided into three major parts: data preprocessing, modeling, and model evaluation.

In this experiment, we aim to investigate the tradeoff between accuracy and interpretability in machine learning models. To establish a fair comparison, we will first assess the performance of a baseline ResNet50 model against our ProtoTree model. This comparison will be based on four evaluation metrics: F1 scores, accuracy, precision, and recall, analyzed using a 30-70 test-train split. Additionally, we will evaluate the models' ability to provide local and global explanations, which serves as an interpretability metric between the two models.

Our experiment aims to showcase the tradeoff between accuracy and interpretability in machine learning models. By comparing the ProtoTree model, which provides both accuracy and interpretability metrics, with the baseline ResNet50 model, which only provides accuracy metrics, we aim to determine which model is more valuable based on the perspective of the end user.

# Data Preprocessing
In this report, data preprocessing involves several key steps. First, all images are resized to 224224. Image transformation is then performed to adjust color, with parameters tuned based on the ProtoTree method and other relevant papers detailed.

# Modeling
We frame our project as a classification problem. Given an input image, our model generates a class probability distribution over the labels. We train the model by minimizing the cross-entropy objective.  

### Baseline Model Notebook:

**Functions Contained:**
1. `get_nih(augment, train_dir, project_dir, test_dir, img_size)`: This function prepares the NIH Chest X-ray dataset for training, projection, and testing. It handles image resizing, normalization, and augmentation if specified.
2. `get_dataloaders(batch_size)`: This function creates data loaders for the training, projection, and testing sets, utilizing the prepared dataset from `get_nih`.

**Functionality:**
- It employs an untrained ResNet50 model as a baseline for performance comparison with the ProtoTree model.
- ResNet50 is chosen for its use of skip connections and deep residual learning approach, which are known to deliver state-of-the-art performance in image recognition tasks.
- Depending on the chosen number of classes, the baseline model can adjust the number of nodes in the final layer of the neural network. In this case, the model is configured to have 3 nodes in its final layer.

### ProtoTree Model Summary:

**Functions Contained:**
1. `run_ensemble()`: Trains and evaluates the ProtoTree model ensemble, saving the trained trees and their accuracies.
2. `run_tree(args)`: Trains a single tree in the ensemble and returns the trained tree, pruned tree, pruned and projected tree, original test accuracy, pruned test accuracy, pruned projected test accuracy, project info, evaluation info for sample max, evaluation info for greedy, and fidelity info.
3. `analyse_ensemble(log, all_args, test_loader, device, trained_orig_trees, trained_pruned_trees, trained_pruned_projected_trees, orig_test_accuracies, pruned_test_accuracies, pruned_projected_test_accuracies, project_infos, infos_sample_max, infos_greedy, infos_fidelity)`: Analyzes the ensemble with more than one tree, evaluating the performance and providing insights into the ensemble's behavior.

**Functionality:**
- The ProtoTree model is a combination of a Convolutional Neural Network (ResNet50) and a soft binary decision tree structure.
- An input image is forwarded through the convolution network, resulting in feature maps that serve as input to the binary tree.
- The tree consists of a set of leaf nodes and edges, where the probability of routing a sample through the right edge is calculated using a similarity function.
- The final prediction is the product of all the probabilities along the route the image traverses within the tree.
- The `run_ensemble()` function trains and evaluates the ProtoTree model ensemble, saving the trained trees, their accuracies, and various evaluation metrics for analysis.
- The `run_tree(args)` function trains a single tree in the ensemble, returning the trained tree, pruned tree, pruned and projected tree, and various evaluation metrics.
- The `analyse_ensemble()` function analyzes the ensemble with more than one tree, evaluating its performance and providing insights into the ensemble's behavior.

### Hyper-Tuning Notebook Summary:

**Functions Contained:**
1. `get_nih(augment, train_dir, project_dir, test_dir, img_size)`: Prepares the NIH dataset for training, projection, and testing with the specified augmentation and image size.
2. `torch.cuda.is_available()`: Checks if CUDA is available and sets the device accordingly.
3. Loading data using `get_dataloaders(batch_size=32)`.
4. Loading the ProtoTree model from a saved state.
5. Evaluation of the ProtoTree model on the test set, including prediction and confusion matrix generation.

**Functionality:**
- The notebook hyper-tunes the ProtoTree model using the NIH dataset.
- It prepares the dataset and model for training and evaluation.
- It loads the model and optimizer states from saved checkpoints.
- It evaluates the model on the test set, generating predictions and a confusion matrix to analyze the model's performance.

**Note:** Some parts of the code, such as loading the model and optimizer states, and evaluating the model, are commented out. These sections may need to be uncommented to run the complete hyper-tuning process.

### Model Evaluation Notebook Summary:

**Functions Contained:**
1. Various utility functions imported from different modules (`util.log`, `util.args`, `util.data`, `util.init`, `util.net`, `util.visualize`, `util.analyse`, `util.save`, `prototree.train`, `prototree.test`, `prototree.prune`, `prototree.project`, `prototree.upsample`) for initializing, training, testing, pruning, and projecting the ProtoTree model.
2. `get_nih(augment, train_dir, project_dir, test_dir, img_size)`: Prepares the NIH dataset for training, projection, and testing with the specified augmentation and image size.
3. `get_dataloaders(batch_size)`: Prepares the data loaders for training, projection, and testing.

**Functionality:**
- The notebook implements the ProtoTree model for classification tasks on the NIH dataset.
- It provides functions for training, testing, pruning, and projecting the ProtoTree model.
- It loads a pretrained ResNet50 model and evaluates it on the test set, generating a confusion matrix and classification report.
- It also loads and evaluates the trained ProtoTree model on the test set, generating a confusion matrix and classification report.
- Validation curves for the ProtoTree model, showing epoch vs. accuracy and epoch vs. loss, are also plotted.

**Note:** Some parts of the code, such as loading the ProtoTree model and generating validation curves, are commented out. These sections may need to be uncommented to run the complete evaluation and analysis of the ProtoTree model.


## Model Paper
A ProtoTree is an intrinsically interpretable deep learning method for fine-grained image recognition. It includes prototypes in an interpretable decision tree to faithfully visualize the entire model. Each node in our binary tree contains a trainable prototypical part. The presence or absence of this prototype in an image determines the routing through a node. Decision making is therefore similar to human reasoning.

We used the following repo to understand how to utilize the Prototree Model for Chest X-ray detection.
(https://openaccess.thecvf.com/content/CVPR2021/html/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.html). 

## Prerequisites

### General
* Python 3
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.5 and <= 1.7!
* Optional: CUDA

### Required Python Packages:
* numpy
* pandas
* opencv
* tqdm
* scipy
* matplotlib
* requests (to download the CARS dataset, or download it manually)
* gdown (to download the CUB dataset, or download it manually)

## Running the Main_tree.py

The folder `preprocess_data` contains python code to download, extract and preprocess these datasets. 

A ProtoTree can be trained by running `main_tree.py` with arguments. An example: `main_tree.py --epochs 100 --log_dir ./runs/protoree_NIH_chest_Xrays --dataset NIH_Chest_Xrays --lr 0.001 --lr_block 0.001 --lr_net 1e-5 --num_features 256 --depth 9 --net resnet50_inat --freeze_epochs 30 --milestones 60,70,80,90,100` To speed up the training process, the number of workers of the [DataLoaders](https://github.com/M-Nauta/ProtoTree/blob/main/util/data.py#L39) can be increased by setting `num_workers` to a positive integer value (suitable number depends on your available memory).

Check your `--log_dir` to keep track of the training progress. This directory contains `log_epoch_overview.csv` which prints per epoch the test accuracy, mean training accuracy and the mean loss. File `log_train_epochs_losses.csv` prints the loss value and training accuracy per batch iteration. File `log.txt` logs additional info. 

The resulting visualized prototree (i.e. *global explanation*) is saved as a pdf in your `--log_dir /pruned_and_projected/treevis.pdf`. NOTE: this pdf can get large which is not supported by Adobe Acrobat Reader. Open it with e.g. Google Chrome or Apple Preview. 

To train and evaluate an ensemble of ProtoTrees, run `main_ensemble.py` with the same arguments as for `main_tree.py`, but include the `--nr_trees_ensemble` to indicate the number of trees in the ensemble. 

### Conclusion:

In summary, we successfully developed an interpretable machine learning model for disease identification from chest X-rays. The ProtoTree model, although slightly less accurate (62%) than the ResNet50 model (72%), offers invaluable interpretability that is crucial for high-stakes decision-making in healthcare. However, there are key areas for future work and challenges that need to be addressed.

One significant future direction is to improve the overall accuracy of the model. While interpretability is essential, accuracy is paramount in medical decision-making. One approach is to train a ResNet50 model from scratch on a larger, more diverse dataset of chest X-ray images. Alternatively, we could explore using pre-trained models specifically trained on chest X-ray data to enhance the model's proficiency in chest X-ray detection.

Another challenge is the class imbalance, particularly concerning Cardiomegaly. Collecting more images related to Cardiomegaly could help mitigate this issue and improve the model's performance on this specific condition. Additionally, we aim to generalize the model to identify more types of diseases beyond Pleural Effusion, Cardiomegaly, and normal findings. This will require further experimentation and hyperparameter tuning to ensure the accuracy and reliability of the model across different disease categories.

In conclusion, while we have made significant progress in developing an interpretable model for disease identification from chest X-rays, there are ongoing challenges and opportunities for improvement. By addressing these challenges and expanding the model's capabilities, we can enhance its utility in clinical practice and improve patient outcomes.

