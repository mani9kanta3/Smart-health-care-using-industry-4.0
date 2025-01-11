# Terrain Classification and Identification for Autonomous Vehicles

This project focuses on enhancing the safety and adaptability of autonomous vehicles by enabling reliable identification and classification of off-road terrains. By leveraging the YOLOv8 instance segmentation model, the project achieves accurate terrain detection, supporting autonomous vehicles in making informed navigation decisions.

The model was trained using the CAT-CaVS Traversability Dataset, which includes diverse off-road scenarios. Key features include the use of advanced convolutional neural network (CNN) architectures, real-time image segmentation, and precise delineation of drivable regions.

This research lays the groundwork for integrating terrain analysis with practical hardware implementations, paving the way for further advancements in autonomous vehicle technologies.

## Overview
Off-road terrain plays a critical role in determining the performance and safety of autonomous vehicles. This project uses YOLOv8, a state-of-the-art CNN architecture, for instance segmentation to classify drivable and non-drivable regions in complex terrains. The model is trained on annotated off-road images and demonstrates high accuracy in detecting drivable regions.

## Features
- **Instance Segmentation:** Differentiates drivable regions from obstacles such as rocks, trees, and rough terrain.
- **Real-Time Analysis:** Processes images to identify paths in off-road scenarios with high precision.
- **Preprocessing Pipeline:** Includes resizing, augmentation, and annotation refinement for robust model training.
- **Scalable Framework:** Can be extended to include path planning and obstacle avoidance.

## Data Collecting and Cleaning

**Dataset:**

The Dataset used for the Experiment to identify the Terrain is Cat: CAVS Traversability Dataset which consists 
of 3.45k Off-Road Terrain images which were collected by Mississippi State University and is stated in. The 
data collection was performed near HPCC (High Performance Company Collaboratory) of Mississippi State 
University. The data is collected from three trails i.e., main trail with a length of 0.64 km , powerline trail with 
a length of 0.82 km, brownfield trail with a length of 0.21 km . The images were collected considering the 
required light exposure with different filters compatible with Sekonix SF3325-100 camara model.

source: https://www.cavs.msstate.edu/resources/autonomous_dataset.php#:~:text=CaT%3A%20CAVS%20Traversibility%20Dataset&text=This%20dataset%20contains%20a%20set,to%20drive%20through%20the%20terrain

**Data Cleaning Process**

 In the final dataset, we retained only the most relevant features for the project.

    1. Removed Duplicative Copies:
      - Removed similar images to train diverse off-road scenarios for better accuracy
    2. Image Resize:
      - The image dimensions were resized to 640x640 pixels to ensure uniformity across all images.  
    3. Data Augmentation:
      -  Applied color adjustments such as brightness, contrast, and saturation changes, along with geometric transformations like rotation and flipping, to enhance dataset diversity and improve model generalization.
This process ensured a clean and concise dataset focused solely on features relevant to the research.

## Annotation Refinement:

- **Platform:** Used Roboflow for efficient and user-friendly manual labeling.
- **Focus:** Labeled drivable regions in images, distinguishing them from obstacles and non-drivable terrain.
- **Workflow:**
  - Uploaded raw images to Roboflow.
  - Outlined drivable regions and assigned specific color-coded masks.
  - Performed quality checks to ensure accuracy and consistency.
- **Refinement:** Iterative corrections were made to improve annotation precision.
- **Dataset:** Selected 1.24k high-quality annotated images from the 3.45k dataset for training.
- **Outcome:** Enhanced data quality improved the YOLOv8 model's ability to accurately detect drivable regions.
- **Advantages:**
  - Ensured precise and clean data for training.
  - Improved segmentation accuracy and model performance.

## Model Development

**Training Parameters**

- **Model Architecture:** YOLOv8 instance segmentation model, a state-of-the-art Convolutional Neural Network (CNN) for object detection and segmentation.
- **Training Configuration:**
  - Epochs: 100
  - Image Size: 640 pixels
  - Batch Size: 16
  - Loss Functions: Box loss, segmentation loss, and class loss were monitored during training for both training and validation data.
  - Feature extraction across 21 stages for refined segmentation performance.

**Validation Parameters**

- Validation data constituted 8% of the total dataset to evaluate model generalization.
- Performance metrics such as precision, recall, and F1-score were monitored to measure the effectiveness of the trained model.
- The confusion matrix was analyzed to identify misclassifications and refine training where necessary.

**Testing Parameters**

- Testing data formed 4% of the dataset and was used to validate the model’s real-world applicability.
- Predicted outputs were visually compared with ground truth annotations, showcasing the model's ability to segment drivable regions and differentiate obstacles.


## Result

**Performance Metrics:**

- **Precision:** It measures the proportion of true positive predictions out of all positive predictions, indicating the accuracy of the model's positive classifications.
- **Recall:** It measures the proportion of true positive predictions out of all actual positive cases, indicating the model's ability to identify all relevant instances.
- **F1-Score:** Harmonic mean of precision and recall, providing a balanced measure of a model's accuracy that considers both false positives and false negatives.
- **mAP@50 (Mean Average Precision at IoU 0.50):** This metric evaluates the detection model's performance by calculating the average precision (AP) for all classes at a single Intersection over Union (IoU) threshold of 0.50, which is considered a lenient criterion for overlap between predicted and ground truth bounding boxes.
- **mAP@50-95 (Mean Average Precision at IoU 0.50 to 0.95):** This metric provides a more comprehensive evaluation by averaging AP over multiple IoU thresholds ranging from 0.50 to 0.95 (in steps of 0.05), offering a stricter and more detailed measure of the model's detection accuracy across varying overlap levels.

**Loss Function:**

- **Box loss (box_loss):** It measures the discrepancy between the predicted bounding boxes and the ground truth bounding boxes, typically using metrics like Smooth L1 loss or IoU-based loss to optimize the model's localization accuracy.
- **Segmentation Loss (seg_loss):** It quantifies the difference between the predicted and ground truth segmentation masks, commonly using metrics like Cross-Entropy Loss or Dice Loss, to optimize pixel-wise classification accuracy.
- **Classification Loss (cls_loss):** It measures the error in predicting class labels for objects, typically using Cross-Entropy Loss, Focal Loss, or other variations to enhance the model's accuracy in distinguishing between categories.
- **Distribution Focal Loss (dfl_loss):** It is used in object detection to refine bounding box regression by learning the distribution of precise locations, improving the accuracy of boundary predictions for objects.

![results](https://github.com/user-attachments/assets/10d85de8-efe1-4226-95be-79f24aa89bb9)


## Conclusion

This project presents a comprehensive approach to off-road terrain identification and analysis, utilizing the YOLOv8 instance segmentation architecture. By training the model on the CAT-CaVS Traversability Dataset, which includes diverse off-road terrain images, the system achieves high accuracy in detecting drivable regions and differentiating them from obstacles such as trees, rocks, and other environmental features. The YOLOv8 model's advanced capabilities in instance segmentation allow it to deliver refined and precise results, making it well-suited for real-world applications in autonomous vehicles.

The study emphasizes the importance of computational techniques in enabling autonomous vehicles to adapt to challenging terrains, a critical factor for ensuring safety and efficiency in navigation. The current implementation focuses on computational aspects, providing a robust foundation for further research into integrating hardware such as sensors, accelerometers, and LIDAR for real-time terrain analysis.

Future work will aim to develop a more generalized and sophisticated hardware model capable of handling diverse terrains while dynamically adjusting vehicle maneuverability. The trained model can also be extended to incorporate additional functionalities like path planning, object detection, and GPS integration, moving closer to the realization of fully autonomous vehicles. This research motivates ongoing advancements in autonomous navigation systems, contributing to the broader adoption of Industry 4.0 technologies and improving the reliability and safety of self-driving vehicles across various environments.







