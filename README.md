# Terrain Classification and Identification for Autonomous Vehicles

This project focuses on enhancing the safety and adaptability of autonomous vehicles by enabling reliable identification and classification of off-road terrains. By leveraging the YOLOv8 instance segmentation model, the project achieves accurate terrain detection, supporting autonomous vehicles in making informed navigation decisions.

The model was trained using the CAT-CaVS Traversability Dataset, which includes diverse off-road scenarios. Key features include the use of advanced convolutional neural network (CNN) architectures, real-time image segmentation, and precise delineation of drivable regions.

This research lays the groundwork for integrating terrain analysis with practical hardware implementations, paving the way for further advancements in autonomous vehicle technologies.

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


## Confusion Matrix

![confusion_matrix_normalized](https://github.com/user-attachments/assets/ea2978b8-f20c-4e1b-96f6-fae568f9685c)



