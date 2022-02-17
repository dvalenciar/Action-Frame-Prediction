
<h1 align="center">
  <br>
Action-Conditioned Frame Prediction Without Discriminator
  <br>
 </h1>
 
   <p align="center">
    • David Valencia, Henry Williams, Bruce MacDonald, Ting Qiao •
   </p>
   
   <p align="center">
    • Centre for Automation and Robotic Engineering Science, University of Auckland •
   </p>
<h4 align="center">Official repository of the paper(https://link.springer.com/chapter/10.1007/978-3-030-95467-3_24)</h4>

## Prerequisites

|Library         | Version (TESTED) |
|----------------------|----|
| Python | 3.8|
| torch | 1.7.1+cu101|
| numpy | 1.19.4|
| PIL |  7.0.0 |
| tqdm|  4.54.0|
| matplotlib|  3.3.3|
| gym| 0.17.3|

## General Overview

Our model consists of two networks: an encoder E and a Generator G. No extra discriminator is needed in our proposal since the encoder here also plays the role of a discriminator. Not having an extra discriminator makes our network more stable. (More info and details of the architecture in our paper)

![](https://github.com/dvalenciar/Action-Conditioned-Frame-Prediction-Without-Discriminator/blob/main/Read_Img_Files/image_net.png)


## Sample Images from Datasets

We have two datasets available in this repository. ***Car_Racing Dataset*** and ***Two_Cubes Dataset***. We have collected and standardized each of the images that compose these datasets. Each dataset includes input images, target images and actions. Examples of frames from the two used datasets are present below. 

![](https://github.com/dvalenciar/Action-Conditioned-Frame-Prediction-Without-Discriminator/blob/main/Read_Img_Files/Example_of_Data.png)

## Repository Organization

* **/Car_Racing**
  - **/Preprocessed_Data** ----> directory containing the collected data from Gym; data cropped, sorted and normalized, separated into Input and Target images

* **/Two_Cubes**
  - **/DataSet**           ----> directory containing the raw data divided into episodes
  - **/Preprocessed_Data** ----> directory containing the preprocessed data, sorted and normalized, divided into Input and Target images

## How to run the code

To train the model from scratch, please, first clone this repository in your local workstation; This will download the dataset and the necessary files.

**Training Two Cubes Dataset**

To train the model with the Two_Cubes dataset, please run:

  ```
  python3 Two_Cubes/Frame_Prediction_Intro_VAE_V4_pytorch.py
  ```
  
**Training Car Racing Dataset**

To train the model with the Car_Racing dataset, please run:

  ```
  python3 Car_Racing/Frame_Prediction_Intro_VAE_Network.py

  ```

These scripts will load the data from Preprocessed_Data folder and train the model for 10K epochs (Two_Cubes) and 5k epochs (Car_Racing). Also, _/Images_Result_ folder and _/Model_Saved_ folder will be created automatically to save some images samples and checkpoints respectively.

**Evaluate the model**

To evaluate the performance of the model ***after training***, please run the following commands. 

- Car_Racing dataset

  ```
  python3 Car_Racing/Evaluate_Model.py

  ```
- Two Cubes Dataset
   ```
  python3 Two_Cubes/Evaluate_model_pytorch.py

  ```

## Results

Below is a sample of the results obtained with our model for the two datasets. Additionally,  we present a comparison against VAE  (a more in-depth analysis in our article)
![](https://github.com/dvalenciar/Action-Conditioned-Frame-Prediction-Without-Discriminator/blob/main/Read_Img_Files/Results.png)


## Citation
If you use either the code or data in your paper, please kindly star this repo and cite our paper

Cite this paper as:

Valencia D., Williams H., MacDonald B., Qiao T. (2022) Action-Conditioned Frame Prediction Without Discriminator. In: Nicosia G. et al. (eds) Machine Learning, Optimization, and Data Science. LOD 2021. Lecture Notes in Computer Science, vol 13163. Springer, Cham. https://doi.org/10.1007/978-3-030-95467-3_24

  ```
@inproceedings{valencia2021action,
  title={Action-Conditioned Frame Prediction Without Discriminator},
  author={Valencia, David and Williams, Henry and MacDonald, Bruce and Qiao, Ting},
  booktitle={International Conference on Machine Learning, Optimization, and Data Science},
  pages={324--337},
  year={2021},
  organization={Springer}
}
  ```


## Contact
Please feel free to contact us or open an issue if you have questions or need additional explanations.

######  The released codes are only allowed for non-commercial use.
