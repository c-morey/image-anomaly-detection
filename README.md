# image-anomaly-detection
This repository provides 2 approaches to detect anomalies within an image dataset.

-------------------------------------------------------------------------------
### Project Guidelines
* Repository: ``` image-anomaly-detection ```
* Duration: ``` 10 days```
* Deadline: ```13-10-21```
* Mission: ``` Image Anomaly Detection```


### Explanation of Approaches
**1. Cumulative Image + Numpy Masking**
  - cumulative_images.py : This file creates a cumulative image based on given dataset. It converts it into a binary image, draws the contours, and saves it in both train and test dataset.
  - train_dice.py : This file trains the dices given during cumulative image creation (train dataset). It then outputs the standard deviation and average for each contour.
  - find_defect.py: This file is the last step to detect the anomalies. With numpy masking method, it will measure each contours' standard deviation and average and compare if they are within given threshold and magin. If not, they are outputted as an anomaly.

**2. Isolation Forest**
  - isolation_forest.py : Isolation Forest is an outlier detection technique that identifies anomalies instead of normal observations. The advantage of it is that it can be scaled up to handle large, high-dimensional datasets.


### Prerequisites
* Python3


### Tools & Libraries
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [image_similarity_measures](https://pypi.org/project/image-similarity-measures/)
* [CV2](https://pypi.org/project/opencv-python/)
* [scikit-learn](https://scikit-learn.org/stable/)


### Usage
```streamlit run streamlit.py```

### Contributors
* [Ceren Morey](https://github.com/c-morey)
* [Arnaud Durieux](https://github.com/Pablousse)
* [Atefeh Hossein Niay](https://github.com/atefehhosseinniay)
