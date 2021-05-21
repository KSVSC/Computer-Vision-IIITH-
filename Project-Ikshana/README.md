## Is that you? Metric Learning Approaches for Face Identification

### Team Members
- Ananya Arun (20171019)
- Samhita Kanaparthy (2018121005)
- Sravani Kaza (20171189)
- Thejasvi Konduru (20171134)

### Dataset Used
- We have used the LFW datasetwith 13233 images
- This contains about 5749 people and 1680 people with 2 or more images

### Pipeline of our Project
![Screenshot from 2021-04-24 20-14-26](https://user-images.githubusercontent.com/32260628/115962724-53f23300-a53a-11eb-992f-d70fcdc28118.png)

### Results

- Observed Accuracies with different feature extraction methods

Feature Extraction Method | KNN | LMNN + KNN
--- | --- | --- 
None | 66.67% | 73%
SIFT | 33.34% | 50% 
Dense SIFT | 68% | 80% 
Dense SIFT + Bounding Box | 78.26% | 86.95%

- Final results considering Dense sift with bounding box as the feature extraction method

Method | Accuracy 
--- | --- 
KNN using L2 | 78.26% 
LMNN + KNN | 86.95% 
MKNN using L2 | 85.45%  
MKNN + KNN | 91.95%
