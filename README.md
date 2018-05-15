# EE211A project: Prediction of Hemorrhage in the brain from MRI

--- Group member: Yutong Lu, Xin Zeng ---

During the reperfusion therapy in acute ischemic stroke (AIS), there is a chance that hemorrhagic transformation (HT) may happen, which is a common but devastating complication of AIS. However, if accurate prediction on the risk of HT could be made at early stage, it can increase the effectiveness and success of the stroke therapy. Therefore, developing high-performance machine learning tools for prediction of hemorrhagic transformation could provide surgeons with valuable insights, and more importantly, provide patients with timely treatment.

In this project, we basically designed and implemented a CNN structure in Tensorflow for the prediction of area and extent of hemorrhagic
transformation in brain MRI images, and it shows that CNN performs best among all structures we have used. We have shown the ground truth images and the predicted images where "red" indicates a large possibility of HT, "blue" means a low possibility of HT and "black" is just the background. By comparing with the ground truth images, it is believed that CNN is the most robust model that gives the clearest results shown since it is finding local features that is significant in predicting HT areas. 

![](https://github.com/amylyt000/EE211A-Digital-Image-Processing/raw/master/images/Picture2.png)
