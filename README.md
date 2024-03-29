# Image-Super-Resolution
![image](https://user-images.githubusercontent.com/93210989/190902049-4a7d2a31-9b63-4e07-b344-972a81fc3d54.png)

Here we deal with the task that need to upscale the LR image to HR in scale=3.  
And the evaluate the performance metric is PSNR. 

Data Prepare
---
Before training model, we need to create the LR image from HR  by applying ```create_train_pairs.py``` and ```create_valid_pairs.py``` these two codes.

<!-- Requirement -->
<!-- --- -->
Download model weight
---
Please download the following links folder : ( folder : weights )  
https://drive.google.com/file/d/1LzMI3zr6lvoDbNDnCv5TIrECQt0FsZXy/view?usp=sharing  
This folder will help you to implement the inference.py to generate the answer file in the ```answer``` folder.  

Folder Structure  
---
Please download the weight yourself and put into ```weights\ 895_31.3226.pth ``` like below structure.
Here the ```model``` fold we apply [saeed-anwar/DRLN](https://github.com/saeed-anwar/DRLN.git).
```
├── dataset/
|    ├─ testing_lr_images
|    │     ├─ 00.png      # pic 
|    │     ├─ 01.png
|    │     │     .
|    │     ├─ 13.png
|    ├─ training_hr_images
|    │     ├─ xxx.png      
|    │     │     .
|    │     └─ xxxx.png
|    ├─ valid  #  YOU CAN FIND VALID PIC YOURSELF
|    │     ├─ yyy.png
|    │     │     .
|    |     └─ yyyy.png
|    |
|    ├─ Train   # after conducting create_train_pairs.py will appears
|    |    ├─ HR
|    |    |  └─ 3x ...
|    |    └─ LR
|    |       └─ 3x ...
|    └─ Valid   # Same like above
|         ├─ HR
|         |  └─ 3x ...
|         └─ LR
|            └─ 3x ...
├── model/  
|    ├─ commom.py
|    ├─ drln.py
|    └─ ops.py
|
|    
├── src/
|    ├─ AccumulateAvg.py
|    ├─ TrainFunction.py
|    ├─ dataset.py
|    └─ utils.py
|
├── weights/
|    └─ 895_31.3226.pth
|
├─ create_train_pairs.py
├─ create_valid_pairs.py
├─ train.py
└─ inference.py

```

Generate the answer in ```answer``` folder
---
Before conduct the following command, be sure you had download the weight or trained it by yourself.  
weights_fold : decide where the weight you load.  
weights : weight.  
save_fold : your save location.    
```
python inference.py --weights_fold weights --weights 895_31.3226.pth --save_fold answer --transforms True
```

Training
---
weights_fold : decide where the weight saved.  
cuda : assigned cuda.
```
python train.py --weights_fold weights3 --cuda cuda:1
```


Codalab Result
---
Here the bicubic is the traditional and basic method, PSNR at least 26.  
| Type        | Bicubic | Baseline | My Result |
| ------------- |:-------------:|:-------------:|:-------------:|
| PSNR| 26.0654 | 27.4162 |28.0441|

Thanks to 
---
Here we apply the model from  
[saeed-anwar/DRLN](https://github.com/saeed-anwar/DRLN.git)

Reference
---
[1] Y. Zhang et al.: Image Super-Resolution Using Very Deep Residual Channel Attention Networks.  
[[2](https://arxiv.org/abs/1906.12021v1)] Saeed, Anwar, and Nick Barnes, Densely Residual Laplacian Super-Resolution.


