# Image-Super-Resolution

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
|    └─    └─ yyyy.png
|
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


Generate the answer in ```answer``` folder
---
weights_fold : decide where the weight you load.  
weights : weight.  
save_fold : your save location.    
```
python inference.py --weights_fold weights --weights 895_31.3226.pth --save_fold answer --transforms True
```
Thanks to 
---
Here we apply the model from  
[saeed-anwar/DRLN](https://github.com/saeed-anwar/DRLN.git)

Reference
---
[1] Y. Zhang et al.: Image Super-Resolution Using Very Deep Residual Channel Attention Networks.  
[[2](https://arxiv.org/abs/1906.12021v1)] Saeed, Anwar, and Nick Barnes, Densely Residual Laplacian Super-Resolution.


