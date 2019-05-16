# Stacked Hourglass for extrinsic camera parameter calibration

### Prepare the environment
    pip install torch
    pip install torchvision
    pip install tensorboardX
    pip install tensorflow-cpu
    pip install tqdm, imageio, 

### Prepare the dataset
The dataset can be found [here](http://users.utcluj.ro/~razvanitu/VPdataset.zip). The author just provide the download link, 
So I download the image for your convenience, the iamge data can be found in [BaiduYunDisk](https://pan.baidu.com/s/1EQZNFCJhjfnG87aKn9gY7w)
```
mkdir checkpoints
mkdir VisImage
mkdir -p runninglogs/runs
mkdir -p runinglogs/save
ln -s /yourdata  ./data/processed/VP_Img_resize
    
```


### Run the program
* Training
    - sh train.sh   
* Test
    - sh test.sh