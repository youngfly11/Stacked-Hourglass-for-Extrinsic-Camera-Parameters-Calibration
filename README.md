# HumanProtein
Course Project For DIP
### Prepare the environment
* Install Plus 
    ``pip install git+https://github.com/SHTUPLUS/plus.git``
* Install Goodluck
    ``pip install git+https://github.com/Rhyssiyan/goodluck.git``
* Data
```
   git clone https://github.com/SHTUPLUS/HumanProtein.git
   cd HumanProtein/codebase
   mkdir -p ./data/processed
   ln -s /group/readonly/classification/HumanProtein/processed/train ./data/processed/train
   ln -s /group/readonly/classification/HumanProtein/processed/test ./data/processed/test
   ln -s /group/readonly/classification/HumanProtein/processed/train_labels.npy ./data/train_labels.npy
   mkdir -p .protein/models
   ln -s /group/readonly/models/proteins/resnet34-333f7ec4.pth .protein/models/
```
### Run the program
* Training
    - ``CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train``   
    - goodluck run "python main.py train" --ngpu 4 --env l2al
    - Tip:
        + 将数据放在shared memory中，在default.yaml中修改img_path
        ``cp -r ~/projects/HumanProtein/codebase/data/processed/train /dev/shm/``