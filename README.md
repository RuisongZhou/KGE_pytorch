# KGE_pytorch

**Introduction**

This is the Pytorch implementaion of some nowledge graph embedding(KGE) models.
And I have test these models with popular datasets.
 
**Implemented features**

Models:
 - [x] TransE
 - [x] TransH
 - [x] TransD
 - [ ] TransA
 - [x] ConvE
 - [x] RotatE
 
 **Usage**
 1. Download datasets and put it in `./data` directory.
 2. Then you can run commend as follows to train/test/valid the models. All training codes can be found in `./examples/`
 ```bash
 python3 ./examples trainTransE.py
```
 If you want to modify parameters or datasets. You can rewrite training files and overwrite raw parameters. The parameters at `Config.py`
 come from their paper.