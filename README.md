# WDTANC
This is the source code for WDTANC

#### Preprocessing
  1. Download DEMAND Corpus dataset and unzip it to './datas/train_datas'
  2. 
   ```
   python rirs_2d.py
   python load_rirs.py
   ```
#### WDTANC
  Set 
  ```
  if_sh = ''
  ```
  in line 19 in train.py
  
  Set
  ```
  Model_type = 'conv_trans'
  ```
  in line 1 in train.py
