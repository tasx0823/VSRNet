# VSRNet
Inplementation of VSRNet


## Prerequisites

* PyTorch 0.4.1

## Required Data
Please repare the data following the instruction of [dual_encoding](https://github.com/danieljf24/dual_encoding).
You need to download
1. a pretrained [word2vec(3.0G)](http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz)
2. [video and text feature](https://drive.google.com/file/d/1Ku06oIuAEqagyIRqo1nVYuHzJyxRvECI/view?usp=sharing)

## Model Training
```
./do_all.sh densecaptrain densecapval densecapval full 0
```

## Model Testing
```
python tester.py 
```

