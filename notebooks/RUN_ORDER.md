# Ordem de Execução (Sprint Final)

## Glaucoma (TransUNet)
1. `00_setup_verificacao.py`
2. `01_transunet_train.py`
3. `01_transunet_test.py`

## DR Lesion Segmentation (U-Net++)
1. `00_setup_verificacao.py`
2. `02_unetpp_train.py` (alias oficial; chama `03_unetpp_train.py`)
3. `04_unetpp_test.py`

## DR Grading (EfficientNet-B4)
1. `00_setup_verificacao.py`
2. `07_efficientnet_train.py`
3. `08_efficientnet_test.py`
