from torchvision .models import resnet101
from torchvision .models .segmentation .deeplabv3 import _deeplabv3_resnet ,DeepLabV3
def deeplabv3_resnet101 (
*,
progress :bool =True ,
num_classes =None ,
)->DeepLabV3 :
    backbone =resnet101 (weights =None ,
    replace_stride_with_dilation =[False ,True ,True ])
    model =_deeplabv3_resnet (backbone ,num_classes ,True )
    return model