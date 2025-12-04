from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vgg import VGG11, VGG13, VGG16, VGG19
from .mobilenetv3 import MobileNetV3Small, MobileNetV3Large
from .wideresnet import WRN168, WRN282, WRN2810, WRN502, WRN1012
from .inceptionv3 import InceptionV3
from .vit import ViT
from typing import Tuple


def get_model(model_name: str, im_channels: int = 3, num_classes: int = 10, im_size: Tuple[int, ...] = (32,32), logger: callable = None):
    printer_func = print if logger is None else logger.print_it
    printer_func('Setting up model "{}"...'.format(model_name))
    # Setup model
    if model_name == 'resnet18':
        model = ResNet18(channel=im_channels, num_classes=num_classes)
    elif model_name == 'resnet34':
        model = ResNet34(channel=im_channels, num_classes=num_classes)
    elif model_name == 'resnet50':
        model = ResNet50(channel=im_channels, num_classes=num_classes)
    elif model_name == 'resnet101':
        model = ResNet101(channel=im_channels, num_classes=num_classes)
    elif model_name == 'resnet152':
        model = ResNet152(channel=im_channels, num_classes=num_classes)
    elif model_name == 'wideresnet_16_8':
        model = WRN168(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'wideresnet_28_2':
        model = WRN282(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'wideresnet_28_10':
        model = WRN2810(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'wideresnet_50_2':
        model = WRN502(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'wideresnet_101_2':
        model = WRN1012(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'vgg11':
        model = VGG11(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'vgg13':
        model = VGG13(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'vgg16':
        model = VGG16(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'vgg19':
        model = VGG19(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'mobile_small':
        model = MobileNetV3Small(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'mobile_large':
        model = MobileNetV3Large(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'inception_v3':
        model = InceptionV3(channel=im_channels, num_classes=num_classes, im_size=im_size)
    elif model_name == 'vit':
        model = ViT(pretrained=True, in_channels=im_channels, num_classes=num_classes, image_size=im_size)
    else:
        printer_func('Specified model "{}" not recognized!'.format(model_name))
    return model