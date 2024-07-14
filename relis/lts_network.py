''' 
Adapted from https://github.com/uncbiag/LTS/blob/main/code/calibration_models.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureModel(nn.Module):
    def __init__(self, num_classes):
        super(TemperatureModel, self).__init__()
        self.temperature_level_2_conv1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.num_classes = num_classes
        
    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

    def forward(self, logits, image):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).cuda()
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).cuda()
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).cuda()
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).cuda()
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_12 = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_12 * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(image) + torch.ones(1).cuda()
        temp_param = self.temperature_level_2_param_img(logits)
        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).cuda()) + sigma
        temperature = temperature.repeat(1, self.num_classes, 1, 1)
        return logits / temperature
    
    
    
class TemperatureModelLogits(nn.Module):
    def __init__(self, num_classes):
        super(TemperatureModelLogits, self).__init__()
        self.temperature_level_2_conv1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.num_classes = num_classes
        
    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

    def forward(self, logits, image):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).cuda()
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).cuda()
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).cuda()
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).cuda()
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_12 = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_12 * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(logits) + torch.ones(1).cuda()
        temp_param = self.temperature_level_2_param_img(logits)
        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).cuda()) + sigma
        temperature = temperature.repeat(1, self.num_classes, 1, 1)
        return logits / temperature
    
    

class TemperatureModelImage(nn.Module):
    def __init__(self, num_classes):
        super(TemperatureModelImage, self).__init__()
        self.temperature_level_2_conv1 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.num_classes = num_classes
        
    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

    def forward(self, logits, image):
        temperature_1 = self.temperature_level_2_conv1(image)
        temperature_1 += (torch.ones(1)).cuda()
        temperature_2 = self.temperature_level_2_conv2(image)
        temperature_2 += (torch.ones(1)).cuda()
        temperature_3 = self.temperature_level_2_conv3(image)
        temperature_3 += (torch.ones(1)).cuda()
        temperature_4 = self.temperature_level_2_conv4(image)
        temperature_4 += (torch.ones(1)).cuda()
        temperature_param_1 = self.temperature_level_2_param1(image)
        temperature_param_2 = self.temperature_level_2_param2(image)
        temperature_param_3 = self.temperature_level_2_param3(image)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_12 = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_12 * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(image) + torch.ones(1).cuda()
        temp_param = self.temperature_level_2_param_img(image)
        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).cuda()) + sigma
        temperature = temperature.repeat(1, self.num_classes, 1, 1)
        return logits / temperature