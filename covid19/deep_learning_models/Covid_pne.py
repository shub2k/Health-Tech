


import torch 
import torch.nn as nn 
from torch.utils.data import Dataset , DataLoader 
import albumentations as a 
import cv2 
import matplotlib.pyplot as plt 
import timm 
import numpy as np
from torchvision import transforms




class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x



class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam , target_category






class dataset(Dataset):
    def __init__(self,path,augmentations = None):
        super().__init__()
        self.image_path = path
        self.aug = augmentations 
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self,idx):
        
        img = cv2.imread(self.image_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.aug is not None:
            res = self.aug(image = img)
            img = res['image']
        img = img.astype(np.float32)
        img = img.transpose(2,0,1)
        img = torch.tensor(img)
        return img.unsqueeze(0)
        


class Prediction:
    def __init__(self,path):
        self.path = path 

    def show_prediction(self):
        def preprocess_image(img):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            preprocessing = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
            return preprocessing(img.copy()).unsqueeze(0)

        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask),cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return np.uint8(255 * cam)

        use_cuda = False
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = {
        'image_size':256,
        'backbone':'resnet200d',
        'batch_size':1,
        'num_workers':4
            }
        trans = a.Compose([
            a.Resize(cfg['image_size'],cfg['image_size']),
            a.Normalize()
                ])

        
        model = timm.create_model('tf_efficientnet_b0_ns',pretrained = False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 3)
        model.load_state_dict(torch.load('covid19/deep_learning_models/Covid_classifier_tf_efficientnet_b0_ns_fold0_validf10.9897097350755854.pth',map_location = 'cpu'))
        model.to(device)
        predi = ''
        
        covid_dataset = dataset(self.path,augmentations = trans)
      #  loader = DataLoader(covid_dataset,num_workers = cfg['num_workers'],shuffle = False,
      #  batch_size = cfg['batch_size'],pin_memory = False)
        
       # for i , image in enumerate(loader):
        image = covid_dataset[0]
        model.eval()
        image = image.to(device,dtype = torch.float32)
        logits = model(image)
        prediction = torch.argmax(logits, 1).detach().cpu().numpy()
        
        json_data = {
        'Covid':0,
        'Pneumonia':1,
        'Normal':2
        }
        new_image = cv2.imread(self.path[0])
        new_image = np.float32(new_image) / 255
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        input_img = preprocess_image(new_image)
        
        # 6 layer [ grad cam ]
        grad_cam = GradCam(model=model, feature_module=model.blocks,
                            target_layer_names=["6"], use_cuda=use_cuda)
        target_category = None
        grayscale_cam , output = grad_cam(input_img, target_category)

        grayscale_cam = cv2.resize(grayscale_cam, (new_image.shape[1],new_image.shape[0]))
        cam = show_cam_on_image(new_image, grayscale_cam)

        cv2.imwrite("static\image\cam.jpg", cam)
        if prediction[0] == json_data['Covid']:
              #  print('Dude you have covid maintain social distance')
            return 'Yes you have Covid'
        elif prediction[0] == json_data['Pneumonia']:
              #  print('Bro you have pneumonia have some hot soup')
            return 'Yes you have pneumonia'
        else:
              #  print('Bro you are fine just stay healthy')
            return 'You are fine just stay healthy'



#image_path = [r'C:\Users\ACER\Desktop\Health tech\images\Normal-7.png']
# obj = Prediction(image_path)
# pred = obj.show_prediction()
# print(pred)
