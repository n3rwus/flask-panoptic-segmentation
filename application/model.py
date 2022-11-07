import torch
from PIL import Image
import torchvision
from torchvision import transforms


# user has to input 3 files
# 1. imagenet_classes.txt -> dog eating pizza
# 2. model for image recognition
# 3. picture for coco annotation

class BusinessLogic:
    def __int__(self):
        with open('imagenet_classes.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.model.eval()

    def infer(self, image_path):
        input_image = Image.open(image_path)
        preprocess = transforms.Compose({
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        })

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')
        with torch.no_grad():
            output = self.model(input_batch)

        output = torch.nn.functional.softmax(output[0], dim=0)
        confidence, index = torch.max(output, 0)

        return self.classes[index.item()], confidence.item()
