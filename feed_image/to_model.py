import torch
import torchvision
import torchvision.transforms.v2 as v2

def read_image(image):
    print(image)
    print(image.shape)
    model = torch.load('/workspaces/number_recognition_program/Saved_Model', weights_only=False)
    model.eval()
    print(model)
    image = image.float()
    input = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input)
    predicted_class = prediction.argmax(dim=1).item()
    return predicted_class