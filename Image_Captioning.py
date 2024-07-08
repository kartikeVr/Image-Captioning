import requests, os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

def img_path(input):
    if isinstance(input, str):
        if input.startswith("https"):
            raw_image = Image.open(requests.get(input, stream=True).raw).convert('RGB')
            return raw_image
        else:
            indices = [i for i, char in enumerate(input) if char == '\\']
            direct = input[:indices[-1]]
            directory = direct.replace("\\", "/")
            filename = input[indices[-1]+1:]
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            return image
    else:
        print("Enter your proper path")


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

text = "a photography of"
inputs = processor(img_path("https://th.bing.com/th/id/R.2e428e8ae830e4015f0df533b8f006e1?rik=zskWlzdQaXpE1g&riu=http%3a%2f%2fwww.dumpaday.com%2fwp-content%2fuploads%2f2016%2f02%2frandom-pictures-1.jpg&ehk=xuubRylr%2bQ819mR1Fmu%2bbeB0Nbh5KEQ37YIe0L0JaK4%3d&risl=&pid=ImgRaw&r=0"), text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

