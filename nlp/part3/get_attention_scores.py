from PIL import Image


image = Image.open("img/attention_scores.png")
pixel_values = list(image.getdata())
print(pixel_values)
