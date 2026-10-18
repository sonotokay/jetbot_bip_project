from PIL import Image

# Load images
img1 = Image.open('training_image_1.png')
img2 = Image.open('training_image_2.png')
img3 = Image.open('training_image_3.png')

# Combine side-by-side
width = img1.width + img2.width + img3.width
height = max(img1.height, img2.height, img3.height)
combined = Image.new('RGB', (width, height), 'white')

combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width, 0))
combined.paste(img3, (img1.width + img2.width, 0))

combined.save('training_curves_all_surfaces.png')