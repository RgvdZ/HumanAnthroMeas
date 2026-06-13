from PIL import Image

image_path = r"C:\Users\rugve\Downloads\Startup\Action\Dataset\train\mask\00a19f879a88dafab8af4eabbfef045a.png"

with Image.open(image_path) as img:
    width, height = img.size

    print(f"Dimensions: {width} x {height}")
    print(f"Mode: {img.mode}")

    channels = len(img.getbands())
    print(f"Channels: {channels}")
    print(f"Bands: {img.getbands()}")
