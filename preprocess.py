import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # setting image to grayscale and returning only one channel
    transforms.CenterCrop(64), # crop the image to center, if input is to small --> padding and then crop
    transforms.Normalize((0.5),(0.5)) #'''still have to get the means, so far 0.5 as base value'''
    ])

