import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from sklearn import model_selection
import torch
import torchvision
from tqdm import tqdm

def get_dataset(data_dir, filename, img_size=224, invert=False):
    """
    Load and and process all images in our data folder. The folder contains all images in the original MNIST dataset as well as
    math symbols from the CROHME dataset. MNIST images are 28 x 28, CROHME images are 45 x 45. All images are converted to grayscale,
    padded, and then resized to img_size x img_size. Returns a PyTorch tensor intended for use with PyTorch models.
    """
    if invert:
        invert_grayscale(data_dir, filename)

    folder = os.path.join(data_dir, filename)
    dataset = torchvision.datasets.ImageFolder(folder,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.Resize([img_size, img_size]),
                    torchvision.transforms.ToTensor(),
                ]))

    return dataset

def invert_grayscale(data_dir, filename):
    """
    This inverts the pixel values of images in the dataset.
    """
    for folder in tqdm(os.listdir(os.path.join(data_dir, filename))):
        if folder != "DS_Store":
            path = os.path.join(data_dir, filename, folder)
            for file in os.listdir(path):
                img = cv2.imread(os.path.join(path, file))
                inverted = abs(255 - img)
                cv2.imwrite(os.path.join(path, file), inverted)

def normalize(dataset):
    """
    Calculates the mean and standard deviation of the dataset. Averages the mean and standard deviation of each of the images.
    """
    mean = 0
    std = 0

    for img, _ in tqdm(dataset):
        mean += img.mean()
        std += img.std()

    return mean/len(dataset), std/len(dataset)

def test_train_split(dataset, test_pct=0.2):
    """
    Calculates the indices for performing a stratified test-train split. Uses test_pct of the dataset to set aside for testing.
    Returns indices of a PyTorch dataset.
    """
    targets = dataset.targets
    train_index, test_index = model_selection.train_test_split(np.arange(len(targets)), test_size=test_pct, random_state=42,
                                                               shuffle=True, stratify=targets)
    return train_index, test_index

def get_dataloaders(dataset, train_index, test_index, batch_size=128):
    """
    Generates a dataloader of size batch_size and shuffles it if specified. The dataloaders should be unshuffled for testing.
    If the dataset is going to be used for an SKLearn model, set batch size to the length of the dataset. This dataloader is used
    only for PyTorch models.
    """
    dataloaders = {"train": None, "test": None}
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_index)
    dataloaders["train"] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    dataloaders["test"] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return dataloaders

def threshold(pixel):
    """
    If a pixel has a value below 1, set it to 0. meant to be used with the PyTorch apply_ method.
    """
    if pixel < 1:
        return 0
    else:
        return pixel

def preprocess(img, img_size=224, mean=None, std=None, to_numpy=False, display=False):
    """
    Takes in an image as a cv2 Image object to be used in our model and preprocesses it. The image is made up of black and white
    only. We pad both axes by one eighth of our desired image size minus the images current size. We then resize the image and crop
    it to the center with size img_size x img_size. If mean and std are not none, the image and normalized. If to_numpy is true,
    return as a numpy array rather than a PyTorch tensor.
    """
    height, width = img.shape

    pad_size_w = int((img_size - width) / 8)
    pad_size_h = int((img_size - height) / 8)

    if type(mean) == torch.Tensor and type(std) == torch.Tensor:
        loader = torchvision.transforms.Compose([
            torchvision.transforms.Pad((pad_size_w, pad_size_h), fill=0, padding_mode="constant"),
            torchvision.transforms.Resize([img_size, img_size]),
            torchvision.transforms.CenterCrop([img_size, img_size]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((mean), (std))
        ])
    else:
        loader = torchvision.transforms.Compose([
            torchvision.transforms.Pad((pad_size_w, pad_size_h), fill=0, padding_mode="constant"),
            torchvision.transforms.Resize([img_size + 20, img_size + 20]),
            torchvision.transforms.CenterCrop([img_size, img_size]),
            torchvision.transforms.ToTensor()
        ])

    img = loader(PIL.Image.fromarray(img))
    img = img.apply_(threshold)

    if display:
        plt.imshow(img[0].numpy(), "gray")
        plt.show()

    if to_numpy:
        return img.numpy()
    return img

def find_symbols(img_path, display=False):
    """
    Identifies all of the symbols in a given image. Returns two lists: a list of PILImage objects and a corresponding list of OpenCV
    Moment objects containing details about the symbols location in the image. If display is true, display the original image with
    bounding boxes around each detected character.
    """
    bounding_boxes = [] # The bounding boxes of each contour
    corners = [] # The corners of each bounding box

    image = cv2.imread(img_path) # Load the example
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) # Convert to grayscale (one channel)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # Apply a Gaussian blur with kernel size of 5 x 5
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Compute an Otsu cutoff and apply a binary threshold to the image
    threshold = cv2.bitwise_not(threshold)
    contours, parents = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Locates the contours of the image and which other contours they are nested inside

    for i in range(0, len(contours)): # Find the bounding box of each contour
        if (parents[0][i][3] == -1): # If the contour is not nested inside any others (i.e. it is actually around a character)
            left = tuple(contours[i][contours[i][:, :, 0].argmin()][0])
            right = tuple(contours[i][contours[i][:, :, 0].argmax()][0])
            top = tuple(contours[i][contours[i][:, :, 1].argmin()][0])
            bottom = tuple(contours[i][contours[i][:, :, 1].argmax()][0])
            bounding_boxes.append([top, right, bottom, left])

    for box in bounding_boxes: # Find the corners of each bounding box
        c1 = [box[3][0], box[0][1]]
        c2 = [box[1][0], box[0][1]]
        c3 = [box[1][0], box[2][1]]
        c4 = [box[3][0], box[2][1]]
        corners.append([c1, c2, c3, c4])

    area_x = [] # The width of each bounding box
    area_y = [] # The height of each bounding box
    area = [] # The area of each bounding box

    for c in corners: # Compute the height and width of each bounding box
        x = abs(c[0][0] - c[1][0])
        y = abs(c[0][1] - c[2][1])
        area_x.append(x)
        area_y.append(y)
        area.append(x * y)

    mean_x = np.mean(area_x)
    mean_y = np.mean(area_x)
    std_x = np.std(area_y)
    std_y = np.std(area_y)

    valid_indices = [] # Which bounding boxes we will ultimately use

    for i in range(len(area_x)): # Filter out outliers
        if not (area_x[i] < mean_x - std_x and area_y[i] < mean_y - std_y):
            valid_indices.append(i)

    if display:
        plt.imshow(image, "gray")
        for box in corners:
            if corners.index(box) in valid_indices:
                plt.plot([box[0][0], box[1][0]],[box[0][1], box[1][1]],'g-',linewidth=2)
                plt.plot([box[1][0], box[2][0]], [box[1][1], box[2][1]],'g-',linewidth=2)
                plt.plot([box[2][0], box[3][0]], [box[2][1], box[3][1]],'g-',linewidth=2)
                plt.plot([box[3][0], box[0][0]], [box[3][1], box[0][1]],'g-',linewidth=2)
        plt.show()
        cv2.waitKey(0)

    characters = []
    filtered_corners = []
    filtered_area = []

    for i in range(len(corners)):
        if i in valid_indices:
            c = corners[i]
            cropped = threshold.copy()[c[0][1]:c[2][1], c[0][0]:c[1][0]] # Crop the image at the bounding box and invert the colors
            characters.append(cropped) # Add to the list as a PIL image
            filtered_corners.append(corners[i])
            filtered_area.append(area[i])

    return characters, filtered_corners, filtered_area