import convert
import cnn
import knn
import os
import processing
import torch
import torchvision
import tree

def check_gpu():
    """
    Returns true if there is a GPU available, else false.
    """
    if torch.cuda.is_available():
        return True

    return False

def train_cnn(rootpath, dataset, dataloader, device):
    """
    Trains the convolutional Neural Network
    """
    num_classes = len(dataset.classes)
    model, input_size = cnn.initialize_model("resnet", num_classes, resume_from=None)
    criterion = cnn.get_loss()
    optimizer = cnn.make_optimizer(model)
    scheduler = cnn.make_scheduler(optimizer)
    num_epochs = 20
    save_dir = os.path.join(rootpath, "weights")
    trained_model, val_acc_history = cnn.train_model(model, device, dataloader, criterion, optimizer, scheduler=scheduler, save_dir=save_dir, num_epochs=num_epochs)

    return trained_model, val_acc_history

def main(rootpath, data_dir, device, training = True, model_type="cnn", sample_name=None, display=False):
    """
    The primary method for controlling the training and testing of data. rootpath is the parent directory of main.py, data_dir
    is the subdirectory containing all training data, device is either CPU or GPU. Training is true if we are training a model
    and false otherwise. If training is false, sample name is the path to the image that you are converting to LaTeX. If display
    is true, print the output of the model for each character it is passed. The model type can be a convolutional neural network (cnn),
    k-nearest neighbors (knn), or a decision tree (tree). The knn and tree models have already been trained and will not be trained if
    training is true. This will only train the cnn.
    """
    dataset = processing.get_dataset(data_dir, img_size=28, filename="symbols_rectified")
    mean, std = torch.Tensor([.0942]), torch.Tensor([.2202])  # The values that were computed using the processing.normalize() function

    dataset.transforms = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.RandomRotation(degrees=5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomErasing(p=.2),
        torchvision.transforms.Normalize((mean), (std))
    ])

    if training:
        train_index, test_index = processing.test_train_split(dataset)
        dataloader = processing.get_dataloaders(dataset, train_index, test_index, batch_size=128)
        model, val_acc_history = train_cnn(rootpath, dataset, dataloader, device)

    else:
        output = []
        characters, corners, areas = processing.find_symbols(sample_name, display=True)

        if model_type == "cnn":
            num_classes = len(dataset.classes)

            model, input_size = cnn.initialize_model("simple", num_classes, resume_from=os.path.join(rootpath, "weights", "cnn_weights_epoch_4"))
            model.eval()

            for character in characters:
                img = processing.preprocess(character, img_size=224, mean=mean, std=std, to_numpy=False, display=True)
                out = model(img.unsqueeze(0))
                _, prediction = torch.topk(out, k=2, dim=1)

                symbols = []
                for pred in prediction[0]:
                    symbols.append(cnn.label_number_to_name(dataset, pred))

                if display:
                    print(symbols[0])

                output.append(symbols)

        elif model_type == "knn":
            for character in characters:
                img = processing.preprocess(character, img_size=224, mean=None, std=None, to_numpy=True, display=True)
                out = knn.get_nearest_neighbors(img, k=51) # The two most likely outputs

                symbols = [out[0], out[1]]
                if display:
                    print(symbols[0])

                output.append(symbols)

        elif model_type == "tree":
            for character in characters:
                img = processing.preprocess(character, img_size=224, mean=None, std=None, to_numpy=True, display=True)
                out = tree.get_label(img)

                symbols = [out, out]
                if display:
                    print(symbols[0])

                output.append(symbols)

        else:
            print("That is not a valid model name")

        equation = convert.to_latex(output, corners)
        print(equation)

if __name__ == "__main__":
    try:
        from google.colab import drive # Colab specific setup

    except Exception:
        rootpath = "." # Local setup

    else:
        drive.mount('/content/gdrive')
        rootpath = None
        for (parent_dir, subfolders, subfiles) in os.walk('/content/gdrive'):
            if "main.ipynb" in subfiles:
                rootpath = parent_dir
                break
        if rootpath is None:
            raise Exception("Could not find the file in your Google Drive")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = rootpath + "/data" # Where all data used for training is located
    training = False # True if we are training a model, false if we are evaluating it
    model_type = "cnn" # Select from ["knn", "cnn", "tree"]
    sample_name = os.path.join(rootpath, "samples", "sample2.png") # The name of the image to test the model on
    display = True # Print model output for each character it identifies
    main(rootpath, data_dir, device, training, model_type, sample_name, display)