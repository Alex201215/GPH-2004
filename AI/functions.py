import pandas as pd
import matplotlib.pyplot as plt
import torch as t
import numpy as np
import random
import os

from tqdm.auto import tqdm


def accuracy_fn(y_true,
                y_pred):
    correct = t.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def plot_decision_boundary(model: t.nn.Module,
                           X: t.Tensor,
                           y: t.Tensor):
    """
    Plots decision boundaries of model predicting on X in comparison to y.
    """

    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = t.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with t.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(t.unique(y)) > 2:
        y_pred = t.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = t.round(t.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap="Blues", alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="Blues")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def print_train_time(start: float,
                     end: float,
                     device: t.device):
    """
    Prints difference between start and end time
    """

    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds.")
    return total_time


def eval_model(model: t.nn.Module,
               data_loader: t.utils.data.DataLoader,
               loss_fn: t.nn.Module,
               accuracy_fn,
               device: t.device):
    """
    Returns a dictionay containing the results of model predicting on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with t.inference_mode():
        for x, y in data_loader:

            # move data to device
            x, y = x.to(device), y.to(device)

            # make predictions
            y_pred = model(x)

            # accumulate loss and acc value *per batch*
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))

        # scale loss and acc to get average loss/acc *per batch*
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,    # only works when model is a class
            "model_loss": loss.item(),                 # turns loss into a single value
            "model_acc": acc}


def train_step(model: t.nn.Module,
               data_loader: t.utils.data.DataLoader,
               loss_fn: t.nn.Module,
               optimizer: t.optim.Optimizer,
               accuracy_fn,
               device: t.device):
    """
    Performs training with model learning on data loader.
    """
    train_loss, train_acc = 0, 0

    # set model to training mode
    model.train()

    # loop through training batches
    for batch, (x, y) in enumerate(data_loader):  # equivalent to (image, label)

        # put data on target device
        x, y = x.to(device), y.to(device)

        # 1. forward pass
        y_pred = model(x)

        # 2. calculate loss *per batch*
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # 3. calculate acc *per batch*
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))

        # 4. optimizer zero grad
        optimizer.zero_grad()

        # 5. loss backward
        loss.backward()

        # 6. optimizer step
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


def test_step(model: t.nn.Module,
              data_loader: t.utils.data.DataLoader,
              loss_fn: t.nn.Module,
              accuracy_fn,
              device: t.device):
    """
    Performs testing on model learning on data loader
    """
    test_loss, test_acc = 0, 0

    # set model to evaluating mode
    model.eval()

    # loop through testing batches
    with t.inference_mode():
        for x, y in data_loader:

            # put data on target device
            x, y = x.to(device), y.to(device)

            # 1. forward pass
            test_pred = model(x)

            # 2. calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y)

            # 3. calculate accuracy
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))

        # calculate test loss/acc average *per batch*
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    # print out what's happening
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.3f}%\n")


def make_predictions(model: t.nn.Module,
                     data: list,
                     device: t.device):
    """
    Computes prediction of a model for an input data
    """
    pred_probs = []

    # send model to device
    model.to(device)

    # set model to evaluating mode
    model.eval()
    with t.inference_mode():
        for sample in data:

            # prepare the sample
            sample = t.unsqueeze(sample, dim=0).to(device)

            # forward pass (model outputs raw logits)
            pred_logits = model(sample)

            # get prediction probability
            pred_prob = t.softmax(pred_logits.squeeze(), dim=0)

            # send pred_prod to cpu
            pred_probs.append(pred_prob.cpu())

    # stack pred_probs to turn list into a tensor
    return t.stack(pred_probs)


def get_subset(image_path: str,
               data_dir: str,
               data_splits: list,
               target_classes: list,
               amount: int,
               seed: int):
    """
    separate a larger dataset into smaller subsets each containing a specific class - Works well for FOOD101
    """
    random.seed(seed)
    label_splits = {}

    # get labels
    for data_split in data_splits:
        print(f"[INFO] Creating an image split for: {data_split}...")
        label_path = data_dir/"fft"/f"{data_split}.txt"
        print(data_dir)
        print(label_path)
        with open(label_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes]

        # get random subset of target classes image ID's
        number_to_sample = round(amount*len(labels))
        print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
        sampled_images = random.sample(labels, k=number_to_sample)

        # apply full paths
        image_paths = [pathlib.Path(str(image_path/sample_image) + ".jpg") for sample_image in sampled_images]
        label_splits[data_split] = image_paths

    return label_splits


def walk_through_dir(dir_path: str):
    """
    walks through dir_path returning its content
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
