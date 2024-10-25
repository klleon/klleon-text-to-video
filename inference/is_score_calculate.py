# import os
# import torch
# from pytorch_image_generation_metrics import (
#     get_inception_score_from_directory,
#     get_fid_from_directory,
#     get_inception_score_and_fid_from_directory)
# from glob import glob

# IS, IS_std = get_inception_score_from_directory(
#     './IS_Testset/lrw_val/')


# print(IS, IS_std)

import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
from torchvision import datasets, transforms
import cv2

import numpy as np
from scipy.stats import entropy
from glob import glob


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    print(f"N: {N}")

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=True)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, path):
            # self.dirs = glob(f"{path}/*/*", recursive = True)


            self.transfomrs = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
            self.img_paths = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(('png', 'jpg', 'jpeg')):
                        self.img_paths.append(os.path.join(root, file))

        def __getitem__(self, index):
            img_path = self.img_paths[index]
            img = cv2.resize(cv2.imread(img_path), (720, 480))

            return self.transfomrs(img)

        def __len__(self):
            return len(self.img_paths)

    lrw_dataset = IgnoreLabelDataset('./results_img/')
    print(len(lrw_dataset))


    print ("Calculating Inception Score...")
    print (inception_score(lrw_dataset, cuda=True, batch_size=2, resize=True, splits=1))