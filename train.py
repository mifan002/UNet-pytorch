import torch as t
from datetime import datetime, timedelta
import torch.backends.cudnn as cudnn
import pandas as pd
from tqdm import tqdm
import os
import imageio as io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
# !pip install torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
from model import UNet
from data_generator import DataGenerator



# print device when GPU is available
print("Device name: {}".format(t.cuda.get_device_name(0)))


# create the model
num_classes = 1
unet = UNet(in_img_channel=1, filter_num=16, class_num=num_classes)

# training parameters
lr = 0.5 * 10e-3
start_epoch = 0
epochs = 20
batch_size =32
optimizer = t.optim.Adam(unet.parameters(), lr=lr)

# create data
# bagls_path = r"F:\hiwi\dataset_bagls\training_224x224"
# df = pd.read_csv(r"F:\hiwi\dataset_bagls\bagls_ids.csv")
bagls_img_path = "gdrive/MyDrive/bagls/training_4096/training_4096/images/"
bagls_mask_path = "gdrive/MyDrive/bagls/training_4096/training_4096/masks/"

df = pd.read_csv("gdrive/MyDrive/bagls/bagls_4096.csv")
images_paths= [bagls_img_path + x for x in df['images'].to_list()]
masks_paths = [bagls_mask_path + x for x in df['masks'].to_list()]
X_train, X_val, y_train, y_val = train_test_split(images_paths, masks_paths, test_size=0.3, random_state=1)  # split data: train : val = 7 : 3

train_data_generator = DataGenerator(image_ids=X_train, mask_ids=y_train, batch_size=batch_size, num_class=num_classes, height=224, width=224, augment=True, shuffle=True)
val_data_generator = DataGenerator(image_ids=X_val, mask_ids=y_val, batch_size=batch_size, num_class=num_classes, height=224, width=224, augment=True, shuffle=True)

# training setting
cudnn.benchmark = True  # change 1 compared to cpu
# t.autograd.set_detect_anomaly(True)
start_time = datetime.now()
device = t.device('cuda') # change 1 compared to cpu
unet.to(device)

# need recording: loss, accuracy, IoU, duration of each epoch
loss_epochs = {'train': [], 'val':[]}
accuracy_epochs = {'train': [], 'val':[]}
iou_epochs = {'train': [], 'val':[]}
epoch_durations = []

# train the model with for-loop
# start training: from 0 - last epoch
for i in tqdm(range(start_epoch, epochs)):
    # set start time and create metrics for the upcoming epoch
    epoch_start = datetime.now()
    loss_1_epoch = {'train': [], 'val':[]}
    accuracy_train = BinaryAccuracy().to(device)
    accuracy_val = BinaryAccuracy().to(device)
    iou_train = BinaryJaccardIndex().to(device)
    iou_val = BinaryJaccardIndex().to(device)

    # start train-epoch: from 0 - last batch
    for (X_1_batch_train, y_1_batch_train) in train_data_generator:
        # train with autograd on
        with t.set_grad_enabled(True):
            unet.train()  # set mode: affect BN
            X_1_batch_train = X_1_batch_train.cuda()
            y_1_batch_train = y_1_batch_train.cuda()
            loss_train, prob_map_train = unet.forward(X_1_batch_train, y_1_batch_train)
            loss_1_epoch['train'].append(loss_train.item())
            accuracy_train.update(prob_map_train, y_1_batch_train.int())
            iou_train.update(prob_map_train, y_1_batch_train.int())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

    # start val-epoch: from 0 - last batch
    for (X_1_batch_val, y_1_batch_val) in val_data_generator:
        # validation with autograd off
        with t.set_grad_enabled(False):
            unet.eval()
            X_1_batch_val = X_1_batch_val.cuda()
            y_1_batch_val = y_1_batch_val.cuda()
            loss_val, prob_map_val = unet.forward(X_1_batch_val, y_1_batch_val)
            loss_1_epoch['val'].append(loss_val.item())
            accuracy_val.update(prob_map_val, y_1_batch_val.int())
            iou_val.update(prob_map_val, y_1_batch_val.int())

    # 1 epoch ends: shuffle the data, reset the batch counter
    train_data_generator.on_epoch_end()
    val_data_generator.on_epoch_end()

    # recording: loss
    loss_1_epoch['train'] = sum(loss_1_epoch['train']) / len(loss_1_epoch['train'])
    loss_epochs['train'].append(loss_1_epoch['train'])
    loss_1_epoch['val'] = sum(loss_1_epoch['val']) / len(loss_1_epoch['val'])
    loss_epochs['val'].append(loss_1_epoch['val'])

    # recording: accuracy
    accuracy_1_epoch_train = accuracy_train.compute()
    accuracy_epochs['train'].append(accuracy_1_epoch_train)
    accuracy_1_epoch_val = accuracy_val.compute()
    accuracy_epochs['val'].append(accuracy_1_epoch_val)

    # recording: IoU
    iou_1_epoch_train = iou_train.compute()
    iou_epochs['train'].append(iou_1_epoch_train)
    iou_1_epoch_val = iou_val.compute()
    iou_epochs['val'].append(iou_1_epoch_val)

    # recording: epochs duration
    current_epoch_duration = (datetime.now() - epoch_start).seconds
    epoch_durations.append(current_epoch_duration)

# all epochs end: write summary
writer = SummaryWriter(comment="Summary of Train with fp32 on GPU")
for i_th_epoch in range(epochs):
    writer.add_scalars('Loss', {'train': loss_epochs['train'][i_th_epoch], 'val': loss_epochs['val'][i_th_epoch]})
    writer.add_scalars('Accuracy', {'train': accuracy_epochs['train'][i_th_epoch], 'val': accuracy_epochs['val'][i_th_epoch]})
    writer.add_scalars('IoU', {'train': iou_epochs['train'][i_th_epoch], 'val': iou_epochs['val'][i_th_epoch]})
    writer.add_scalar('epoch duration', epoch_durations[i_th_epoch])
writer.close()

# save the model after training
t.save(unet.state_dict(), ".")