import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import dataset
import utility
import os
from glob import glob
import numpy as np

from monai.transforms.utils import allow_missing_keys_mode
from monai.utils import first
import nibabel as nib
import cv2
from utility import window_center_adjustment, dice_loss, calculate_metrics
import pickle
from tqdm import tqdm
import config
from dataset import KneeDataset
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets.swin_unetr import SwinUNETR
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from copy import deepcopy
import patchify
import shutil
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import Dice

bone_type='FC'
run_name = 'FC_crop_resize';
start_fold=0
print(bone_type)
print(run_name)
model_path=f'FC'

if os.path.exists(model_path) is False:
    os.mkdir(model_path);

def patchify_valid_data(valid_imgs, valid_masks, fold):
    ret_img = [];
    ret_mask = [];


    os.makedirs(os.path.join('valid_patches', f'{fold}'));
    for img_path, mask_path in tqdm(zip(valid_imgs, valid_masks)):
        file_name = os.path.basename(img_path);
        file_name = file_name[:file_name.rfind('.')];
        img = nib.load(img_path);
        mask = nib.load(mask_path);
        canonical_mask = nib.as_closest_canonical(mask)
        canonical_img = nib.as_closest_canonical(img)
        img = canonical_img.get_fdata();
        mask = canonical_mask.get_fdata();

        #zero-pad image and mask
        d,h,w = mask.shape;
        new_d = (d%config.CROP_SIZE_D != 0)*(config.CROP_SIZE_D - d%config.CROP_SIZE_D);
        new_h = (h%config.CROP_SIZE_H != 0) * (config.CROP_SIZE_H - h%config.CROP_SIZE_H);
        new_w = (w%config.CROP_SIZE_L != 0) * (config.CROP_SIZE_L - w%config.CROP_SIZE_L);

        padded_img = np.zeros((d+new_d, h+new_h, w+new_w), dtype=img.dtype);
        padded_mask = np.zeros((d+new_d, h+new_h, w+new_w), dtype=mask.dtype);

        padded_img[:d,:h,:w] = img;
        padded_mask[:d,:h,:w] = mask;

        img_patches = patchify.patchify(
            padded_img,(config.CROP_SIZE_D, config.CROP_SIZE_H, config.CROP_SIZE_L), 
            (config.CROP_SIZE_D, config.CROP_SIZE_H, config.CROP_SIZE_L));
        mask_patches = patchify.patchify(padded_mask,config.CROP_SIZE_D,config.CROP_SIZE_D);
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                for k in range(img_patches.shape[2]):
                    # for m in range(96):
                    #     im = img_patches[i][j][k][m];
                    #     ma = mask_patches[i][j][k][m];
                    #     im = (((im - np.min(im)) / np.max(im))*255).astype("uint8")
                    #     cv2.imshow('img', im);
                    #     cv2.imshow('mask', ma*255);
                    #     cv2.waitKey();
                    pickle.dump(img_patches[i][j][k],
                                 open(os.path.join('valid_patches',f'{fold}',f'{file_name}_{i}{j}{k}_img.dmp'), 'wb'));
                    pickle.dump(mask_patches[i][j][k],
                                 open(os.path.join('valid_patches',f'{fold}',f'{file_name}_{i}{j}{k}_mask.dmp'), 'wb'));
                    ret_img.append(os.path.join('valid_patches',f'{fold}',f'{file_name}_{i}{j}{k}_img.dmp'));
                    ret_mask.append(os.path.join('valid_patches',f'{fold}',f'{file_name}_{i}{j}{k}_mask.dmp'));
    return ret_img, ret_mask;

def store_folds():
    
    img_list = glob(os.path.join('/home', 'peyman.tahghighi', 'img', '*.nii.gz'));
    mask_list = [];
    img_list_temp = [];
    for img_path in tqdm(img_list):
        file_name = os.path.basename(img_path);
        if os.path.exists(os.path.join('/home', 'peyman.tahghighi', bone_type, file_name)) is True:
            mask_list.append(os.path.join('/home', 'peyman.tahghighi', bone_type, file_name));
            img_list_temp.append(img_path);
    img_list = img_list_temp;
    
    img_list = np.array(img_list);
    mask_list = np.array(mask_list);
    print(len(img_list));
    print(len(mask_list));
    
    if os.path.exists(f'valid_patches'):
        shutil.rmtree('valid_patches');
    
    if os.path.exists(model_path+'_folds') is False:
        os.mkdir(model_path+'_folds');
    kfold = KFold(n_splits=config.FOLDS, random_state=42, shuffle=True);
    fold = 0;
    for train_indices, valid_indices in kfold.split(img_list):
        train_img_list, train_mask_list, valid_img_list, valid_mask_list = img_list[train_indices], mask_list[train_indices], img_list[valid_indices], mask_list[valid_indices];
        #valid_img_list, valid_mask_list = patchify_valid_data(valid_img_list, valid_mask_list, fold);
        pickle.dump([train_img_list, train_mask_list, valid_img_list, valid_mask_list],open(os.path.join(model_path+'_folds', f'{fold}.dmp'), 'wb'));
        fold +=1;

def loss_func(pred, mask):
    #print(f'mask count in loss: {torch.sum(mask).item()}');
    bce_loss = sigmoid_focal_loss(pred.squeeze(dim=1), mask.float(), reduction="mean");
    d_loss = dice_loss(pred, mask, sigmoid=True, arange_logits=True, spatial_dims=3);
    return bce_loss + d_loss;

def train_one_epoch(epoch, model, optimizer, loader, scaler, dice_calculator):
    epoch_loss = [];
    epoch_dice = [];

    pbar = enumerate(loader);
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'Dice'));
    pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    for batch_idx, (mri, mask) in pbar:
        mri, mask = mri.to(config.DEVICE), mask.to(config.DEVICE);

        # if config.DBG is True:
        #     for j in range(config.BATCH_SIZE):
        #         mri_np = mri.detach().cpu().numpy()[j];
        #         mask_np = mask.detach().cpu().numpy()[j];
        #         mri_np = ((mri_np*0.5 + 0.5)).astype("uint8")

        #         for i in range(96):
        #             mri_np_s = mri_np[i];
        #             mask_np_s = mask_np[i].astype("uint8")*255;
        #             b = cv2.addWeighted(mri_np_s, 0.5, mask_np_s, 0.5, 0.0);
        #             cv2.imshow('b', b);
        #             cv2.imshow('mri', mri_np_s);
        #             cv2.imshow('mask', mask_np_s);
        #             cv2.waitKey();


        #with torch.cuda.amp.autocast_mode.autocast():
        pred = model(mri.unsqueeze(dim=1));
        loss = loss_func(pred, mask);
        loss = loss / config.VIRTUAL_BATCH_SIZE;
        pred = torch.sigmoid(pred);
        if torch.sum(mask).item() != 0:
            dice = dice_calculator(pred.flatten(), mask.squeeze().long().flatten());
            epoch_dice.append(dice.item());

        scaler.scale(loss).backward();
        epoch_loss.append(loss.item());

        if ((batch_idx+1) % config.VIRTUAL_BATCH_SIZE == 0) or (batch_idx+1 == len(loader)):
            scaler.step(optimizer);
            scaler.update();
            model.zero_grad(set_to_none = True);


        pbar.set_description(('%10s' + '%10.4g'*2) %(epoch, np.mean(epoch_loss), np.mean(epoch_dice)));

    return np.mean(epoch_loss), np.mean(epoch_dice);

def valid_one_epoch(epoch, model, loader, confusion_matrix_calculator):
    epoch_loss = [];
    epoch_prec = [];
    epoch_rec = [];
    epoch_f1 = [];

    pbar = enumerate(loader);
    print(('\n' + '%10s'*5) %('Epoch', 'Loss', 'Precision', 'Recal', 'F1'));
    pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    cumulative_cm = None;

    with torch.no_grad():
        for batch_idx, (mri, mask) in pbar:
            mri, mask = mri.to(config.DEVICE), mask.to(config.DEVICE)
            if config.DBG is True:
                for j in range(config.BATCH_SIZE):
                    mri_np = mri.detach().cpu().numpy()[j];
                    mask_np = mask.detach().cpu().numpy()[j];
                    mri_np = ((mri_np*0.5 + 0.5)).astype("uint8")

                    for i in range(96):
                        mri_np_s = mri_np[i];
                        mask_np_s = mask_np[i].astype("uint8")*255;
                        b = cv2.addWeighted(mri_np_s, 0.5, mask_np_s, 0.5, 0.0);
                        cv2.imshow('b', b);
                        cv2.imshow('mri', mri_np_s);
                        cv2.imshow('mask', mask_np_s);
                        cv2.waitKey();

            #with torch.cuda.amp.autocast_mode.autocast():
            pred = model(mri.unsqueeze(dim=1));
            loss = loss_func(pred, mask);
            pred = torch.sigmoid(pred);
            pred = pred > 0.5;
            loss = loss/config.VIRTUAL_BATCH_SIZE;
            
            cm = confusion_matrix_calculator(pred.squeeze(dim=1), mask);
            if cumulative_cm is None:
                cumulative_cm = cm;
            else:
                cumulative_cm += cm;
            
            precision, recall, f1 = calculate_metrics(cumulative_cm);

           
            epoch_loss.append(loss.item());

            pbar.set_description(('%10s' + '%10.4g'*4) %(epoch, np.mean(epoch_loss),precision, recall, f1));
    precision, recall, f1 = calculate_metrics(cumulative_cm);
    return np.mean(epoch_loss), [precision, recall, f1];

store_folds();

model = SwinUNETR(img_size=(config.CROP_SIZE_D,config.CROP_SIZE_H,config.CROP_SIZE_L), in_channels=1, out_channels=1, feature_size=48);

m = torch.load('model.pt');
pretrain_weights = dict();
for k in m.keys():
    if k != 'out.conv.conv.weight' and k!= 'out.conv.conv.bias':
        pretrain_weights[k] = m[k];

RESUME_TRAINING = True;
if RESUME_TRAINING is True:
    checkpoint = torch.load('training_checkpoint.pt', map_location = 'cuda');
    start_fold = checkpoint['fold'];

for f in range(start_fold,config.FOLDS):
    print(f"+++++++++ Starting fold : {f} +++++++++")
    #initialize model weight, initialize other variables every fold
    if RESUME_TRAINING is False:
        model.load_state_dict(pretrain_weights, strict=False);
    if RESUME_TRAINING is True:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.to(config.DEVICE);
    
    scaler = torch.cuda.amp.grad_scaler.GradScaler();
    optimizer = optim.Adam(model.parameters(), config.LEARNING_RATE);
    summary_writer = SummaryWriter(os.path.join(model_path + '_exp', f'_{run_name}', f'_{run_name}_{f}'));
    best_loss = 10;
    best_metrics = None;
    early_stopping = config.EARLY_STOPPING_TOLERANCE;
    epoch = 0;
    
    if RESUME_TRAINING is True:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['gradscalar_state_dict']);
        epoch = checkpoint['epoch'];
        best_loss = checkpoint['best_loss'];
        early_stopping = checkpoint['ES'];
        best_metrics = checkpoint['best_metrics'];
        RESUME_TRAINING = False;
        print(f'Resuming training from epoch {epoch}')
    
    confusion_matrix_calculator = ConfusionMatrix(task='binary').to(config.DEVICE);
    dice = Dice(num_classes=1).to(config.DEVICE);

    #define training and testing dataset
    train_img_list, train_mask_list, valid_img_list, valid_mask_list = pickle.load(open(os.path.join(model_path+'_folds', f'{f}.dmp'),'rb'));
    kneedataset_valid = KneeDataset(valid_img_list, valid_mask_list, train = False);
    kneedataset_train = KneeDataset(train_img_list, train_mask_list);
    train_loader = DataLoader(kneedataset_train, num_workers=config.NUM_WORKERS, shuffle = True, batch_size=config.BATCH_SIZE, drop_last=False, pin_memory=True);
    valid_loader = DataLoader(kneedataset_valid, num_workers=config.NUM_WORKERS, shuffle = False, batch_size=config.BATCH_SIZE, drop_last=False, pin_memory=True);

    
    
    while(True):
        model.train();
        loss_train,dice_train = train_one_epoch(epoch, model, optimizer, train_loader, scaler, dice);

        model.eval();
        loss_valid, valid_metrics = valid_one_epoch(epoch, model, valid_loader, confusion_matrix_calculator)

        if loss_valid < best_loss:
            #stroe best results so far and reset early stopping tolerance
            best_loss = loss_valid;
            early_stopping = config.EARLY_STOPPING_TOLERANCE;
            best_metrics = valid_metrics;
            # model_path=f'/home/reza.kakavand/model_3D_Unet/{bone_type}/{run_name}/'
            model_save=f'best_model_fold: {f}.ckpt'
            pickle.dump(model.state_dict, open(model_path+model_save, 'wb'));
            print('New best model found!');
        else:
            early_stopping -=1;

        if early_stopping <= 0:
            break;

        epoch += 1;
        summary_writer.add_scalar('train/loss', loss_train, epoch);
        summary_writer.add_scalar('train/dice', dice_train, epoch);
        summary_writer.add_scalar('valid/f1', valid_metrics[2], epoch);
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'gradscalar_state_dict': scaler.state_dict(),
                'fold': f,
                'best_loss': best_loss,
                'ES': early_stopping,
                'best_metrics': best_metrics
            },
            f'training_checkpoint.pt'
            )
    f = open(os.path.join(model_path, f'{f}_results.txt'), 'w');
    f.write(model_path+f'Epochs: {epoch}\tPrecision: {best_metrics[0]}\tRecall: {best_metrics[1]}\tF1: {best_metrics[2]}');
    