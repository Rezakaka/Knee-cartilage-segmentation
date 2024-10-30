#! /usr/bin/env python 
import torch
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from utility import  dice_loss
import pickle
from tqdm import tqdm
import config
from dataset import KneeDataset
from monai.networks.nets.swin_unetr import SwinUNETR
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import SimpleITK as sitk
from utility import  dice_loss, calculate_metrics

num_mris=1 #number of images to segment
path_t_mri=r'D:\Cartilage_seg\9002430.nii.gz'# MRI address
path_model=r'D:\Cartilage_seg\FCbest_model_fold_ 1.ckpt'#model address

torch.multiprocessing.set_sharing_strategy('file_system')
PYTORCH_NO_CUDA_MEMORY_CACHING=1

def loss_func(pred, mask):
    bce_loss = sigmoid_focal_loss(pred.squeeze(dim=1), mask.float(), reduction="mean");
    d_loss = dice_loss(pred, mask, sigmoid=True, arange_logits=True, spatial_dims=3);
    return bce_loss + d_loss, 1-d_loss;

def train_one_epoch(epoch, model, optimizer, loader, scaler):
    epoch_loss = [];
    epoch_dice = [];

    pbar = enumerate(loader);
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'Dice'));
    pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    for batch_idx, (mri, mask) in pbar:
        mri, mask = mri.to(config.DEVICE), mask.to(config.DEVICE)

        #with torch.cuda.amp.autocast_mode.autocast():
        pred = model(mri.unsqueeze(dim=1));
        loss,dice = loss_func(pred, mask);
        loss = loss / config.VIRTUAL_BATCH_SIZE;

        scaler.scale(loss).backward();
        epoch_loss.append(loss.item());
        epoch_dice.append(dice.item());

        if ((batch_idx+1) % config.VIRTUAL_BATCH_SIZE == 0) or (batch_idx+1 == len(loader)):
            scaler.step(optimizer);
            scaler.update();
            model.zero_grad(set_to_none = True);
            #scaler.update();

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

def store_folds(img_list, mask_list):

    if os.path.exists('folds') is False:
        os.mkdir('folds');
    kfold = KFold(n_splits=config.FOLDS, random_state=42, shuffle=True);
    fold = 0;
    for train_indices, valid_indices in kfold.split(img_list):
        train_img_list, train_mask_list, valid_img_list, valid_mask_list = img_list[train_indices], mask_list[train_indices], img_list[valid_indices], mask_list[valid_indices];
        pickle.dump([train_img_list, train_mask_list, valid_img_list, valid_mask_list],open(os.path.join('folds', f'{fold}.dmp'), 'wb'));
        fold +=1;

def predict(model):
    img=[]
    mask=[]
    for i in range (0,num_mris):
        img.append(path_t_mri)
        #img.append(path_t_mri[i]) if more than one image
        # mask.append(path_t_mask)#uncomment this if you have a mask
    mri = KneeDataset(img, img, train = False);#you can replace the second img with a mask for comparison
    mri_loader = DataLoader(mri, num_workers=0, shuffle = False, batch_size=1, drop_last=False, pin_memory=True);
    pbar = enumerate(mri_loader);
    pbar = tqdm(pbar, total= len(mri_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    with torch.no_grad():
        pred_t=np.zeros((num_mris, 1, 1, 160, 128, 128))
        # mri = mri.to(config.DEVICE)
        for batch_idx, (mri1, mask1) in pbar:
            # mri1, mask1 = mri1.to(config.DEVICE), mask1.to(config.DEVICE)
            mri1, mask1 = mri1, mask1
            print('mri shape',np.shape(mri1))
        #that is the prediction for the mri
            pred=model(mri1.unsqueeze(dim=1));
            pred = torch.sigmoid(pred);
            pred=pred>0.5
            print('pred shape',np.shape(pred))
            
        #this is the numpy array of the prediction,
        #you can use this one to visualize the prediction of the model
            pred = pred.detach().cpu().numpy();
            pred_t[batch_idx,...]=pred

    
    return pred_t,mri1
pre_test=[]
if __name__ == "__main__":

    model = SwinUNETR(img_size=96, in_channels=1, out_channels=1, feature_size=48);
    d = pickle.load(open(path_model, 'rb'));
    model.load_state_dict(d());
    # Model=model.to("cuda")
    
    ##MAKE SURE THE MODEL IS IN EVAL STATE
    model.eval();
    out_pre=predict(model)
    pre_test=out_pre[0]
    mri=out_pre[1];
    

for i in range (100,105):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 9),gridspec_kw={'wspace':0.005, 'hspace':0.005},frameon=False)
    ax1.imshow(mri[0,i , :, :], cmap = 'gray', interpolation = 'bilinear')
    ax1.set_title('MRI')
    ax1.axis('off')
    ax2.imshow(pre_test[0,0,0,i,:,:], cmap = 'gray', interpolation = 'bilinear')
    ax2.set_title('Prediction')
    ax2.axis('off')
    ax3.contour(pre_test[0,0,0,i,:,:], colors = 'g', linewidths = 5, levels = [0.5])
    ax3.imshow(mri[0,i , :, :], cmap = 'gray', interpolation = 'none',alpha=0.8)
    ax3.set_title('Prediction on MRI')
    ax3.axis('off')

# a function to save the numpy array
def save_DICOM(i,numpy_pred,path):
    img1 = sitk.GetImageFromArray(numpy_pred)
    img1.SetOrigin([-114.623, -57.5773,-80.2892])#you can set the origin of the shape here
    img1.SetSpacing([0.364583, 0.364583,0.7])# set spacing here
    sitk.WriteImage(img1, path+'//'+'Prediction_num_'+str(i)+".dcm")

# this will adjust the prediction slices with the original MRI
MRI_size_H=384
MRI_size_D=160
for m in range (0,num_mris):
    pre01=pre_test[m, 0,0,:, :, :]
    pre011=np.zeros((MRI_size_D,MRI_size_H,MRI_size_H))
    pre022=np.zeros((MRI_size_D,MRI_size_H,MRI_size_H))
    for i in range (0,MRI_size_D):
        imgt1=pre01[MRI_size_D-1-i,:,:]
        pre011[i,:,:]=cv2.resize(imgt1,dsize=(MRI_size_H,MRI_size_H),interpolation=cv2.INTER_CUBIC)
    for i in range (0,MRI_size_H):
        imgt1=pre011[:,:,i]
        pre022[:,:,i]=cv2.resize(imgt1,dsize=(MRI_size_H,MRI_size_D),interpolation=cv2.INTER_CUBIC)
    for ii in range (0,MRI_size_D):
        for jj in range (0,MRI_size_H):
            for kk in range(0,MRI_size_H):
                if pre022[ii,jj,kk]>=0.5:
                    pre022[ii,jj,kk]=1
    pre022=pre022.astype(np.uint8)
    path=r'D:\Cartilage_seg'; #address to save the mri
    save_DICOM(m,pre022,path)
