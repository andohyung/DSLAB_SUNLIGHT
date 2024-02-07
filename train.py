import argparse

import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
# from utils._utils_jiwon import make_data_loader
#from model import BaseModel
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision.models import efficientnet_b4, efficientnet_b3
from efficientnet_pytorch import EfficientNet
#from utils._utils import crop_and_save_images
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, valid_loader, model):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
            
        model.train()
        pbar = tqdm(data_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)          
            optimizer.zero_grad()

            output = model(image)
            
            #label = label.squeeze()
            loss = criterion(output, label)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        
        writer.flush()
        # writer.close()
        
        #valid code 추가
        val_losses = [] 
        val_acc = 0.0
        val_total=0
        # best_loss = 10 ** 9 # 매우 큰 값으로 초기값 가정
        # patience_limit = 1 # 몇 번의 epoch까지 지켜볼지를 결정
        # patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
        
        model.eval()
        pbar1 = tqdm(valid_loader)
        with torch.no_grad():
            for i, (x,y) in enumerate(pbar1):
                image_val = x.to(args.device)
                label_val = y.to(args.device)
                #optimizer.zero_grad()
                
                output_val = model(image_val)
                
                #label_val = label_val.squueze()
                loss_val = criterion(output_val, label_val)
                
                val_losses.append(loss_val.item())
                val_total += label_val.size(0)
                
                val_acc += acc(output_val, label_val)
                
            epoch_val_loss = np.mean(val_losses)
            epoch_val_acc = val_acc / val_total

        # ### early stopping 여부를 체크하는 부분 ###
        # if loss_val > best_loss: # loss가 개선되지 않은 경우
        #     atience_check += 1

        #     if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
        #         break

        # else: # loss가 개선된 경우
        #     best_loss = loss_val
        #     patience_check = 0
        
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        
        #print(f'Validation - Epoch {epoch + 1}')
        print(f'valid_loss : {epoch_val_loss}')
        print('valid_accuracy : {:.3f}'.format(epoch_val_acc * 100))
        #
        
        torch.save(model.state_dict(), f'{args.save_path}/effimodel.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DSLAB_INTERN')
    parser.add_argument('--save_path', default='/home/pink/intern/project_el/dohyung/checkpoints', help="checkpoint")
    parser.add_argument('--train_csv', default='/home/pink/intern/project_el/datasets/classification/train/re_data_df_first.csv', type=str, help='train csv')
    parser.add_argument('--valid_csv', default='/home/pink/intern/project_el/datasets/classification/valid/re_data_df_first.csv', type=str, help='valid csv')
    parser.add_argument('--test_csv', default='/home/pink/intern/project_el/datasets/classification/test/re_data_df_first.csv', type=str, help='test csv')
    parser.add_argument('--img_dir', default='/home/pink/intern/project_el/datasets/first', type=str, help='image')

    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    args.device = device
    
    # hyperparameters
    args.epochs = 7
    args.learning_rate = 0.0001
    args.batch_size = 8

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, valid_loader = make_data_loader(args)

    # custom model
    # model = BaseModel()
    
    # torchvision model
    #model = resnet34(weights=ResNet34_Weights)
    #model = efficientnet_b4(pretrained=True)
    
    # you have to change num_classes to 10
    # model.fc.out_features=10 => 이러면 fc가 안 바뀌어요!
    #model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)# 다시 선언을 해주어야 바뀌어요!
    model.to(device)
    print(model)

    # folder_path = "/home/pink/intern/project_el/datasets/first/non_fault"
    # file_list = os.listdir(folder_path)
    # file_count = len(file_list)
    # print(file_count)
    
    # source_folder = "/home/pink/intern/project_el/datasets/first/non_fault"  # 소스 폴더
    # destination_folder = "/home/pink/intern/project_el/datasets/adh/non_fault"  # 목적지 폴더
    
    # crop_and_save_images(source_folder, destination_folder)
    
    # Training The Model
    train(args, train_loader, valid_loader, model)