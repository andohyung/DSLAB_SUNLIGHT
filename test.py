import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils._utils import make_data_loader, make_test_loader
#from model import BaseModel
from torchvision.models import resnet50
from sklearn.metrics import precision_recall_fscore_support
from torchvision.models import efficientnet_b4
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights

def test(args, test_loader, model):
    true = np.array([])
    pred = np.array([])
    
    model.eval()
    
    pbar = tqdm(test_loader)
    for i, (x, y) in enumerate(pbar):
        
        image = x.to(args.device)
        label = y.to(args.device)                

        output = model(image)
        
        label = label.squeeze()
        output = output.argmax(dim=-1)
        output = output.detach().cpu().numpy()
        pred = np.append(pred,output, axis=0)
        
        label = label.detach().cpu().numpy()
        true =  np.append(true,label, axis=0)
        
    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSLAB_INTERN')
    parser.add_argument('--save_path', default='/home/pink/intern/project_el/dohyung/checkpoints/model.pth', help="Model's state_dict")
    parser.add_argument('--train_csv', default='/home/pink/intern/project_el/datasets/classification/train/re_data_df_first.csv', type=str, help='train csv')
    parser.add_argument('--valid_csv', default='/home/pink/intern/project_el/datasets/classification/valid/re_data_df_first.csv', type=str, help='valid csv')
    parser.add_argument('--test_csv', default='/home/pink/intern/project_el/datasets/classification/test/re_data_df_first.csv', type=str, help='test csv')
    parser.add_argument('--img_dir', default='/home/pink/intern/project_el/datasets/first', type=str, help='image')
    args = parser.parse_args()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # hyperparameters
    args.batch_size = 8
    
    # Make Data loader and Model
    test_loader = make_test_loader(args)

    # instantiate model
    # model = BaseModel()
    model = resnet34(num_classes=2)
    #model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model.load_state_dict(torch.load(args.save_path))
    model = model.to(device)
    
    # Test The Model
    pred, true = test(args, test_loader, model)
        
    #추가
    # Calculate Precision, Recall, and F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary')

    print("Precision: {:.5f}".format(precision))
    print("Recall: {:.5f}".format(recall))
    print("F1 Score: {:.5f}".format(f1))    
        
    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))
    print([(i, int(p), int(t)) for i, (p, t) in enumerate(zip(pred, true)) if p != t])

    