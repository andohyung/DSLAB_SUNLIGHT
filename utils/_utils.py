import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image as Image
import os
from tqdm import tqdm

#Define Transform
custom_transform = transforms.Compose([
    transforms.Resize((1000, 429)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# add_transform = transforms.Compose([
#     transforms.RandomRotation(degrees=30),  # 랜덤한 각도로 회전 (예: 최대 30도)
#     transforms.RandomResizedCrop(size=(1000, 429), scale=(0.8, 1.0)), # 랜덤한 크기와 비율로 잘라내기
#     transforms.RandomHorizontalFlip(),  # 랜덤하게 수평으로 뒤집기
#     transforms.RandomVerticalFlip(),  # 랜덤하게 수직으로 뒤집기
# ])

# # Combine the original transform with additional transforms
# full_transform = transforms.Compose([
#     custom_transform,
#     add_transform
# ])

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file (string): CSV 파일의 경로.
        root_dir (string): 모든 이미지가 있는 디렉토리의 경로.
        transform (callable, optional): 샘플에 적용할 선택적 변환.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        #self.X=[]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename, label = self.df.iloc[idx,0], self.df.iloc[idx,1]
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.root_dir, label_str, filename)
        
        #수정 전 부분은 카톡에
        
        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        return img, label

# Function to Create Data Loaders
def make_data_loader(args):
    # Create Custom Dataset
    train_dataset = CustomImageDataset(args.train_csv, args.img_dir, transform=custom_transform)
    valid_dataset = CustomImageDataset(args.valid_csv, args.img_dir, transform=custom_transform)
    #test_dataset = CustomImageDataset(args.test_csv, args.img_dir, transform=custom_transform)

    # Create Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 16)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 16)
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 8)
    return train_loader , valid_loader


def make_test_loader(args):
    test_dataset = CustomImageDataset(args.test_csv, args.img_dir, transform=custom_transform)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 16)
    return test_loader






# def crop_and_save_images(source_folder, destination_folder):
#     # 만약 목적지 폴더가 없다면 생성
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)

#     # 소스 폴더에 있는 모든 파일 목록 가져오기
#     file_list = os.listdir(source_folder)

#     # tqdm을 사용하여 진행 상황 표시
#     with tqdm(total=len(file_list), desc="Processing Images") as pbar:
#         for j, file_name in enumerate(file_list):
#             # 이미지 파일만 처리
#             if file_name.lower().endswith('.jpg'):
#                 # 이미지 로드
#                 image_path = os.path.join(source_folder, file_name)
#                 image = Image.open(image_path)

#                 # 이미지의 윗부분을 잘라내기 (예시로 20% 자름)
#                 width, height = image.size
#                 cropped_image = image.crop((0, 0, width, int(0.98 * height)))
                
#                 # 새로운 경로 설정하여 저장
#                 destination_path = os.path.join(destination_folder, f"{file_name}")
#                 cropped_image.save(destination_path)

#             pbar.update(1)
##
# if __name__ == "__main__":
#     source_folder = "/home/pink/intern/project_el/datasets/first/fault"  # 소스 폴더
#     destination_folder = "/home/pink/intern/project_el/datasets/adh/fault"  # 목적지 폴더

#     crop_and_save_images(source_folder, destination_folder)