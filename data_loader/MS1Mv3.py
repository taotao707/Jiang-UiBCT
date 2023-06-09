"""
    Face Recognition dataset
"""
import os
import numbers
import torch
import mxnet as mx
from mxnet.recordio import MXIndexedRecordIO
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sampler import DistributedClassSampler, SubsetRandomSampler
from torchvision import transforms

def default_flist_reader(flist):
    """
    flist format: impath label\n impath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            if len(line.strip().split())==2:
                impath, imlabel = line.strip().split()[:2]
                imlist.append((int(impath), int(imlabel)))
            elif len(line.strip().split())==3:
                impath, imlabel, fakelabel = line.strip().split()[:3]
                imlist.append((int(impath), int(imlabel), int(fakelabel)))
                class_num = fakelabel
            else:
                raise NotImplementedError
    return imlist, int(class_num)+1

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, img_list, transform=None, flist_reader=default_flist_reader,):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        patn_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = MXIndexedRecordIO(patn_imgidx, path_imgrec, "r")

        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s) 

        if type(img_list) is str:
            self.img_list, self.class_num = flist_reader(img_list)
        else:
            self.img_list, self.class_num = img_list, 93431

        #EADER(flag=2, label=array([5179511., 5272942.], dtype=float32), id=0, id2=0)
        if header.flag > 0:
            print("header0 label:", header.label)
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = list(range(1, int(header.label[0])))
            # print(self.imgidx)
        else:
            self.imgidx = list(self.imgrec.keys)
        print("Number of Samples:{} Number of Classes: {}".format(len(self.imgidx), int(self.header0[1] - self.header0[0])))

    def __getitem__(self, index): 
        fake_label = self.img_list[index][-1]
        index = self.img_list[index][0]
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label.astype(int), dtype=torch.int)
        # print(label)

        img = mx.image.imdecode(img).asnumpy()  # RGB
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fake_label

    def __len__(self): 
        # print(len(self.imgidx))
        return len(self.img_list)
    
        

def generate_train_dataloder(data_set, distributed=False, batch_size=64, num_workers=32,
                       pin_memory=True, use_pos_sampler=False):
    if distributed:
        if use_pos_sampler:
            sampler = DistributedClassSampler(dataset=data_set, num_instances=2)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    else:
        sampler = None
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=(sampler is None),
                        pin_memory=pin_memory, num_workers=num_workers, sampler=sampler)
    return loader

def MS1Mv3_train_dataloader(traindir, img_list, train_transform, distributed=False,
                           batch_size=64, num_workers=32, pin_memory=True,
                           use_pos_sampler=False):
    train_set = MXFaceDataset(traindir, img_list, train_transform)
    class_num = train_set.class_num
    train_loader = generate_train_dataloder(train_set, distributed, batch_size,
                                      num_workers, pin_memory, use_pos_sampler)
    return train_loader, class_num


def IJBC_test_dataloader():
    pass


if __name__ == '__main__':
    #MS1Mv3数据集测试
    # root_dir = 'E:/HHUC design Representation Learning/dataset_select/ms1m-retinaface-t1'
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # train_trans = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # trainset = MXFaceDataset(root_dir, transform=train_trans)
    # num_dataset = len(trainset)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    # for batch_idx, (sample, label) in enumerate(train_loader):
    #     print(sample.shape, label)
    #     exit()

    root_dir = '/ProjectRoot/webpage_pretrain/runner_jlt/dataset/ms1m-retinaface-t1/ms1m-retinaface-t1'
    img_list = '/ProjectRoot/webpage_pretrain/runner_jlt/dataset/ms1m-retinaface-t1/ms1m-retinaface-t1/ms1mv3_train_old_30percent_class_openclass.txt'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    # train_set = MXFaceDataset(root_dir, img_list, train_trans)
    # train_set.imgrec.close()
    
    train_loader, class_num = MS1Mv3_train_dataloader(root_dir, img_list, train_trans, distributed=False,
                           batch_size=1, num_workers=0, pin_memory=False,
                           use_pos_sampler=False)
    # del train_loader
    print(len(train_loader))    
    print(class_num)                       
    for batch_idx, (sample, label) in enumerate(train_loader):
        print(sample.shape, label)
        if batch_idx==5:
            exit()