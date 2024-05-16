# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Jiaqi Guo
"""

import numpy as np
import scipy.misc
import os
from PIL import Image
# import PIL.Image as Image
from torchvision import transforms
import torch.nn as nn
# from config import INPUT_SIZE
import scipy.io as io
import imageio
import random
import torch.utils.data as data
import shutil

INPUT_SIZE = (448, 448)


class RS_Scene():
    def __init__(self, root, mode='train', data_len=None):  # train, val, test
        self.root = root
        self.mode = mode
        train_imgs = []
        train_labels = []
        val_imgs = []
        val_labels = []
        test_imgs = []

        if self.mode == 'train':
            contents = os.listdir(os.path.join(self.root, 'train'))
            for i in range(len(contents)):
                tmp_path = os.path.join(self.root, 'train', contents[i])
                tmp_contents = os.listdir(tmp_path)
                for j in range(len(tmp_contents)):
                    tmp_img_path = os.path.join(tmp_path, tmp_contents[j])
                    train_imgs.append(tmp_img_path)
                    train_labels.append(i)
            self.train_imgs = train_imgs
            self.train_labels = train_labels

        if self.mode == 'val':
            contents = os.listdir(os.path.join(self.root, 'val'))
            for i in range(len(contents)):
                tmp_path = os.path.join(self.root, 'val', contents[i])
                tmp_contents = os.listdir(tmp_path)
                for j in range(len(tmp_contents)):
                    tmp_img_path = os.path.join(tmp_path, tmp_contents[j])
                    val_imgs.append(tmp_img_path)
                    val_labels.append(i)
            self.val_imgs = val_imgs
            self.val_labels = val_labels

        if self.mode == 'test':  # 测试集没有标签，注意
            contents = os.listdir(os.path.join(self.root, 'test'))
            for i in range(len(contents)):
                tmp_path = os.path.join(self.root, 'test', contents[i])
                test_imgs.append(tmp_path)
            self.test_imgs = test_imgs

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_imgs[index], self.train_labels[index]
            # print(img)
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            # print(img.shape)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            return img, target

        if self.mode == 'val':
            img, target = self.val_imgs[index], self.val_labels[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            return img, target

        if self.mode == 'test':
            img = self.test_imgs[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            return img

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_labels)
        if self.mode == 'val':
            return len(self.val_labels)
        if self.mode == 'test':
            return len(self.test_labels)


class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class CUB_():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class fgvc_air_family():
    def __init__(self, root, is_training=True, data_len=None):
        self.root = root
        self.is_training = is_training

        name_path = os.path.join(self.root, 'families.txt')
        trainval_path = os.path.join(self.root, 'images_family_trainval.txt')
        test_path = os.path.join(self.root, 'images_family_test.txt')

        with open(name_path, 'r') as f:
            names = f.readlines()
        tmp_names = []
        tmp_inds = []
        for index, name in enumerate(names):
            tmp_inds.append(index)
            name = name.strip('\n')
            tmp_names.append(name)
            print(index)
            print(name)
        labels = dict(zip(tmp_names, tmp_inds))

        with open(trainval_path, 'r') as f:
            trainval_images = f.readlines()
        tmp_trainval_images = []
        tmp_trainval_names = []
        for index, trainval_images_names in enumerate(trainval_images):
            trainval_images_names = trainval_images_names.strip('\n')
            trainval_image = trainval_images_names.split(' ', 1)[0]
            name = trainval_images_names.split(' ', 1)[1]
            tmp_trainval_images.append(trainval_image)
            tmp_trainval_names.append(name)
        trainval_images_names = dict(zip(tmp_trainval_images, tmp_trainval_names))

        with open(test_path, 'r') as f:
            test_images = f.readlines()
        tmp_test_images = []
        tmp_test_names = []
        for index, test_images_names in enumerate(test_images):
            test_images_names = test_images_names.strip('\n')
            test_image = test_images_names.split(' ', 1)[0]
            name = test_images_names.split(' ', 1)[1]
            tmp_test_images.append(test_image)
            tmp_test_names.append(name)
        test_images_names = dict(zip(tmp_test_images, tmp_test_names))

        trainval_images_labels_dict = trainval_images_names
        test_images_labels_dict = test_images_names
        labels_dict = labels

        images_dir = os.path.join(self.root, 'images')
        trainval_contents = list(trainval_images_labels_dict.keys())
        test_contents = list(test_images_labels_dict.keys())
        class_count = len(labels_dict)

        if self.is_training:
            trainval_images = []
            trainval_labels = []

            for i in range(len(trainval_images_labels_dict)):
                tmp_image = os.path.join(images_dir, trainval_contents[i])
                tmp_image = tmp_image + '.jpg'
                trainval_images.append(tmp_image)
                tmp_labels = np.zeros(class_count, dtype=np.float32)
                labels_name = trainval_images_labels_dict[trainval_contents[i]]
                labels_ind = labels_dict[labels_name]
                tmp_labels[labels_ind] = 1.0
                #                trainval_labels.append(tmp_labels)
                trainval_labels.append(labels_ind)
            #            trainval_images = np.asarray(trainval_images)
            #            trainval_labels = np.asarray(trainval_labels)

            self.train_img = trainval_images
            self.train_label = trainval_labels

        if not self.is_training:
            test_images = []
            test_labels = []

            for i in range(len(test_images_labels_dict)):
                tmp_image = os.path.join(images_dir, test_contents[i])
                tmp_image = tmp_image + '.jpg'
                test_images.append(tmp_image)
                tmp_labels = np.zeros(class_count, dtype=np.float32)
                labels_name = test_images_labels_dict[test_contents[i]]
                labels_ind = labels_dict[labels_name]
                tmp_labels[labels_ind] = 1.0
                #                test_labels.append(tmp_labels)
                test_labels.append(labels_ind)
            #            test_images = np.asarray(test_images)
            #            test_labels = np.asarray(test_labels)

            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class fgvc_air_manufacturers():
    def __init__(self, root, is_training=True, data_len=None):
        self.root = root
        self.is_training = is_training

        name_path = os.path.join(self.root, 'manufacturers.txt')
        trainval_path = os.path.join(self.root, 'images_manufacturer_trainval.txt')
        test_path = os.path.join(self.root, 'images_manufacturer_test.txt')

        with open(name_path, 'r') as f:
            names = f.readlines()
        tmp_names = []
        tmp_inds = []
        for index, name in enumerate(names):
            tmp_inds.append(index)
            name = name.strip('\n')
            tmp_names.append(name)
            print(index)
            print(name)
        labels = dict(zip(tmp_names, tmp_inds))

        with open(trainval_path, 'r') as f:
            trainval_images = f.readlines()
        tmp_trainval_images = []
        tmp_trainval_names = []
        for index, trainval_images_names in enumerate(trainval_images):
            trainval_images_names = trainval_images_names.strip('\n')
            trainval_image = trainval_images_names.split(' ', 1)[0]
            name = trainval_images_names.split(' ', 1)[1]
            tmp_trainval_images.append(trainval_image)
            tmp_trainval_names.append(name)
        trainval_images_names = dict(zip(tmp_trainval_images, tmp_trainval_names))

        with open(test_path, 'r') as f:
            test_images = f.readlines()
        tmp_test_images = []
        tmp_test_names = []
        for index, test_images_names in enumerate(test_images):
            test_images_names = test_images_names.strip('\n')
            test_image = test_images_names.split(' ', 1)[0]
            name = test_images_names.split(' ', 1)[1]
            tmp_test_images.append(test_image)
            tmp_test_names.append(name)
        test_images_names = dict(zip(tmp_test_images, tmp_test_names))

        trainval_images_labels_dict = trainval_images_names
        test_images_labels_dict = test_images_names
        labels_dict = labels

        images_dir = os.path.join(self.root, 'images')
        trainval_contents = list(trainval_images_labels_dict.keys())
        test_contents = list(test_images_labels_dict.keys())
        class_count = len(labels_dict)

        if self.is_training:
            trainval_images = []
            trainval_labels = []

            for i in range(len(trainval_images_labels_dict)):
                tmp_image = os.path.join(images_dir, trainval_contents[i])
                tmp_image = tmp_image + '.jpg'
                trainval_images.append(tmp_image)
                tmp_labels = np.zeros(class_count, dtype=np.float32)
                labels_name = trainval_images_labels_dict[trainval_contents[i]]
                labels_ind = labels_dict[labels_name]
                tmp_labels[labels_ind] = 1.0
                #                trainval_labels.append(tmp_labels)
                trainval_labels.append(labels_ind)
            #            trainval_images = np.asarray(trainval_images)
            #            trainval_labels = np.asarray(trainval_labels)

            self.train_img = trainval_images
            self.train_label = trainval_labels

        if not self.is_training:
            test_images = []
            test_labels = []

            for i in range(len(test_images_labels_dict)):
                tmp_image = os.path.join(images_dir, test_contents[i])
                tmp_image = tmp_image + '.jpg'
                test_images.append(tmp_image)
                tmp_labels = np.zeros(class_count, dtype=np.float32)
                labels_name = test_images_labels_dict[test_contents[i]]
                labels_ind = labels_dict[labels_name]
                tmp_labels[labels_ind] = 1.0
                #                test_labels.append(tmp_labels)
                test_labels.append(labels_ind)
            #            test_images = np.asarray(test_images)
            #            test_labels = np.asarray(test_labels)

            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class fgvc_air_variants():
    def __init__(self, root, is_training=True, data_len=None):
        self.root = root
        self.is_training = is_training

        name_path = os.path.join(self.root, 'variants.txt')
        trainval_path = os.path.join(self.root, 'images_variant_trainval.txt')
        test_path = os.path.join(self.root, 'images_variant_test.txt')

        with open(name_path, 'r') as f:
            names = f.readlines()
        tmp_names = []
        tmp_inds = []
        for index, name in enumerate(names):
            tmp_inds.append(index)
            name = name.strip('\n')
            tmp_names.append(name)
            print(index)
            print(name)
        labels = dict(zip(tmp_names, tmp_inds))

        with open(trainval_path, 'r') as f:
            trainval_images = f.readlines()
        tmp_trainval_images = []
        tmp_trainval_names = []
        for index, trainval_images_names in enumerate(trainval_images):
            trainval_images_names = trainval_images_names.strip('\n')
            trainval_image = trainval_images_names.split(' ', 1)[0]
            name = trainval_images_names.split(' ', 1)[1]
            tmp_trainval_images.append(trainval_image)
            tmp_trainval_names.append(name)
        trainval_images_names = dict(zip(tmp_trainval_images, tmp_trainval_names))

        with open(test_path, 'r') as f:
            test_images = f.readlines()
        tmp_test_images = []
        tmp_test_names = []
        for index, test_images_names in enumerate(test_images):
            test_images_names = test_images_names.strip('\n')
            test_image = test_images_names.split(' ', 1)[0]
            name = test_images_names.split(' ', 1)[1]
            tmp_test_images.append(test_image)
            tmp_test_names.append(name)
        test_images_names = dict(zip(tmp_test_images, tmp_test_names))

        trainval_images_labels_dict = trainval_images_names
        test_images_labels_dict = test_images_names
        labels_dict = labels

        images_dir = os.path.join(self.root, 'images')
        trainval_contents = list(trainval_images_labels_dict.keys())
        test_contents = list(test_images_labels_dict.keys())
        class_count = len(labels_dict)

        if self.is_training:
            trainval_images = []
            trainval_labels = []

            for i in range(len(trainval_images_labels_dict)):
                tmp_image = os.path.join(images_dir, trainval_contents[i])
                tmp_image = tmp_image + '.jpg'
                trainval_images.append(tmp_image)
                tmp_labels = np.zeros(class_count, dtype=np.float32)
                labels_name = trainval_images_labels_dict[trainval_contents[i]]
                labels_ind = labels_dict[labels_name]
                tmp_labels[labels_ind] = 1.0
                #                trainval_labels.append(tmp_labels)
                trainval_labels.append(labels_ind)
            #            trainval_images = np.asarray(trainval_images)
            #            trainval_labels = np.asarray(trainval_labels)

            self.train_img = trainval_images
            self.train_label = trainval_labels

        if not self.is_training:
            test_images = []
            test_labels = []

            for i in range(len(test_images_labels_dict)):
                tmp_image = os.path.join(images_dir, test_contents[i])
                tmp_image = tmp_image + '.jpg'
                test_images.append(tmp_image)
                tmp_labels = np.zeros(class_count, dtype=np.float32)
                labels_name = test_images_labels_dict[test_contents[i]]
                labels_ind = labels_dict[labels_name]
                tmp_labels[labels_ind] = 1.0
                #                test_labels.append(tmp_labels)
                test_labels.append(labels_ind)
            #            test_images = np.asarray(test_images)
            #            test_labels = np.asarray(test_labels)

            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class standford_dogs(data.Dataset):
    def __init__(self, root, is_training=True, data_len=None):
        self.root = root
        self.is_training = is_training

        train_path = os.path.join(self.root, 'train_list.mat')
        test_path = os.path.join(self.root, 'test_list.mat')

        if self.is_training:
            train_contents = io.loadmat(train_path)

            train_images_path = train_contents['file_list']
            train_images = []
            for i in range(len(train_images_path)):
                tmp_path = train_images_path[i][0][0]
                tmp_path = os.path.join(self.root, 'Images', tmp_path)
                train_images.append(tmp_path)

            train_labels = train_contents['labels']
            train_labels = train_labels[:, 0]
            train_labels -= 1
            train_labels = list(train_labels)

            self.train_img = train_images
            self.train_label = train_labels

        if not self.is_training:
            test_contents = io.loadmat(test_path)

            test_images_path = test_contents['file_list']
            test_images = []
            for i in range(len(test_images_path)):
                tmp_path = test_images_path[i][0][0]
                tmp_path = os.path.join(self.root, 'Images', tmp_path)
                test_images.append(tmp_path)

            test_labels = test_contents['labels']
            test_labels = test_labels[:, 0]
            test_labels -= 1
            test_labels = list(test_labels)

            self.test_img = test_images
            self.test_label = test_labels

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            #            img = self.pil_loader(img)
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class standford_cars():
    def __init__(self, root, is_training=True, data_len=None):
        self.root = root
        self.is_training = is_training

        train_images_path = os.path.join(self.root, 'cars_train')
        test_images_path = os.path.join(self.root, 'cars_test')
        mats_path = os.path.join(self.root, 'devkit')

        train_path = os.path.join(mats_path, 'cars_train_annos.mat')
        test_path = os.path.join(mats_path, 'cars_test_annos_withlabels.mat')

        if self.is_training:
            train_contents = io.loadmat(train_path)
            train_images = []
            train_labels = []

            train_list = train_contents['annotations']
            train_list = list(train_list[0])

            for i in range(len(train_list)):
                tmp_dir = train_list[i][5][0]
                tmp_dir = os.path.join(train_images_path, tmp_dir)
                train_images.append(tmp_dir)
                #                tmp_label = np.zeros(196, dtype=np.float32)
                tmp_label_val = train_list[i][4][0][0]
                #                tmp_label[tmp_label_val-1] = 1.0
                train_labels.append(tmp_label_val - 1)

            self.train_img = train_images
            self.train_label = train_labels

        if not self.is_training:
            test_contents = io.loadmat(test_path)

            test_images = []
            test_labels = []

            test_list = test_contents['annotations']
            test_list = list(test_list[0])

            for i in range(len(test_list)):
                tmp_dir = test_list[i][5][0]
                tmp_dir = os.path.join(test_images_path, tmp_dir)
                test_images.append(tmp_dir)
                #                tmp_label = np.zeros(196, dtype=np.float32)
                tmp_label_val = test_list[i][4][0][0]
                #                tmp_label[tmp_label_val-1] = 1.0
                test_labels.append(tmp_label_val - 1)

            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class food101():
    def __init__(self, root, is_training=True, data_len=None):
        self.root = root
        self.is_training = is_training

        images_path = os.path.join(self.root, 'images')
        data_dir = os.path.join(self.root, 'meta')

        with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
            classes = f.readlines()

        tmp_classes = []
        tmp_ind = []

        for index, name in enumerate(classes):
            tmp_ind.append(index)
            name = name.strip('\n')
            tmp_classes.append(name)
        labels = dict(zip(tmp_classes, tmp_ind))

        if self.is_training:
            with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
                train_content = f.readlines()

            train_images = []
            train_labels = []
            for i in range(len(train_content)):
                tmp = train_content[i].strip('\n')
                tmp_path = os.path.join(images_path, tmp)
                tmp_path = tmp_path + '.jpg'
                train_images.append(tmp_path)

                tmp = tmp.split('/')[0]
                #                tmp_labels = np.zeros(101, dtype = np.float32)
                ind = labels[tmp]
                #                tmp_labels[ind] = 1.0
                train_labels.append(ind)

            self.train_img = train_images
            self.train_label = train_labels

        if not self.is_training:
            with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
                test_content = f.readlines()

            test_images = []
            test_labels = []

            for i in range(len(test_content)):
                tmp = test_content[i].strip('\n')
                tmp_path = os.path.join(images_path, tmp)
                tmp_path = tmp_path + '.jpg'
                test_images.append(tmp_path)

                tmp = tmp.split('/')[0]
                #                tmp_labels = np.zeros(101, dtype = np.float32)
                ind = labels[tmp]
                #                tmp_labels[ind] = 1.0
                test_labels.append(ind)

            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class NWPU():
    def __init__(self, root, is_training=True, ratio=0.5):
        self.root = root
        self.is_training = is_training

        image_path = self.root

        contents = os.listdir(self.root)

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for i in range(len(contents)):
            index = 0
            sub_cont = os.listdir(os.path.join(image_path, contents[i]))
            train_len = np.floor(ratio * len(sub_cont))
            for j in range(len(sub_cont)):
                tmp_path = os.path.join(image_path, contents[i], sub_cont[j])
                if index < train_len:
                    train_images.append(tmp_path)
                    train_labels.append(i)

                else:
                    test_images.append(tmp_path)
                    test_labels.append(i)
                index += 1

        if self.is_training:
            train_images = np.asarray(train_images)
            train_labels = np.asarray(train_labels)
            ind = np.random.permutation(len(train_labels))
            train_images = train_images[ind]
            train_labels = train_labels[ind]

            self.train_img = list(train_images)
            self.train_label = list(train_labels)
        if not self.is_training:
            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            # img = scipy.misc.imread(img)
            img = imageio.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class MITINDOOR():
    def __init__(self, root, is_training=True, ratio=0.5):
        self.root = root
        self.is_training = is_training

        image_path = os.path.join(self.root, 'Images')

        contents = os.listdir(image_path)

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for i in range(len(contents)):
            index = 0
            sub_cont = os.listdir(os.path.join(image_path, contents[i]))
            train_len = np.floor(ratio * len(sub_cont))
            for j in range(len(sub_cont)):
                tmp_path = os.path.join(image_path, contents[i], sub_cont[j])
                if index < train_len:
                    train_images.append(tmp_path)
                    train_labels.append(i)

                else:
                    test_images.append(tmp_path)
                    test_labels.append(i)
                index += 1

        if self.is_training:
            train_images = np.asarray(train_images)
            train_labels = np.asarray(train_labels)
            ind = np.random.permutation(len(train_labels))
            train_images = train_images[ind]
            train_labels = train_labels[ind]

            self.train_img = list(train_images)
            self.train_label = list(train_labels)
        if not self.is_training:
            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class UIUC_Texture():
    def __init__(self, root, is_training=True, ratio=0.5):
        self.root = root
        self.is_training = is_training

        image_path = self.root

        contents = os.listdir(image_path)

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for i in range(len(contents)):
            index = 0
            sub_cont = os.listdir(os.path.join(image_path, contents[i]))
            train_len = np.floor(ratio * len(sub_cont))
            for j in range(len(sub_cont)):
                if sub_cont[j] == 'Thumbs.db':
                    continue
                tmp_path = os.path.join(image_path, contents[i], sub_cont[j])
                if index < train_len:
                    train_images.append(tmp_path)
                    train_labels.append(i)

                else:
                    test_images.append(tmp_path)
                    test_labels.append(i)
                index += 1

        if self.is_training:
            train_images = np.asarray(train_images)
            train_labels = np.asarray(train_labels)
            ind = np.random.permutation(len(train_labels))
            train_images = train_images[ind]
            train_labels = train_labels[ind]

            self.train_img = list(train_images)
            self.train_label = list(train_labels)
        if not self.is_training:
            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


class DTD_Texture():
    def __init__(self, root, is_training=True, ratio=0.5):
        self.root = root
        self.is_training = is_training

        image_path = self.root

        contents = os.listdir(image_path)

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for i in range(len(contents)):
            index = 0
            sub_cont = os.listdir(os.path.join(image_path, contents[i]))
            train_len = np.floor(ratio * len(sub_cont))
            for j in range(len(sub_cont)):
                if sub_cont[j] == 'Thumbs.db':
                    continue
                tmp_path = os.path.join(image_path, contents[i], sub_cont[j])
                if index < train_len:
                    train_images.append(tmp_path)
                    train_labels.append(i)

                else:
                    test_images.append(tmp_path)
                    test_labels.append(i)
                index += 1

        if self.is_training:
            train_images = np.asarray(train_images)
            train_labels = np.asarray(train_labels)
            ind = np.random.permutation(len(train_labels))
            train_images = train_images[ind]
            train_labels = train_labels[ind]

            self.train_img = list(train_images)
            self.train_label = list(train_labels)
        if not self.is_training:
            self.test_img = test_images
            self.test_label = test_labels

    def __getitem__(self, index):
        if self.is_training:
            img, target = self.train_img[index], self.train_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = scipy.misc.imread(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_training:
            return len(self.train_label)
        else:
            return len(self.test_label)


def split_and_save():
    root = 'E:/dataset/CUB_200_2011/CUB_200_2011'
    output_dir = 'E:/dataset/CUB_200_2011_split'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_air = CUB(root, is_train=True)
    train_dir = os.path.join(output_dir, 'train')
    for i in range(len(dataset_air.train_label)):
        temp_dir = os.path.join(train_dir, str(dataset_air.train_label[i]))
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_img_dir = dataset_air.train_img[i]
        shutil.copy(temp_img_dir, temp_dir)

    dataset_air = CUB(root, is_train=False)
    test_dir = os.path.join(output_dir, 'test')
    for i in range(len(dataset_air.test_label)):
        temp_dir = os.path.join(test_dir, str(dataset_air.test_label[i]))
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_img_dir = dataset_air.test_img[i]
        shutil.copy(temp_img_dir, temp_dir)

    return 1


# %%
if __name__ == '__main__':
    #     dataset = CUB(root='E:/dataset/CUB_200_2011/CUB_200_2011')
    #     print(len(dataset.train_img))
    #     print(len(dataset.train_label))
    # #    for data in dataset:
    # #        print(data[0].size(), data[1])
    #     dataset = CUB(root='E:/dataset/CUB_200_2011/CUB_200_2011', is_train=False)
    #     print(len(dataset.test_img))
    #     print(len(dataset.test_label))
    # #    for data in dataset:
    # #        print(data[0].size(), data[1])
    #
    #     dataset_air = fgvc_air_family(root='E:/dataset/fgvc-aircraft-2013b/data')
    #
    #     dataset_dogs = standford_dogs(root='E:/dataset/Stanford_Dog_120')
    #
    #     dataset_cars = standford_cars(root='E:/dataset/CAR-196')
    #
    #     dataset_food = food101(root='E:/dataset/food_101')
    #
    #     dataset_indoor = MITINDOOR(root='I:/study/database/indoorCVPR_09')
    #
    #     dataset_uiucTex = UIUC_Texture(root='I:/study/database/uiuc_texture_dataset')
    #
    #     dataset_dtdTex = DTD_Texture(root='H:/study_backup/database/DTD/dtd-r1.0.1/dtd/images')
    dataset = RS_Scene(root='C:/zhaozy/datasets/remote_sensing/rssrai2019_scene_classification', mode='val')