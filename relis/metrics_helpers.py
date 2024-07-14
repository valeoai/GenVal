''' Adapted from https://github.com/naver/oasis/blob/master/metrics_helpers.py '''

import numpy as np
import argparse
import json
from PIL import Image
import imageio
from os.path import join
from sklearn import preprocessing

with open('./dataset/cityscapes_list/info.json', 'r') as f:
    info = json.load(f)


IDD_TO_CITYSCAPES_MAPPING = {0:0, 2:1, 22:2, 14:3, 15:4, 20:5, 19:6, 18:7, 24:8, None:9, 25:10, 4:11,
                             5:12, 9:13, 10:14, 11:15, None:16, 6:17, 7:18}


def fast_hist(a_, b_, n_):
    k_ = (a_ >= 0) & (a_ < n_)
    return np.bincount(n_ * a_[k_].astype(int) + b_[k_], minlength=n_ ** 2).reshape(n_, n_)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir='./dataset/cityscapes_list'):
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int64(info['classes'])
    print(f'Num classes {num_classes}')
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int64)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred_image = Image.open(pred_imgs[ind])
        pred = np.array(pred_image) # pred: (1024, 2048)
        label_image = Image.open(gt_imgs[ind])
        label = np.array(label_image) # label: (1024, 2048)
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print(f'Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}')
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print(f'===> mIoU: {round(np.nanmean(mIoUs) * 100, 2)}')
    return mIoUs


def compute_acc_single_image(label, pred):
    """
    Compute pixel and mean accuracy given a predicted colorized image and the GT
    (also GT in color format, not label format)
    """

    assert len(label.flatten()) == len(pred.flatten())

    pred = pred[label!=255]
    label = label[label!=255]
    
    
    correct_pred = (label == pred)
    pixel_acc = 100. * np.sum(correct_pred)/len(correct_pred)

    image_labels = np.unique(label)
    mean_acc = 0.
    for cl in image_labels:
        tmp_correct_pred = (label[label==cl] == pred[label==cl])
        mean_acc += (1./len(image_labels)) * np.sum(tmp_correct_pred)/len(tmp_correct_pred)

    return pixel_acc, mean_acc


def compute_mIoU_single_image(label, pred, args):
    """
    Compute IoU given a predicted colorized image and the GT
    (also GT in color format, not label format)
    """

    if len(label.flatten()) != len(pred.flatten()):
        print('Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(len(label.flatten()), len(pred.flatten())))
        return -1

    hist = fast_hist(label.flatten(), pred.flatten(), args.num_classes)
    #sample_mIoU = 100*np.nanmean(per_class_iu(hist))

    mIoUs = per_class_iu(hist)

    return mIoUs


def compute_acc_fromlist(gt_imgs, pred_imgs, args):
    """
    Compute pixel and mean accuracy given the predicted colorized images and
    """

    assert len(gt_imgs) == len(pred_imgs)

    pred_imgs_list = []
    gt_imgs_list = []

    le = preprocessing.LabelEncoder()

    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}
    
    for ind in range(len(gt_imgs)):
        pred_image = Image.open(pred_imgs[ind])
        pred = np.array(pred_image) # pred: (1024, 2048)

        if ('Cityscapes' in args.trg_dataset) or ('ACDC' in args.trg_dataset):
            mapping = np.array(info['label2train'], dtype=np.int64)
            label_image = Image.open(gt_imgs[ind])
            label = np.array(label_image) # label: (1024, 2048)
            label = label_mapping(label, mapping)
        
        elif 'IDD' in args.trg_dataset:
            label_image = Image.open(gt_imgs[ind])
            label_image = np.array(label_image)
            label = 255 * np.ones(label_image.shape, dtype=np.float32)
            for k, v in IDD_TO_CITYSCAPES_MAPPING.items():
                label[label_image == k] = v
            
        else:
            raise NotImplementedError("Unknown target dataset")

        if len(label.flatten()) != len(pred.flatten()):
            print(f'Skipping: len(gt) = {len(label.flatten())}',
                  f'len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}')
            continue

        pred = pred[label!=255]
        label = label[label!=255]
        correct += np.sum(label==pred)
        total += len(label)
        
        image_labels = np.unique(label)
        for cl in image_labels:
            if cl in per_class_correct.keys():
                per_class_correct[cl] += np.sum(label[label==cl] == pred[label==cl])
                per_class_total[cl] += np.sum(label==cl)
            else:
                per_class_correct[cl] = np.sum(label[label==cl] == pred[label==cl])
                per_class_total[cl] = np.sum(label==cl)
        
    mean_acc = 0
    for cl in per_class_correct.keys():
        mean_acc += 1/len(per_class_correct.keys()) * per_class_correct[cl] / per_class_total[cl]
        
    pixel_acc = correct / total * 100
    
    return pixel_acc, mean_acc



def compute_mIoU_fromlist(gt_imgs, pred_imgs, args):
    """
    Compute IoU given the predicted colorized images and
    """
    num_classes = args.num_classes
    name_classes = args.name_classes

    hist = np.zeros((num_classes, num_classes))

    for ind in range(len(gt_imgs)):
        pred_image = Image.open(pred_imgs[ind])
        pred = np.array(pred_image)

        if ('Cityscapes' in args.trg_dataset) or ('ACDC' in args.trg_dataset):
            mapping = np.array(info['label2train'], dtype=np.int64)
            label_image = Image.open(gt_imgs[ind])
            label = np.array(label_image)
            label = label_mapping(label, mapping)
        elif 'IDD' in args.trg_dataset:
            label_image = Image.open(gt_imgs[ind])
            label_image = np.array(label_image)
            label = 255 * np.ones(label_image.shape, dtype=np.float32)
            for k, v in IDD_TO_CITYSCAPES_MAPPING.items():
                label[label_image == k] = v
        else:
            raise NotImplementedError("Unknown target dataset")
        
        if len(label.flatten()) != len(pred.flatten()):
            print(f'Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}')
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.nanmean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    
    # Only take into account present classes in the ground truth (especifically for IDD)
    if 'IDD' in args.trg_dataset:
        # All classes present except 9 and 16.
        present_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18]
        for ind_class in range(num_classes):
            if ind_class not in present_classes:
                mIoUs[ind_class] = np.nan

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print(f'===> mIoU: {round(np.nanmean(mIoUs) * 100, 2)}')

    return mIoUs


def main(args):
    compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)
