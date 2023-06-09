import numpy as np, cv2, torch
from auxiliary.auxiliary_word2vec import classes2embedding, load_word2vec
from auxiliary.auxiliary_dataset import VideoDataset, get_ucf101, get_hmdb, get_ucf101_val, get_hmdb_val, get_ucf101_train, get_hmdb_train


def get_datasets(opt, dtype='all'):
    get_datasets = get_both_datasets(opt, dtype)

    datasets = get_datasets

    # Move datasets to dataloaders.
    dataloaders = {}
    for key, datasets in datasets.items():
        dataloader = []
        for dataset in datasets:
            dl = torch.utils.data.DataLoader(dataset,
                      batch_size=opt.bs // 2 if (not dataset.is_validation and 'image' in opt.dataset and opt.class_total != 0) else opt.bs,
                      num_workers=8, shuffle=not dataset.is_validation, drop_last=False)
            dataloader.append(dl)
        dataloaders[key] = dataloader
    return dataloaders


def get_both_datasets(opt, dtype):
    wv_model = load_word2vec()

    if opt.evaluate:
        # TESTING ON UCF101
        test_fnames, test_labels, test_classes = get_ucf101()
        test_class_embedding = classes2embedding('ucf101', test_classes, wv_model)
        print('Test UCF101: total number of videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

        # TESTING ON HMDB51
        test_fnames2, test_labels2, test_classes2 = get_hmdb()
        test_class_embedding2 = classes2embedding('hmdb51', test_classes2, wv_model)
        print('Test HMDB51: total number of videos {}, classes {}'.format(len(test_fnames2), len(test_classes2)))


    else:
        # VAL ON UCF101
        test_fnames, test_labels, test_classes = get_ucf101_val()
        test_class_embedding = classes2embedding('ucf101', test_classes, wv_model)
        print('Val UCF101: total number of videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

        # VAL ON HMDB51
        test_fnames2, test_labels2, test_classes2 = get_hmdb_val()
        test_class_embedding2 = classes2embedding('hmdb51', test_classes2, wv_model)
        print('Val HMDB51: total number of videos {}, classes {}'.format(len(test_fnames2), len(test_classes2)))

        if opt.dataset == 'ucf':
        # TRAINING ON UCF101
            train_fnames, train_labels, train_classes = get_ucf101_train(opt, dtype)
            train_class_embedding = classes2embedding('ucf101', train_classes, wv_model)
            print('Train UCF101: total number of videos {}, classes {}'.format(len(train_fnames), len(train_classes)))

        elif opt.dataset == 'hmdb':
        # TRAINING ON HMDB51
            train_fnames, train_labels, train_classes = get_hmdb_train(opt, dtype)
            train_class_embedding = classes2embedding('hmdb51', train_classes, wv_model)
            print('Train HMDB51: total number of videos {}, classes {}'.format(len(train_fnames), len(train_classes)))

        # Initialize datasets
        dname = {'ucf':'ucf101', 'hmdb':'hmdb51'}
        train_dataset = VideoDataset(train_fnames, train_labels, train_class_embedding, train_classes,
                                     dname[opt.dataset], clip_len=opt.clip_len, n_clips=opt.n_clips,
                                     crop_size=opt.size, is_validation=False)

    val_dataset   = VideoDataset(test_fnames, test_labels, test_class_embedding, test_classes, 'ucf101',
                                 clip_len=opt.clip_len, n_clips=opt.n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    val_dataset2  = VideoDataset(test_fnames2, test_labels2, test_class_embedding2, test_classes2, 'hmdb51',
                                 clip_len=opt.clip_len, n_clips=opt.n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    if opt.evaluate:
        return {'training': [], 'testing': [val_dataset, val_dataset2]}
    else:
        return {'training': [train_dataset], 'testing': [val_dataset, val_dataset2]}
