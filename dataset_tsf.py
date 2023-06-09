import torch
from auxiliary.auxiliary_word2vec import classes2embedding, load_word2vec
from auxiliary.auxiliary_dataset_tsf import VideoDataset, get_ucf101, get_hmdb


def get_datasets(opt, dtype='all'):
    datasets, embedding = get_train_datasets(opt, dtype)
    dataloaders = {}
    for key, datasets in datasets.items():
        dataloader = []
        for dataset in datasets:
            dl = torch.utils.data.DataLoader(dataset,
                      batch_size=opt.bs,
                      num_workers=8, shuffle=True, drop_last=False)
            dataloader.append(dl)
        dataloaders[key] = dataloader
    return dataloaders, embedding


def get_train_datasets(opt, dtype):
    wv_model = load_word2vec()
    if opt.dataset == 'ucf':
        train_fnames, train_labels, test_fnames, test_labels, classes = get_ucf101(opt,dtype)
        class_embedding = classes2embedding('ucf101', classes, wv_model)
        print('UCF101: classes {}, train number of videos {}, test number of videos {}'.format(len(classes), len(train_fnames), len(test_fnames)))

    if opt.dataset == 'hmdb':
        train_fnames, train_labels, test_fnames, test_labels, classes = get_hmdb(opt,dtype)
        class_embedding = classes2embedding('hmdb51', classes, wv_model)
        print('HMDB51: classes {}, train number of videos {}, test number of videos {}'.format(len(classes), len(train_fnames), len(test_fnames)))

    train_dataset = VideoDataset(train_fnames, train_labels, class_embedding, classes,
                                 clip_len=opt.clip_len, n_clips=opt.n_clips,
                                 crop_size=opt.size, is_validation=False)

    val_dataset = VideoDataset(test_fnames, test_labels, class_embedding, classes,
                                 clip_len=opt.clip_len, n_clips=opt.n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    
    all_fnames = train_fnames+test_fnames
    all_labels = train_labels+test_labels
    all_dataset = VideoDataset(all_fnames, all_labels, class_embedding, classes,
                                    clip_len=opt.clip_len, n_clips=opt.n_clips, crop_size=opt.size, 
                                    is_validation=False,
                                    evaluation_only=opt.evaluate)
    return {'training': [train_dataset], 'testing': [val_dataset], 'all': [all_dataset]}, class_embedding
