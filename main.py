import os, numpy as np, argparse, multiprocessing
from tqdm import tqdm
import torch
import torch.nn as nn
import network
import dataset
import dataset_tsf
from auxiliary.transforms import batch2gif
from auxiliary.train_test_split import source_classes_
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats.mstats import hmean
from sklearn.cluster import *

"""=========================INPUT ARGUMENTS====================="""

parser = argparse.ArgumentParser()

# ##### Network parameters
parser.add_argument('--network', default='r2plus1d_18', type=str,
                    help='Network backend choice: [r2plus1d_18, r3d, c3d].')

parser.add_argument('--dataset',      default='ucf',   type=str)
parser.add_argument('--train_samples',  default=-1,  type=int, help='Reduce number of train samples to the given value')
parser.add_argument('--class_total',  default=-1,  type=int, help='For debugging only. Reduce the total number of classes')
parser.add_argument('--clip_len',     default=16,   type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips',     default=1,   type=int, help='Number of clips per video')
parser.add_argument('--class_overlap', default=0.05,  type=float, help='tau. see Eq.3 in main paper')

### General Training Parameters
parser.add_argument('--lr',           default=5e-6, type=float, help='Learning Rate for network parameters.')
parser.add_argument('--n_epochs',     default=10,   type=int,   help='Number of training epochs.')
parser.add_argument('--n_epochs_stage1', default=15,   type=int,   help='Number of training stage1 epochs.')
parser.add_argument('--bs',           default=22,   type=int,   help='Mini-Batchsize size per GPU.')
parser.add_argument('--size',         default=112,  type=int,   help='Image size in input.')
parser.add_argument('--fixconvs', action='store_true', default=False,   help='Freezing conv layers')
parser.add_argument('--nopretrained', action='store_false', default=True,   help='Pretrain network.')

### Paths to datasets and storage folder
parser.add_argument('--save_path',    default='workplace/', type=str, help='Where to save log and checkpoint.')
parser.add_argument('--weights',      default=None, type=str, help='Weights to load from a previously run.')
parser.add_argument('--progressbar', action='store_true', default=False,   help='Show progress bar during train/test.')
parser.add_argument('--evaluate', action='store_true', default=False,   help='Evaluation only using 25 clips per video')
parser.add_argument('--sp',        default=0,   type=int)
parser.add_argument('--stage', default=0, type=int,   help='Train stage.')
parser.add_argument('--pretrain', default='k700', type=str)
parser.add_argument('--tsa', action='store_true', default=False)

##### Read in parameters
opt = parser.parse_args()
opt.multiple_clips = False
opt.kernels = multiprocessing.cpu_count()
torch.backends.cudnn.enabled = False

r2plus1d_k700_pre = '/root/data1/sty/TLZSAR/pretarinmodel/r2plus1d_18/zsl_r2plus1d18_kinetics700_ucf101_hmdb51_checkpoint.pth.tar'
pretrainmodel = {'r2plus1d':{'k700':r2plus1d_k700_pre}}

"""=================================SETUPS==============================="""


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    opt.bs = opt.bs * torch.cuda.device_count()
print('Total batch size: %d' % opt.bs)

opt.device = torch.device('cuda')
criterion = nn.MSELoss().to(opt.device)
celoss = nn.CrossEntropyLoss()
softmax_func = nn.Softmax(dim=1)

source_classes = source_classes_[opt.sp]
target_classes = []
for i in range(101):
    if i not in source_classes:
        target_classes.append(i)

if opt.dataset == 'hmdb':
    source_index = [index for index in source_classes if index<51]
    target_index = [index for index in target_classes if index<51]
    if len(source_index) > 26:
        tmp_id = 26 - len(source_index)
        target_index += source_index[tmp_id:]
        source_index = source_index[:tmp_id]
    elif len(source_index) < 26:
        tmp_id = 26 - len(target_index)
        source_index += target_index[tmp_id:]
        target_index = target_index[:tmp_id]
    source_classes = source_index
    target_classes = target_index

loader = {'ucf':0, 'hmdb':1}


"""===========================TRAINER FUNCTION==============================="""


def QFSLloss(y, label):
    source = [v for v,l in enumerate(label) if l in source_classes]
    target = [v for v,l in enumerate(label) if l in target_classes]
    if len(list(source)) == 0:
        y_soft = softmax_func(y)[target,:]
        y_target = y_soft[:, target_classes]
        y_sum = torch.sum(y_target,dim=1)
        loss2 = torch.mean(-torch.log(y_sum))
        return torch.tensor(0).to(opt.device), loss2
    
    if len(list(target)) == 0 :
        loss1 = celoss(y[source,:],label[source])
        return loss1, torch.tensor(0).to(opt.device)
    
    else:
        loss1 = celoss(y[source,:],label[source])
        y_soft = softmax_func(y)[target,:]
        y_target = y_soft[:, target_classes]
        y_sum = torch.sum(y_target,dim=1)
        loss2 = torch.mean(-torch.log(y_sum))
        return loss1, loss2


def train_one_epoch(train_dataloader, model, optimizer, opt, epoch, stage, training):
    if training:
        model.train()
    else:
        model.eval()
    class_embedding = train_dataloader.dataset.class_embed
    acc_reg_source, total_loss = [], []
    total_celoss, total_ploss = [],[]

    data_iterator = train_dataloader
    if opt.progressbar:
        if training:
            data_iterator = tqdm(train_dataloader, ncols=0, desc='Epoch {} Training...'.format(epoch))
        else:
            data_iterator = tqdm(train_dataloader, ncols=0, desc='Epoch {} Test...'.format(epoch))

    for i, (X, classid, Z, _) in enumerate(data_iterator):
        Y = model(X.to(opt.device))
        Z = Z.to(opt.device)
        classid = classid.to(opt.device)
        if stage == 0:
            pred_embed = Y.detach().cpu().numpy()
            pred_label = cdist(pred_embed, class_embedding, 'cosine').argmin(1)
            acc_source = accuracy_score(classid.cpu().detach().numpy(), pred_label) * 100
            acc_reg_source.append(acc_source)
            loss = criterion(Y, Z)
            total_loss.append(loss.cpu().detach().numpy())

        else:
            celoss, ploss = QFSLloss(Y, classid)
            loss = celoss + 1.0*ploss
            total_celoss.append(celoss.cpu().detach().numpy())
            total_ploss.append(ploss.cpu().detach().numpy())

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if stage == 0:
        total_loss = np.mean(total_loss)
        acc_reg_source = np.mean(acc_reg_source)
    else:
        total_celoss = np.mean(total_celoss)
        total_ploss = np.mean(total_ploss)

    if training:
        if stage == 0:
            print('Train loss: {}'.format(total_loss))
            print('Train Accuracy: {:.2f}%'.format(acc_reg_source))
        else:
            print('Train celoss: {}, ploss: {}'.format(total_celoss, total_ploss))

    else:
        if stage == 0:
            print('Test loss: {}'.format(np.mean(total_loss)))
            print('Test Accuracy: {:.2f}%'.format(acc_reg_source))
        else:
            print('Test celoss: {}, ploss: {}'.format(total_celoss, total_ploss))


"""===========================EVALUATION==============================="""


def evaluate(test_dataloader, eval_model, sel_classes, yp=False,):
    name = test_dataloader.dataset.name
    _ = eval_model.eval()
    with torch.no_grad():
        ### For all test images, extract features
        n_samples = len(test_dataloader.dataset)

        predicted_embed = np.zeros([n_samples, 300], 'float32')
        true_embed = np.zeros([n_samples, 300], 'float32')
        true_label = np.zeros(n_samples, 'int')
        good_samples = np.zeros(n_samples, 'int') == 1

        final_iter = test_dataloader
        if 'features' not in opt.dataset and opt.progressbar:
            final_iter = tqdm(test_dataloader, ncols=0, desc='Evaluating...')

        fi = 0
        for idx, data in enumerate(final_iter):
            X, l, Z, _ = data
            not_broken = l != -1
            X, l, Z = X[not_broken], l[not_broken], Z[not_broken]
            if len(X) == 0: continue
            # Run network on batch
            Y = eval_model(X.to(opt.device))
            Y = Y.cpu().detach().numpy()
            l = l.cpu().detach().numpy()
            predicted_embed[fi:fi + len(l)] = Y
            true_embed[fi:fi + len(l)] = Z.squeeze()
            true_label[fi:fi + len(l)] = l.squeeze()
            good_samples[fi:fi + len(l)] = True
            fi += len(l)

    predicted_embed = predicted_embed[:fi]
    true_embed, true_label = true_embed[:fi], true_label[:fi]

    # Calculate accuracy over test classes
    class_embedding = test_dataloader.dataset.class_embed
    accuracy, accuracy_top5 = compute_accuracy(predicted_embed, class_embedding, true_embed)

    if opt.dataset == 'hmdb':
        sel_classes = [l for l in sel_classes if l<51]
    sel = [l in sel_classes for l in true_label]
    test_classes = len(sel_classes)
    # Compute accuracy
    acc, acc5 = compute_accuracy(predicted_embed[sel], class_embedding[sel_classes], true_embed[sel])
    accuracy1 = acc
    
    # target
    sel2 = [l not in sel_classes for l in true_label]
    sel_classes2 = []
    for kk in range(len(class_embedding)):
        if kk not in sel_classes:
            sel_classes2.append(kk)
    sel_classes2=np.array(sel_classes2)
    test_classes2 = len(class_embedding)-len(sel_classes)
    # Compute accuracy
    acc2, acc5_2 = compute_accuracy(predicted_embed[sel2], class_embedding[sel_classes2], true_embed[sel2])
    accuracy2 = acc2

    if not yp:
        # Compute hmean
        acc1, acc2, h, p, r, f1, p2, r2, f12, len_s, len_t, y_p, y_p2 = compute_hmean(predicted_embed, class_embedding, true_embed, sel, sel2, sel_classes, sel_classes2)
        print(' -- acc:%2.1f%% acc(s):%2.1f%% acc(t):%2.1f%% | s:%2.1f%% t:%2.1f%% h:%2.1f%% | p:%2.1f%% r:%2.1f%% f1:%2.1f%% | p(t):%2.1f%% r(t):%2.1f%% f1(t):%2.1f%% | num(s):%d/%d num(t):%d/%d' \
                % (accuracy, accuracy1.mean(), accuracy2.mean(), acc1, acc2, h, p, r, f1, p2, r2, f12, len_s, len(true_label), len_t, len(true_label) ))
        results = [accuracy, accuracy1.mean(), accuracy2.mean(), acc1, acc2, h, p, r, f1, p2, r2, f12, len_s, len(true_label), len_t, len(true_label)]
        return y_p, y_p2, results

    else:
        return predicted_embed, true_label


def compute_accuracy(predicted_embed, class_embed, true_embed):
    """
    Compute accuracy based on the closest Word2Vec class
    """
    assert len(predicted_embed) == len(true_embed), "True and predicted labels must have the same number of samples"
    y_pred = cdist(predicted_embed, class_embed, 'cosine').argsort(1)
    y = cdist(true_embed, class_embed, 'cosine').argmin(1)
    accuracy = accuracy_score(y, y_pred[:, 0]) * 100
    accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, :5])]) * 100
    return accuracy, accuracy_top5


"""========================================================="""


def compute_hmean(predicted_embed, class_embed, true_embed, sel, sel2, sel_classes, sel_classes2):
    """
    # czsl: search space: target class
    # gzsl: search space: all class
    """
    assert len(predicted_embed) == len(true_embed), "True and predicted labels must have the same number of samples"
    y_pred = cdist(predicted_embed, class_embed, 'cosine').argsort(1)
    y = cdist(true_embed, class_embed, 'cosine').argmin(1)

    accuracy = accuracy_score(y[sel], y_pred[sel, 0]) * 100 # source
    accuracy2 = accuracy_score(y[sel2], y_pred[sel2, 0]) * 100 # target
    acc_hmean = hmean([accuracy, accuracy2])

    y_p = np.array([l in sel_classes for l in y_pred[:, 0]])
    len_s = y_p.sum()
    p = precision_score(sel, y_p) * 100
    r = recall_score(sel, y_p) * 100
    f1 = f1_score(sel, y_p) * 100

    y_p2 = np.array([l in sel_classes2 for l in y_pred[:, 0]])
    len_t = y_p2.sum()
    p2 = precision_score(sel2, y_p2) * 100
    r2 = recall_score(sel2, y_p2) * 100
    f12 = f1_score(sel2, y_p2) * 100

    return accuracy, accuracy2, acc_hmean, p, r, f1, p2, r2, f12, len_s, len_t, y_p, y_p2


def load_model(opt, dataloaders, wt):
    embedding = dataloaders['testing'][loader[opt.dataset]].dataset.class_embed
    model = network.get_network(opt, embedding, stage=0)

    j = len('module.')
    if 'kinetics700' in wt:
        weights = torch.load(wt)['state_dict']
    else:
        weights = torch.load(wt)
    model_dict = model.state_dict()
    weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    print("LOADED MODEL:  ", wt)

    model = nn.DataParallel(model)
    _ = model.to(opt.device)
    return model


def load_tsf_model(model, wt):
    j = len('module.')
    if 'kinetics700' in wt:
        weights = torch.load(wt)['state_dict']
    else:
        weights = torch.load(wt)
    model_dict = model.state_dict()
    weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    print("load model:  ", wt)
    model = nn.DataParallel(model)
    model.to(opt.device)
    return model


def gen_target_wv(test_dataloader, model):
    model.eval()
    target_wv = []
    with torch.no_grad():
        data_iterator = test_dataloader
        if opt.progressbar:
            data_iterator = tqdm(test_dataloader, ncols=0, desc='Evaluate...')

        for i, (X, classid, Z, _) in enumerate(data_iterator):
            Y = model(X.to(opt.device))
            Y = Y.cpu().detach().numpy()
            label = np.array(classid)
            for j in range(label.shape[0]):
                YY = Y[j].reshape((Y[j].shape[0]))
                target_wv.append(YY)
    return target_wv


"""===================SCRIPT MAIN========================="""


def main_train():
    opt.savename = opt.save_path +opt.network+str(opt.lr)+'/'+ opt.dataset + opt.pretrain +'/sp'+str(opt.sp)
    print(opt.savename)
    if not os.path.exists(opt.savename):
        os.makedirs(opt.savename)

    dataloaders = dataset.get_datasets(opt, dtype='all')
    embedding = dataloaders['training'][0].dataset.class_embed
    save_name = opt.savename + '/checkpoint-stage1-'

    model = network.get_network(opt, embedding, stage=opt.stage)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if len(opt.pretrain) > 0:
        wt = pretrainmodel[opt.network][opt.pretrain]
        j = len('module.')
        if 'kinetics700' in wt:
            weights = torch.load(wt)['state_dict']
        else:
            weights = torch.load(wt)
        model_dict = model.state_dict()
        weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        print("LOADED MODEL:  ", wt)

    model = nn.DataParallel(model)
    _ = model.to(opt.device)

    for epoch in range(opt.n_epochs):
        print('epoch:',epoch)
        train_one_epoch(dataloaders['training'][0], model, optimizer, opt, epoch, opt.stage, training=True)
        torch.save(model.state_dict(), save_name+str(epoch)+'.pth.tar')


def main_test():
    dataloaders = dataset.get_datasets(opt, dtype='all')
    wtlist = []
    wtlist += [pretrainmodel[opt.network][opt.pretrain]]
    
    folder = opt.save_path +opt.network+str(opt.lr)+'/'+ opt.dataset + opt.pretrain +'/sp'+str(opt.sp)
    wtlist += [folder+'/'+ff for ff in sorted(os.listdir(str(folder)))]

    results_list = []

    for wt in wtlist[:]:
        model = load_model(opt, dataloaders, wt)
        y_p, y_p2, results = evaluate(dataloaders['testing'][loader[opt.dataset]], model, source_classes_[opt.sp])
        results_list.append(results)
    return results_list, wtlist[:]


def main_tsf_train():
    opt.savename = opt.save_path +opt.network+str(opt.lr)+'/'+ opt.dataset + opt.pretrain +'/sp'+str(opt.sp)
    if not os.path.exists(opt.savename):
        os.makedirs(opt.savename)
    pretrain = True
    stage = opt.stage
    if stage == 0:
        print('## source_learning:')
        dataloaders, embedding = dataset_tsf.get_datasets(opt, 'source')
        model = network.get_network(opt, embedding, stage=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        target_dataloaders, target_embedding = dataset_tsf.get_datasets(opt, 'target')
        if pretrain:
            weights = pretrainmodel[opt.network][opt.pretrain]
            model = load_tsf_model(model, weights)
        else:
            model = nn.DataParallel(model)
            model.to(opt.device)

        for epoch in range(opt.n_epochs):
            print('epoch:',epoch)
            train_one_epoch(dataloaders['training'][0], model, optimizer, opt, epoch, stage, training=True)
            torch.save(model.state_dict(), opt.savename + '/checkpoint-'+str(epoch)+'.pth.tar')
            

    print('## kmeans clustering')
    dataloaders, embedding = dataset_tsf.get_datasets(opt, 'target')
    model = network.get_network(opt, embedding, stage=0)
    weights = pretrainmodel[opt.network][opt.pretrain]
    model = load_tsf_model(model, weights)
    target_wv = gen_target_wv(dataloaders['training'][0], model)
    X=np.array(target_wv)
    kmeans = KMeans(n_clusters=len(target_classes)).fit(X)
    kk = 0
    for i in target_classes:
        embedding[i] = kmeans.cluster_centers_[kk]
        kk+=1

    stage = 1
    print('## transductive_learning')
    dataloaders, _ = dataset_tsf.get_datasets(opt, 'all')
    model = network.get_network(opt, embedding, stage=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    weights = opt.savename +'/checkpoint-'+str(int(opt.n_epochs)-1)+'.pth.tar'

    model = load_tsf_model(model, weights)

    for epoch in range(opt.n_epochs_stage1):
        print('epoch:',epoch)
        train_one_epoch(dataloaders['training'][0], model, optimizer, opt, epoch, 1, training=True)
        torch.save(model.state_dict(), opt.savename + '/checkpoint-stage1-'+str(epoch)+'.pth.tar')
        print('model save in:',opt.savename)


def main_tsf_test(wt):
    opt.savename = opt.save_path +opt.network+str(opt.lr)+'/'+ opt.dataset + opt.pretrain +'/sp'+str(opt.sp)
    print(opt.savename)

    dataloaders = dataset.get_datasets(opt, dtype='all')
    folder = opt.save_path +opt.network+str(opt.lr)+'/'+ opt.dataset + opt.pretrain +'/sp'+str(opt.sp)
    
    class_embedding = dataloaders['testing'][loader[opt.dataset]].dataset.class_embed

    sel_classes = source_classes_[opt.sp]
    if opt.dataset == 'hmdb':
        sel_classes = [l for l in sel_classes if l<51]
    sel_classes2 = []
    for kk in range(len(class_embedding)):
        if kk not in sel_classes:
            sel_classes2.append(kk)
    sel_classes2=np.array(sel_classes2)
    print(len(sel_classes),len(sel_classes2))

    eval_model2 = load_model(opt, dataloaders, wt['acc_s'])
    predicted_embed2, true_label2 = evaluate(dataloaders['testing'][loader[opt.dataset]],eval_model2, source_classes_[opt.sp], True)

    eval_model3 = load_model(opt, dataloaders, wt['acc_t'])
    predicted_embed3, true_label3 = evaluate(dataloaders['testing'][loader[opt.dataset]],eval_model3, source_classes_[opt.sp], True)
    

    eval_model1 = load_model(opt, dataloaders, wt['f_hmean'])
    y_p, y_p2, results = evaluate(dataloaders['testing'][loader[opt.dataset]],eval_model1, source_classes_[opt.sp])


    y_pred2 = cdist(predicted_embed2[y_p], class_embedding[sel_classes], 'cosine').argsort(1)
    ymap2 = dict(zip(np.arange(len(sel_classes)), sel_classes))
    pre1, gt1 = np.array([ymap2[l] for l in y_pred2[:, 0]]), true_label2[y_p]

    y_pred3 = cdist(predicted_embed3[y_p2], class_embedding[sel_classes2], 'cosine').argsort(1)
    ymap3 = dict(zip(np.arange(len(sel_classes2)), sel_classes2))
    pre2, gt2 = np.array([ymap3[l] for l in y_pred3[:, 0]]), true_label3[y_p2]

    y_pred = np.append(pre1,pre2)
    y = np.append(gt1,gt2)
    sel = np.array([l in source_classes_[opt.sp] for l in y])
    sel2 = np.array([l not in source_classes_[opt.sp] for l in y])
    
    accuracy = accuracy_score(y[sel], y_pred[sel]) * 100 # source
    accuracy2 = accuracy_score(y[sel2], y_pred[sel2]) * 100 # target
    acc_hmean = hmean([accuracy, accuracy2])
    
    y_p = np.array([l in sel_classes for l in y_pred])
    len_s = y_p.sum()
    p = precision_score(sel, y_p) * 100
    r = recall_score(sel, y_p) * 100
    f1 = f1_score(sel, y_p) * 100

    y_p2 = np.array([l in sel_classes2 for l in y_pred])
    len_t = y_p2.sum()
    p2 = precision_score(sel2, y_p2) * 100
    r2 = recall_score(sel2, y_p2) * 100
    f12 = f1_score(sel2, y_p2) * 100
    print(' ** s:%2.1f%% t:%2.1f%% h:%2.1f%% | p:%2.1f%% r:%2.1f%% f1:%2.1f%% | p(t):%2.1f%% r(t):%2.1f%% f1(t):%2.1f%% | num(s):%d/%d num(t):%d/%d' \
            % (accuracy,accuracy2,acc_hmean, p, r, f1, p2, r2, f12, len_s, len(y), len_t, len(y) ))


if __name__ == '__main__':
    if opt.tsa:
        main_train()
    else:
        main_tsf_train()

    results_list, wt_list = main_test()

    results_list = np.array(results_list)
    acc_s = np.argmax(results_list[:,1])
    acc_t = np.argmax(results_list[:,2])
    f1 = results_list[:,8]
    f1_t = results_list[:,11]
    f_hmean = np.argmax(hmean([f1,f1_t]))
    wt = {'acc_s':wt_list[acc_s], 'acc_t':wt_list[acc_t], 'f_hmean':wt_list[f_hmean]}

    opt.bs = opt.bs // 2
    print('test batch size:', opt.bs)
    opt.evaluate = True
    opt.n_clips = 8
    
    main_tsf_test(wt)