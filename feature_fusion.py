from tqdm import tqdm
from os.path import join
import os
import configparser
import sys
import numpy as np
import pandas as pd
import torch
import random
import argparse
import jsonlines
import pickle
import preprocess as text_preprocess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src import *
import warnings
from feature_extraction import *

EXP_GROUP = "gnn"
RESULT_FILE = "results/{}/{}/{}/{}.json"

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", default=None, type=int, required=False)
parser.add_argument("--arch", default="sage-base", type=str, required=False)
parser.add_argument("--epochs", default=1000, type=int, required=False)
parser.add_argument("--lr", default=1e-5, type=float, required=False)
parser.add_argument("--weight_decay", default=1e-3, type=float, required=False)
parser.add_argument("--dropout", default=0.5, type=float, required=False)
parser.add_argument("--batch_size", default=32, type=int, required=False)
parser.add_argument("--num_workers", default=4, type=int, required=False)
parser.add_argument("--n_neigh_train", default=16, type=int, required=False,
                    help="n_neighbours hp for train KNN graph")
parser.add_argument("--n_neigh_full", default=16, type=int, required=False,
                    help="n_neighbours hp for full KNN graph")
parser.add_argument("--lbl_train_frac", default=0.4, type=float, required=True)
parser.add_argument("--imagepath", default=None, type=str, required=False,
                    help="image path of the experiment")
parser.add_argument("--datasplit", default=None, type=str,
                    help="dataset path of the experiment")
parser.add_argument("--reg", default=None, type=str,
                    help="regularization")
parser.add_argument("--l2_lambda", default=None, type=float)
parser.add_argument("--exp_id", default=None, type=int, required=True)
parser.add_argument("--loss", default="nll", type=str, required=False, choices=['nll', 'bce'])
parser.add_argument("--shuffle_split", action='store_true', required=False)
parser.add_argument("--best_model", default="best_val", type=str, required=False, choices=['best_val'])
parser.add_argument("--dataset", default='CrisisMMD', type=str, choices=["CrisisMMD"])
parser.add_argument("--imageft", default='mobilenet', type=str, choices=['clip', 'maxvit', 'mobilenet'])
parser.add_argument("--textft", default='mpnet', type=str, choices=['clip', 'mpnet'])
parser.add_argument("--fusion", default="early", choices=['early', 'middle', 'late'])
parser.add_argument("--reduction", default=None, type=str, required=False, choices=['pca'])

args_cl = parser.parse_args()

dev_id = args_cl.gpu_id
if dev_id is not None:
    device = torch.device('cuda:{}'.format(dev_id) if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

use_gpu = device != 'cpu'

torch.manual_seed(13)
np.random.seed(13)
random.seed(13)

imagepath = args_cl.imagepath
datasplit = args_cl.datasplit

datapath_ssl = "./data/CrisisMMD_v2.0_baseline_split/data_splits_ssl/{}/informative_orig"
datapath = datapath_ssl.format(datasplit)

kwargs = vars(args_cl)
del kwargs["datasplit"]
del kwargs["imagepath"]

imageft = args_cl.imageft
textft = args_cl.textft

# Training Step

# Training Graph Creation

[df_text_train, df_text_dev, df_text_test] = mpnet_features()
[df_image_train, df_image_dev, df_image_test] = mobilenet_features()

df_image_train = df_image_train.set_index(['image_files','text'])
df_text_train = df_text_train.set_index(['image_files','text'])
df_image_dev = df_image_dev.set_index(['image_files','text'])
df_text_dev = df_text_dev.set_index(['image_files','text'])
df_image_test = df_image_test.set_index(['image_files','text'])
df_text_test = df_text_test.set_index(['image_files','text'])

df_image_train_dict = df_image_train.to_dict("index")
df_text_train_dict = df_text_train.to_dict("index")
df_image_dev_dict = df_image_dev.to_dict("index")
df_text_dev_dict = df_text_dev.to_dict("index")
df_image_test_dict = df_image_test.to_dict("index")
df_text_test_dict = df_text_test.to_dict("index")

# labeled
data_labeled = [obj for obj in jsonlines.open(join(datapath, "train.jsonl"))]
image_files = []
texts = []
annot_labeled = []
for row in tqdm(data_labeled):
    image_files.append(join(imagepath, row['image']))
    # Applie Preprocess stage here

    #texts.append(text_preprocess.pre_process(row['text'], keep_hashtag = True, keep_special_symbols = False))
    texts.append(row['text'])

    annot_labeled.append(row['label']) 

annot_labeled = [ (1,0) if l == 'not_informative' else (0,1) for l in annot_labeled ]
annot_labeled = np.array(annot_labeled)

ft_labeled_images = torch.zeros([len(data_labeled), len(df_image_train["embeddings"].iloc[0])])
for ix, (img, txt) in enumerate(zip(image_files, texts)):
    ft_labeled_images[ix] = torch.Tensor(df_image_train_dict[(img, txt)]["embeddings"])
ft_labeled_text = torch.zeros([len(data_labeled), len(df_text_train["embeddings"].iloc[0])])
for ix, (img, txt) in enumerate(zip(image_files, texts)):
    ft_labeled_text[ix] = torch.Tensor(df_text_train_dict[(img, txt)]["embeddings"])

# unlabeled
data_unlbl = [obj for obj in jsonlines.open(join(datapath, "unlabeled.jsonl"))]
image_files = []
texts = []
annot_unlabeled = []
for row in tqdm(data_unlbl):
    image_files.append(join(imagepath, row['image']))
    # Applie Preprocess stage here

    #texts.append(text_preprocess.pre_process(row['text'], keep_hashtag = True, keep_special_symbols = False))
    texts.append(row['text'])

    annot_unlabeled.append(row['label']) 

annot_unlabeled = [ (1,0) if l == 'not_informative' else (0,1) for l in annot_unlabeled ]
annot_unlabeled = np.array(annot_unlabeled)

ft_unlabeled_images = torch.zeros([len(data_unlbl), len(df_image_train["embeddings"].iloc[0])])
for ix, (img, txt) in enumerate(zip(image_files, texts)):
    ft_unlabeled_images[ix] = torch.Tensor(df_image_train_dict[(img, txt)]["embeddings"])
ft_unlabeled_text = torch.zeros([len(data_unlbl), len(df_text_train["embeddings"].iloc[0])])
for ix, (img, txt) in enumerate(zip(image_files, texts)):
    ft_unlabeled_text[ix] = torch.Tensor(df_text_train_dict[(img, txt)]["embeddings"])

# dev
data_dev = [obj for obj in jsonlines.open(join(datapath, "dev.jsonl"))]
image_files = []
texts = []
annot_dev = []
for row in tqdm(data_dev):
    image_files.append(join(imagepath, row['image']))
    # Applie Preprocess stage here

    #texts.append(text_preprocess.pre_process(row['text'], keep_hashtag = True, keep_special_symbols = False))
    texts.append(row['text'])

    annot_dev.append(row['label']) 

annot_dev = [ (1,0) if l == 'not_informative' else (0,1) for l in annot_dev ]
annot_dev = np.array(annot_dev)

ft_dev_images = torch.zeros([len(data_dev), len(df_image_dev["embeddings"].iloc[0])])
for ix, (img, txt) in enumerate(zip(image_files, texts)):
    ft_dev_images[ix] = torch.Tensor(df_image_dev_dict[(img, txt)]["embeddings"])
ft_dev_text = torch.zeros([len(data_dev), len(df_text_dev["embeddings"].iloc[0])])
for ix, (img, txt) in enumerate(zip(image_files, texts)):
    ft_dev_text[ix] = torch.Tensor(df_text_dev_dict[(img, txt)]["embeddings"])

if kwargs.get("reduction") == 'pca':
    print("dimensionality reduction")
    # image
    full_ft_images = np.concatenate([ft_labeled_images, ft_unlabeled_images], axis=0)

    pca = PCA(n_components=512)
    pca.fit(full_ft_images)

    ft_labeled_images = pca.transform(ft_labeled_images)
    ft_unlabeled_images = pca.transform(ft_unlabeled_images)
    ft_dev_images = pca.transform(ft_dev_images)
    
    # text
    full_ft_text = np.concatenate([ft_labeled_text, ft_unlabeled_text], axis=0)

    pca = PCA(n_components=512)
    pca.fit(full_ft_text)

    ft_labeled_text = pca.transform(ft_labeled_text)
    ft_unlabeled_text = pca.transform(ft_unlabeled_text)
    ft_dev_text = pca.transform(ft_dev_text)

if torch.is_tensor(ft_labeled_images) is False:
    ft_labeled_images = torch.Tensor(ft_labeled_images)
if torch.is_tensor(ft_labeled_text) is False:
    ft_labeled_text = torch.Tensor(ft_labeled_text)
if torch.is_tensor(annot_labeled) is False:
    annot_labeled = torch.Tensor(annot_labeled)
if torch.is_tensor(ft_unlabeled_images) is False:
    ft_unlabeled_images = torch.Tensor(ft_unlabeled_images)
if torch.is_tensor(ft_unlabeled_text) is False:
    ft_unlabeled_text = torch.Tensor(ft_unlabeled_text)
if torch.is_tensor(annot_unlabeled) is False:
    annot_unlabeled = torch.Tensor(annot_unlabeled)
if torch.is_tensor(ft_dev_images) is False:
    ft_dev_images = torch.Tensor(ft_dev_images)
if torch.is_tensor(ft_dev_text) is False:
    ft_dev_text = torch.Tensor(ft_dev_text)
if torch.is_tensor(annot_dev) is False:
    annot_dev = torch.Tensor(annot_dev)

ft_labeled = torch.concat([ft_labeled_images, ft_labeled_text], dim=1)
ft_unlabeled = torch.concat([ft_unlabeled_images, ft_unlabeled_text], dim=1)
ft_dev = torch.concat([ft_dev_images, ft_dev_text], dim=1)

ft_labeled_copy = ft_labeled.clone()
ft_unlabeled_copy = ft_unlabeled.clone()
ft_dev_copy = ft_dev.clone()

annot_labeled = torch.argmax(annot_labeled, dim=1)
annot_unlabeled = torch.argmax(annot_unlabeled, dim=1)
annot_dev = torch.argmax(annot_dev, dim=1)

ft_mt_training_step = torch.concat([ft_labeled, ft_unlabeled])
annot_mt_training_step = torch.concat([annot_labeled, annot_unlabeled])

labeled_ix = np.arange(0, ft_labeled.shape[0])
unlabeled_ix = np.arange(ft_labeled.shape[0], ft_mt_training_step.shape[0])

n_neigh_train = kwargs.get("n_neigh_train")
pyg_graph_train = generate_graph(ft_mt_training_step, annot_mt_training_step, n_neighbors=n_neigh_train, labeled_ix=labeled_ix, unlabeled_ix=unlabeled_ix, edge_attr=kwargs.get("edge_attr"))

image_ft_size = ft_labeled_images.shape[1]
text_ft_size = ft_labeled_text.shape[1]

# model instance
if args_cl.fusion == "late":
    model = BaseGNNLateFusion(ft_mt_training_step.shape[1], 2, args_cl.arch, dropout=args_cl.dropout)
elif args_cl.fusion == "middle":
    model = BaseGNNMiddleFusionConcat(ft_mt_training_step.shape[1], 2, args_cl.arch, image_size=image_ft_size, text_size=text_ft_size, dropout=args_cl.dropout)
elif args_cl.fusion == "early":
    model = BaseGNN(ft_mt_training_step.shape[1], 2, args_cl.arch, dropout=args_cl.dropout)
print("Model instanciated")

if use_gpu:
    model = model.to(device)
    pyg_graph_train = pyg_graph_train.to(device)
    print("Model and Data on GPU")

# train

best_model = run_base_v2(model, pyg_graph_train, **kwargs)

if kwargs.get("tb") is True:
    del kwargs["datasplit"]

# dev-test step

# dev-test graph
ft_labeled = ft_labeled_copy
ft_unlabeled = ft_unlabeled_copy
ft_dev = ft_dev_copy

ft_mt_dev_step = torch.concat([ft_labeled, ft_unlabeled, ft_dev])
annot_mt_dev_step = torch.concat([annot_labeled, annot_unlabeled, annot_dev])

labeled_ix = np.arange(0, ft_labeled.shape[0])
unlabeled_ix = np.arange(ft_labeled.shape[0], ft_labeled.shape[0] + ft_unlabeled.shape[0])
dev_ix = np.arange(ft_labeled.shape[0] + ft_unlabeled.shape[0], ft_mt_dev_step.shape[0])

n_neigh_full = kwargs.get("n_neigh_full")
pyg_graph_dev = generate_graph(ft_mt_dev_step, annot_mt_dev_step, n_neigh_full, labeled_ix=labeled_ix, unlabeled_ix=unlabeled_ix, test_ix=dev_ix, edge_attr=kwargs.get("edge_attr"))

if use_gpu:
    pyg_graph_train = pyg_graph_train.to("cpu")
    pyg_graph_dev = pyg_graph_dev.to(device)

exp_id = args_cl.exp_id
dataset = args_cl.dataset
# test on dev

os.makedirs(RESULT_FILE[:RESULT_FILE.rfind("/")].format(dataset, EXP_GROUP, exp_id), exist_ok=True)
validate_best_model(model, pyg_graph_dev, RESULT_FILE.format(dataset, EXP_GROUP, exp_id, datasplit), edge_attr=kwargs.get("edge_attr"))

if use_gpu:
    model = model.to('cpu')
    pyg_graph_dev = pyg_graph_dev.to('cpu')
    print("Model and Data on CPU")