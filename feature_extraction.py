from PIL import ImageFile, Image
from tqdm import tqdm
import numpy as np
import jsonlines
import preprocess
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from src.forensic_lib.forensicsEvidences.imgEv.CNN import  load_default_image_model_or_preprocess, ImageDataset
from src.forensic_lib.utils.vector_utils import normalize_vector
from torch.utils.data import DataLoader
import torch
import timm
from os.path import join
import torchvision.models as models

DATAPATH = "./data/CrisisMMD_v2.0_baseline_split/data_splits/informative_orig/"
IMAGEPATH = "./data/CrisisMMD_v2.0/"

def mpnet_features():
    dfs = [None, None, None]
    for ix, split in enumerate(("train", "dev", "test")):
        data = [obj for obj in jsonlines.open(join(DATAPATH, split + ".jsonl"))]
        
        labels = []
        clean_text = [] 
        text = []
        image_files = []
    
        for row in tqdm(data):
            image_files.append(join(IMAGEPATH, row['image']))
            text.append(row['text'])
            clean_text.append(preprocess.pre_process(row['text'], keep_hashtag = True, keep_special_symbols = True))
            labels.append(row['label']) 
    
        # Get pretrained model
        pretrain_model_path = 'all-mpnet-base-v2'
        model = SentenceTransformer(pretrain_model_path, cache_folder='.')
    
        # Enccode text
        tweets_emds = model.encode(clean_text)
    
        labels = [ (1,0) if l == 'not_informative' else (0,1) for l in labels ]
        labels = np.array(labels)
    
        df = pd.DataFrame({"image_files": image_files,
                        "text": text,
                        "clean_tex": clean_text,
                        "embeddings": [f for f in tweets_emds],
                        "labels": np.argmax(labels, axis=1)})
                    
        dfs[ix] = df
    
    return dfs
    
def mobilenet_features():
    dfs = [None, None, None]

    model, transforms_model = load_default_image_model_or_preprocess()
    # get model specific transforms (normalization, resize)
    batch_size = 32
    num_workers = 4
    
    GPU = 5
    if torch.cuda.is_available():
        dev = "cuda:"+ str(GPU) 
    else: 
        dev = "cpu" 
    
    model = model.to(dev)
    
    for ix, split in enumerate(("train", "dev", "test")):
        embedded_vectors = []
        embeddings_ids = []
    
        data = [obj for obj in jsonlines.open(join(DATAPATH, split + ".jsonl"))]
        
        image_files = []
        text = []
        labels = []
        
        for row in tqdm(data):
            image_files.append(join(IMAGEPATH, row['image']))
            text.append(row['text'])
            # Applie Preprocess stage here
            labels.append(row['label']) 
        
        image_ids = [i for i in range(len(image_files))]
        
        img_dataset = ImageDataset(image_files, image_ids, transforms_model)
        img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
        for imgs, ids_batch in tqdm(img_dataloader, total=np.ceil(len(img_dataset) / batch_size).astype(int)):
            imgs = imgs.to(dev)
            
            with torch.no_grad():
                output = model(imgs)  
                output = output.squeeze().cpu().numpy()
                
            output = normalize_vector(output)
                
            embedded_vectors += output.tolist()
            embeddings_ids += [int(_id) for _id in ids_batch]
        
        image_files = [image_files[i] for i in embeddings_ids]
        embeddings = [embedded_vectors[i] for i in embeddings_ids]
        labels = [labels[i] for i in embeddings_ids]
    
        labels = [ (1,0) if l == 'not_informative' else (0,1) for l in labels ]
        labels = np.array(labels)
    
        df = pd.DataFrame({"image_files": image_files,
                            "text": text,
                            "embeddings": [f for f in embeddings],
                            "labels": np.argmax(labels, axis=1)})
    
        dfs[ix] = df
    
    return dfs


def maxvit_features():
    dfs = [None, None, None]
    
    #embeddings, embeddings_ids  = get_image_embedding(image_files, image_ids, model, m_transform, normalize=True, gpu_id=0, use_gpu=True)
    model = timm.create_model(
        #'maxvit_xlarge_tf_512.in21k_ft_in1k',
        'maxvit_tiny_tf_224.in1k',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms_model = timm.data.create_transform(**data_config, is_training=False)
    batch_size = 32
    num_workers = 4
    
    GPU = 5
    if torch.cuda.is_available():
        dev = "cuda:"+ str(GPU) 
    else: 
        dev = "cpu" 
    
    model = model.to(dev)
    
    for ix, split in enumerate(("train", "dev", "test")):
        embedded_vectors = []
        embeddings_ids = []
    
        data = [obj for obj in jsonlines.open(join(DATAPATH, split + ".jsonl"))]
        
        image_files = []
        text = []
        labels = []
        
        for row in tqdm(data):
            image_files.append(join(IMAGEPATH, row['image']))
            text.append(row['text'])
            # Applie Preprocess stage here
            labels.append(row['label']) 
        
        image_ids = [i for i in range(len(image_files))]
        
        img_dataset = ImageDataset(image_files, image_ids, transforms_model)
        img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
        for imgs, ids_batch in tqdm(img_dataloader, total=np.ceil(len(img_dataset) / batch_size).astype(int)):
            imgs = imgs.to(dev)
            
            with torch.no_grad():
                output = model(imgs)  
                output = output.squeeze().cpu().numpy()
                
            output = normalize_vector(output)
                
            embedded_vectors += output.tolist()
            embeddings_ids += [int(_id) for _id in ids_batch]
        
        image_files = [image_files[i] for i in embeddings_ids]
        embeddings = [embedded_vectors[i] for i in embeddings_ids]
        labels = [labels[i] for i in embeddings_ids]
    
        labels = [ (1,0) if l == 'not_informative' else (0,1) for l in labels ]
        labels = np.array(labels)
    
        df = pd.DataFrame({"image_files": image_files,
                            "text": text,
                            "embeddings": [f for f in embeddings],
                            "labels": np.argmax(labels, axis=1)})
    
        dfs[ix] = df

        return dfs
    