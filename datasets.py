from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from torchvision.datasets import ImageFolder
import json
import ast
from itertools import chain, repeat, islice
import torch
import numpy as np
import pickle
from pathlib import Path
import tarfile
import os
import io
import random

from PIL import PngImagePlugin

import clip  # pylint: disable=import-outside-toplevel
import webdataset as wds  # pylint: disable=import-outside-toplevel
    
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def get_classnames(datasetpath):
    if "imagenette_2class" in datasetpath:
        return [ 'church', 'garbage truck']
    else:
        return [ 'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
        

class ObjectAttributeDataset(ImageFolder):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        class_prompt=None,
        size=320,
        center_crop=False,
        random_flip = False,
        prompt_json = None,
        duplication = "nodup",
        args = None
    ):
        super().__init__(instance_data_root)
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer
        self.duplication = duplication
        self.objects = get_classnames(instance_data_root)
        self.trainspecial = args.trainspecial
        self.trainspecial_prob = args.trainspecial_prob
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if self.center_crop else transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.class_prompt = class_prompt
        if class_prompt in ['instancelevel_blip','instancelevel_ogcap','instancelevel_random']:
            
            assert prompt_json != None
            
            with open(prompt_json) as f:
                self.prompts = json.load(f)

        if self.duplication in ['dup_both','dup_image']:
            sw_path = f"{instance_data_root}/weights_{args.weight_pc}_{args.dup_weight}_seed{args.seed}.pickle"
            if Path(sw_path).exists():
                with open(sw_path, 'rb') as handle:
                    samplingweights = pickle.load(handle)
            else:
                samplingweights = [1]*len(self.samples)
                ow_samples = np.random.choice(len(self.samples), int(args.weight_pc*len(self.samples)),replace=False)
                for i in ow_samples:
                    samplingweights[i] = samplingweights[i]*args.dup_weight//1
                with open(sw_path, 'wb') as handle:
                    pickle.dump(samplingweights, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.samplingweights = samplingweights
            print(len(self.samplingweights))    
            
    def __getitem__(self, index):
        instance_image,label = super().__getitem__(index)
        path_img,_ = self.samples[index]
        example = {}
        # instance_image = Image.open(img)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        if self.trainspecial is not None:
            if self.trainspecial in ['allcaps']:
                instance_prompt = np.random.choice(self.prompts[path_img], 1)[0]
            elif self.trainspecial in ['randrepl']:
                instance_prompt = self.prompts[path_img][0]
                rand = np.random.uniform()
                if rand <= self.trainspecial_prob :
                    instance_prompt = list(np.random.randint(49400, size=4))
                    instance_prompt = self.tokenizer.decode(instance_prompt)
            elif self.trainspecial in ['randwordadd']: # 2 random words get added
                instance_prompt = self.prompts[path_img][0]
                rand = np.random.uniform()
                if rand <= self.trainspecial_prob:
                    randword = self.tokenizer.decode(list(np.random.randint(49400, size=1)))
                    instance_prompt = insert_rand_word(instance_prompt,randword) 
                    randword = self.tokenizer.decode(list(np.random.randint(49400, size=1)))
                    instance_prompt = insert_rand_word(instance_prompt,randword) 
            elif self.trainspecial in ['wordrepeat']:
                instance_prompt = self.prompts[path_img][0]
                wordlist = instance_prompt.split(" ")
                rand = np.random.uniform()
                if rand <= self.trainspecial_prob:
                    randword = np.random.choice(wordlist)
                    instance_prompt = insert_rand_word(instance_prompt,randword) 
                    randword = np.random.choice(wordlist)
                    instance_prompt = insert_rand_word(instance_prompt,randword) 

        else:
            if self.class_prompt == 'nolevel':
                instance_prompt = "An image"
            elif self.class_prompt == 'classlevel':
                instance_prompt = f"An image of {self.objects[label]}"
            elif self.class_prompt in ['instancelevel_blip','instancelevel_random','instancelevel_ogcap']:
                if self.duplication in ['nodup','dup_both']:
                    instance_prompt = self.prompts[path_img][0]
                elif self.duplication == 'dup_image':
                    if self.samplingweights[index] > 1:
                        instance_prompt = np.random.choice(self.prompts[path_img], 1)[0]
                    else:
                        instance_prompt = self.prompts[path_img][0]
            if self.class_prompt in ['instancelevel_random']:
                instance_prompt = ast.literal_eval(instance_prompt)
                instance_prompt = self.tokenizer.decode(instance_prompt)
        # print(path_img, instance_prompt)
        example["instance_prompt_ids"] = self.tokenizer(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

def insert_rand_word(sentence,word):
    import random
    sent_list = sentence.split(' ')
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = ' '.join(sent_list)
    return new_sent
    

class MyWebDataset(wds.WebDataset):
    def __init__(self, *args, duplication=None, weight_pc=0.05, dup_weight=5, seed=42, **kwargs):
        self.duplication = duplication
        self.weight_pc = weight_pc
        #eg:weight_pc=0.05 means there are 5% sample being dup
        self.dup_weight = dup_weight
        #eg:dup_weight=5 means each dup sample repeat 5 times
        self.seed = seed
        super(MyWebDataset, self).__init__(*args, **kwargs)

    def iterator(self):
        """Create an iterator through the entire dataset, using the given number of repetitions."""
        
        for i in range(self.repetitions):
            random.seed(self.seed)
            for sample in self.iterator1():
                if self.duplication == "dup_both":
                    if random.random() < self.weight_pc:
                        for _ in range(self.dup_weight):
                            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
                            yield sample
                    else:
                        yield sample
                else:
                    yield sample

    def __iter__(self):
        """Create an iterator through the pipeline, repeating and slicing as requested."""
        if self.repetitions != 1:
            if self.nsamples > 0:
                return islice(self.iterator(), self.nsamples)
            else:
                return self.iterator()
        else:
            return self.iterator()



def create_webdataset(
    urls,
    tokenizer,
    size=256,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    center_crop=False,
    random_flip = False,
    cache_path=None,
    duplication=None, 
    weight_pc=0.05, 
    dup_weight=5, 
    seed=12345,
    use_clean_prompts=False,
    use_multiple_bilp_caption=False,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    

    #dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue)
    dataset = MyWebDataset(urls, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue, duplication=duplication, weight_pc=weight_pc, dup_weight=dup_weight, seed=seed)
    tokenizer = tokenizer

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)
    
    image_transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    if not use_multiple_bilp_caption:
        if use_clean_prompts:
            remove = urls[0].split("/")[-1]
            rootdir = urls[0].replace(remove, "/")
            #with open(os.path.join(rootdir, "laion_10k_data_2_GPT_clean_captions.json"),'r') as f:
            with open(os.path.join(rootdir, "laion_10k_data_2_GPT_5_words_clean_captions.json"),'r') as f:
                json_str = f.read()
                clean_prompts = json.loads(json_str)
                print(len(clean_prompts))
        else:
            remove = urls[0].split("/")[-1]
            rootdir = urls[0].replace(remove, "/")
            with open(os.path.join(rootdir, "all_captions_laion_10k.json"),'r') as f:
                json_str = f.read()
                clean_prompts = json.loads(json_str)
                print(len(clean_prompts))
            
    else:
        remove = urls[0].split("/")[-1]
        rootdir = urls[0].replace(remove, "/")
        with open(os.path.join(rootdir, "all_captions_blip.json"),'r') as f:
            json_str = f.read()
            multiple_prompts = json.loads(json_str)
            print(len(multiple_prompts))
        

    def preprocess_dataset(item):
        output = {}
        if enable_image:
            image_data = item[image_key]
            image = Image.open(io.BytesIO(image_data))
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image_tensor = image_transform(image)
            #output["image_filename"] = item["__key__"]
            #output["image_tensor"] = image_tensor
            output["instance_images"] = image_tensor

        if enable_text:
            if not use_multiple_bilp_caption:
                if not use_clean_prompts:
                    text = item[caption_key]
                    caption = text.decode("utf-8")
                    #print(caption)
                else:
                    metadata_file = item["json"]
                    metadata = metadata_file.decode("utf-8")
                    json_meta = json.loads(metadata)
                    new_prompt_key = json_meta["key"] + ".jpg"
                    caption = clean_prompts[new_prompt_key][0]
                #tokenized_text = tokenizer(caption)
                #output["text_tokens"] = tokenized_text
                #output["text"] = caption
                if caption[-1] == '.':
                    caption = caption[:-1]
                output["instance_prompt_ids"] = tokenizer(
                    caption,
                    truncation=True,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids
            
            else:
                metadata_file = item["json"]
                metadata = metadata_file.decode("utf-8")
                json_meta = json.loads(metadata)
                new_prompt_key = json_meta["key"] + ".jpg"
                caption = multiple_prompts[new_prompt_key]
                output["instance_prompt_ids"] = []
                for i in range(len(caption)):
                    output["instance_prompt_ids"].append(tokenizer(
                        caption[i],
                        truncation=True,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids)
            
        if enable_metadata:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            output["metadata"] = metadata
        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    #print(transformed_dataset)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format, use_multiple_bilp_caption=False):
    """Create a pytorch dataloader from a dataset"""

    #def collate_fn(batch):
        #batch = list(filter(lambda x: x is not None, batch))
        #return default_collate(batch)
    def collate_fn(examples):
        '''
        if not multiple nilp caption, input_ids is a tensor containing a batch of captions tokens
        else, input_ids is a list, each element is also a list containing 20 tensors of captions for one sample
        '''
        pixel_values = [example["instance_images"] for example in examples]


        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        
        if not use_multiple_bilp_caption:
            input_ids = [example["instance_prompt_ids"] for example in examples]
            input_ids = torch.cat(input_ids, dim=0)
        else:
            input_ids = [torch.cat(example["instance_prompt_ids"], dim=0).unsqueeze(dim=0) for example in examples]
            input_ids = torch.cat(input_ids, dim=0)
            #print(input_ids.size())

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        #collate_fn=collate_fn if input_format == "files" else None,
        collate_fn=collate_fn,
    )
    return data

def Webdatasetloader(
        input_dataset,
        tokenizer,
        size,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        center_crop=False,
        random_flip = False,
        cache_path=None,
        duplication=None, 
        weight_pc=0.05, 
        dup_weight=5, 
        seed=12345,
        use_clean_prompts=False,
        use_multiple_bilp_caption=False,
    ):
        
    dataset = create_webdataset(
        input_dataset,
        tokenizer,
        size,
        enable_text=enable_text,
        enable_image=enable_image,
        image_key=wds_image_key,
        caption_key=wds_caption_key,
        enable_metadata=enable_metadata,
        center_crop=center_crop,
        random_flip = random_flip,
        cache_path=cache_path,
        duplication=duplication, 
        weight_pc=weight_pc, 
        dup_weight=dup_weight, 
        seed=seed,
        use_clean_prompts=use_clean_prompts,
        use_multiple_bilp_caption=use_multiple_bilp_caption,
    )
    dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "webdataset", use_multiple_bilp_caption=use_multiple_bilp_caption)
    return dataloader



'''
class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        input_dataset,
        tokenizer,
        size,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        center_crop=False,
        random_flip = False,
        cache_path=None,
    ):
        self.batch_size = batch_size
        
        dataset = create_webdataset(
            input_dataset,
            tokenizer,
            size,
            enable_text=enable_text,
            enable_image=enable_image,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            center_crop=center_crop,
            random_flip = random_flip,
            cache_path=cache_path,
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "webdataset")
        print(self.dataloader)
        print("!!!!!!!!!!!!!!!!!!")

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
'''