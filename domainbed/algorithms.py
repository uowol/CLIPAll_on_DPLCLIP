# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

# import domainbed.captionizer as captionizer
from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches

import clip
import pickle

ALGORITHMS = [
    # some algorithms moved to 'other_algorithms.py'
    'CLIP',
    'CLIPALL',
    'DPLCLIP',
    'DPLCLIPALL',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class CLIP(Algorithm):
    def get_prompts(self, path=None):   # path = "---.jpg"
        f = open(str(path)[:-3]+'txt')
        captions = [caption.replace('\n','').strip() for caption in f.readlines() if caption[:7] == "It is a"]
        prompts = captions
        return prompts

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model = clip.load(self.hparams['clip_backbone'])[0].float()

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print('Set self.clip_model.parameters.reguires_grad = False!')

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512  # 
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        
    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}
    
    def predict(self, x, paths):
        if not self.hparams['use_caption']:
            logits_per_image, _ = self.clip_model(x, self.prompt)
            return logits_per_image.softmax(dim=-1)
        else:
            logits_per_image_ = []
            for i, path in enumerate(paths):
                prompts = self.get_prompts(path)    # [7]
                tokenized_prompts = torch.cat(      # self.tokenized_prompts, [7, 77]
                    [clip.tokenize(p, truncate=True) for p in prompts]
                ).to(self.device)
                logits_per_image, _ = self.clip_model(x[i].unsqueeze(0), tokenized_prompts)
                logits_per_image_.append(logits_per_image)
            logits_per_image = torch.cat(logits_per_image_)
            return logits_per_image.softmax(dim=-1)
            

class CLIPALL(CLIP):
    def encode_image(self, image):
        num_image_layer = self.clip_model.visual.transformer.layers
        image = image.to(self.device)

        out_list = []
        x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)                      # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + 
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)          # NLD -> LND

        for i in range(num_image_layer):
            x = self.clip_model.visual.transformer.resblocks[i](x)
            tmp = x.permute(1, 0, 2)    # LND -> NLD
            tmp = tmp[:, 0, :].detach()
            out_list.append(tmp)

        image_features = torch.stack(out_list)

        return image_features
    
    def encode_text(self, text):
        num_text_layer = self.clip_model.transformer.layers
        text = text.to(self.device)

        out_list = []
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_clip_model]
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)                  # NLD -> LND

        for i in range(num_text_layer):
            x = self.clip_model.transformer.resblocks[i](x)
            tmp = x.permute(1, 0, 2).detach()   # LND -> NLD
            out_list.append(tmp)

        text_features = torch.stack(out_list)

        return text_features
    
    def __init__(self, input_shape, num_classes, num_domains, hparams): 
        super(CLIPALL, self).__init__(input_shape, num_classes, num_domains, hparams)
        visual_width = 768       
        textual_width = 512      
        visual_scale = visual_width ** -0.5
        textual_scale = textual_width ** -0.5
        output_dim = self.EMBEDDING_DIM

        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.num_of_visual_encoder_layers = self.clip_model.visual.transformer.layers
        self.num_of_textual_encoder_layers = self.clip_model.transformer.layers
        self.ln_post = self.clip_model.visual.ln_post#.requires_grad_(True)
        self.ln_final = self.clip_model.ln_final#.requires_grad_(True)
        self.visual_projection = nn.Parameter(  # [12, 768, 512]
            torch.stack([
                visual_scale * torch.randn((visual_width, output_dim), dtype=self.dtype).to(self.device)
                for _ in range(self.num_of_visual_encoder_layers - 1)
            ]+[self.clip_model.visual.proj])).requires_grad_(True)
        self.textual_projection = nn.Parameter( # [12, 512, 512]
            torch.stack([
                textual_scale * torch.randn((textual_width, output_dim), dtype=self.dtype).to(self.device)
                for _ in range(self.num_of_textual_encoder_layers - 1)
            ]+[self.clip_model.text_projection])).requires_grad_(True)
        self.score_type = hparams['score_type']
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        print("LOG:",classnames)    # ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)

        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )

    def update(self, minibatches, unlabeled=None):
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        if not self.hparams['use_caption']:
            # encode image for each domain.
            image_features_ = []
            for x in all_x:
                image_features = self.encode_image(x)
                image_features_.append(image_features)
            image_features = torch.cat(image_features_, dim=1)              # [12, 96, 768]

            # encode text for each domain.
            text_features_ = []
            text_features = self.encode_text(self.prompt)                   # [12, 7, 77, 512]

            image_features = self.ln_post(image_features)                   # [12, 96, 768]
            image_features = image_features @ self.visual_projection        # [12, 96, self.EMBEDDING_DIM]

            text_features = self.ln_final(text_features)                    # [12, 7, 77, 512]
            text_features = torch.einsum('abc,acd->abd',
                                text_features[:,
                                    torch.arange(text_features.shape[1]),   # [7]
                                    self.prompt.argmax(-1)                  # [7, 77] -> [7]
                                ],                                          # [12, 7, self.EMBEDDING_DIM]
                                self.textual_projection)                    # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (12, 96, self.EMBEDDING_DIM)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # (12,  7, self.EMBEDDING_DIM)

            score = 0
            score_tensor = torch.einsum("abc,dec->bead",image_features, text_features)  # (96,  7, 12, 12)
            score_tensor = score_tensor.reshape(*score_tensor.shape[:2],-1)             # (96,  7, 144)

            if self.score_type == 'max':
                score = torch.max(score_tensor, dim=-1)[0]
            elif self.score_type == 'mean':
                score = torch.mean(score_tensor, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * score

            loss = F.cross_entropy(logits, all_y)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item()}

        elif self.hparams['use_caption']:
            all_path = sum([list(data[2]) for data in minibatches], [])
            # all_label = torch.cat([data[3] for data in minibatches])

            # encode image for each domain.
            image_features_ = []
            for x in all_x:
                image_features = self.encode_image(x)
                image_features_.append(image_features)
            image_features = torch.cat(image_features_, dim=1)               # [12, 96, 768]
            
            # encode text for each domain.
            text_features_ = []
            captions_ = []
            for path in all_path:
                try:
                    with open(str(path)[:-3]+'pickle', 'rb') as fr:
                        captions, text_features = pickle.load(fr) 
                    captions_.append(captions)
                    text_features_.append(text_features)
                except:
                    f = open(str(path)[:-3]+'txt')
                    captions = torch.cat(
                        [
                            clip.tokenize(caption.replace('\n','').strip(), truncate=True).to(self.device) 
                            for caption in f.readlines() if caption[:7] == "It is a"
                        ]
                    )
                    captions_.append(captions.unsqueeze(0))
                    text_features = self.encode_text(captions)
                    text_features_.append(text_features.unsqueeze(1))
                    with open(str(path)[:-3]+'pickle', 'wb') as fw:
                        pickle.dump((captions_[-1], text_features_[-1]), fw)
            text_features = torch.cat(text_features_, dim=1)                # [12, 96, 7, 77, self.EMBEDDING_DIM]
            captions = torch.cat(captions_)                                 # [96, 7, 77]
            num_of_class = captions.shape[1]                                # 7

            image_features = self.ln_post(image_features)                   # [12, 96, self.EMBEDDING_DIM]
            image_features = image_features @ self.visual_projection

            text_features = self.ln_final(text_features)                    # [12, 96, 7, 77, self.EMBEDDING_DIM]
            text_features = text_features.view(12,-1,77,self.EMBEDDING_DIM) # [12, 96*7, 77, self.EMBEDDING_DIM]
            text_features = torch.einsum('abcd,adz->abcz',
                                text_features[:,
                                    torch.arange(text_features.shape[1]),   # [96*7]
                                    captions.view(-1,77).argmax(-1)         # [96*7, 77] -> [96*7]
                                ].view(self.num_of_textual_encoder_layers, -1, num_of_class, self.EMBEDDING_DIM),  
                                                                            # [12, 96, 7, self.EMBEDDING_DIM]
                                self.textual_projection)                    # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]

            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (12, 96, self.EMBEDDING_DIM)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # (12, 96, 7, self.EMBEDDING_DIM)

            score = 0
            score_tensor = torch.einsum("abc,xbzc->bzax",image_features, text_features) 
            score_tensor = score_tensor.reshape(*score_tensor.shape[:2],-1)

            if self.score_type == 'max':
                score = torch.max(score_tensor, dim=-1)[0]   
            elif self.score_type == 'mean':
                score = torch.mean(score_tensor, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * score

            loss = F.cross_entropy(logits, all_y)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item()}

    def predict(self, x, paths):
        if not self.hparams['use_caption']:
            # encode image for each domain.
            image_features = self.encode_image(x)

            # encode text for each domain.
            text_features_ = []
            text_features = self.encode_text(self.prompt)                   # [12, 7, 77, 512]

            image_features = self.ln_post(image_features)                   # [12, 96, 768]
            image_features = image_features @ self.visual_projection        # [12, 96, self.EMBEDDING_DIM]

            text_features = self.ln_final(text_features)                    # [12, 7, 77, 512]
            text_features = torch.einsum('abc,acd->abd',
                                text_features[:,
                                    torch.arange(text_features.shape[1]),   # [7]
                                    self.prompt.argmax(-1)                  # [7, 77] -> [7]
                                ],                                          # [12, 7, 512]
                                self.textual_projection)                    # [12, 512, self.EMBEDDING_DIM]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (12, 96, self.EMBEDDING_DIM)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # (12,  7, self.EMBEDDING_DIM)

            score = 0
            score_tensor = torch.einsum("abc,dec->bead",image_features, text_features)  # (96,  7, 12, 12)
            score_tensor = score_tensor.reshape(*score_tensor.shape[:2],-1)             # (96,  7, 144)

        elif self.hparams['use_caption']:
            # encode image for each domain.
            image_features = self.encode_image(x)
            
            # encode text for each domain.
            text_features_ = []
            captions_ = []
            for path in paths:
                try:
                    with open(str(path)[:-3]+'pickle', 'rb') as fr:
                        captions, text_features = pickle.load(fr) 
                    captions_.append(captions)
                    # text_features = torch.cat([text_features_prior, text_features])
                    text_features_.append(text_features)
                except:
                    f = open(str(path)[:-3]+'txt')
                    captions = torch.cat(                                   # [7, 77]
                        [clip.tokenize(caption.replace('\n','').strip(), truncate=True).to(self.device)]
                    )
                    captions_.append(captions.unsqueeze(0))
                    text_features = self.encode_text(captions)
                    text_features_.append(text_features.unsqueeze(1))
                    with open(str(path)[:-3]+'pickle', 'wb') as fw:
                        pickle.dump((captions_[-1], text_features_[-1]), fw)
            text_features = torch.cat(text_features_, dim=1)                # [12, 96, 7, 77, self.EMBEDDING_DIM]
            captions = torch.cat(captions_)                                 # [96, 7, 77]
            num_of_class = captions.shape[1]                                # 7

            image_features = self.ln_post(image_features)                   # [12, 96, 768]
            image_features = image_features @ self.visual_projection        # [12, 96, 512]

            text_features = self.ln_final(text_features)                    # [12, 96, 7, 77, 512]
            text_features = text_features.view(12,-1,77,self.EMBEDDING_DIM) # [12, 96*7, 77, 512]
            text_features = torch.einsum('abcd,adz->abcz',
                                text_features[:,
                                    torch.arange(text_features.shape[1]),   # [96*7]
                                    captions.view(-1,77).argmax(-1)         # [96*7, 77] -> [96*7]
                                ].view(self.num_of_textual_encoder_layers, -1, num_of_class, self.EMBEDDING_DIM),  
                                                                            # [12, 96, 7, 512]
                                self.textual_projection)                    # [12, 512, self.EMBEDDING_DIM]

            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (12, 96, self.EMBEDDING_DIM)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # (12, 96, 7, self.EMBEDDING_DIM)

            score = 0
            score_tensor = torch.einsum("abc,xbzc->bzax",image_features, text_features) 
            score_tensor = score_tensor.reshape(*score_tensor.shape[:2],-1)

        if self.score_type == 'max':
            score = torch.max(score_tensor, dim=-1)[0]   
        elif self.score_type == 'mean':
            score = torch.mean(score_tensor, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score
        return logits
     
class DPLCLIP(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DPLCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)

        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        
        print('Using sentence_prompt in DPLCLIP...')
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in hparams['class_names']]
        prompts = [prompt_prefix + ' ' + prompt + '.' for prompt in prompts]
        # prompts:  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]
        
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        self.network = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.network.apply(init_weights)
        
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        
        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)
            
    def get_prompts(self, path=None):   # path = "---.jpg"
        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * self.hparams['num_domain_tokens'])
        f = open(str(path)[:-3]+'txt')
        captions = [caption.replace('\n','').strip() for caption in f.readlines() if caption[:7] == "It is a"]
        prompts = [prompt_prefix + ' ' + caption for caption in captions]
        return prompts

    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        if not self.hparams['use_caption']:
            #  encode image for each domain.
            image_features = [self.clip_model.encode_image(x) for x in all_x]
            
            #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
            domain_features = [self.network(feature) for feature in image_features]
            image_features = torch.cat(image_features)
            #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
            mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]

            #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
            _mean_domain_features = [feature.repeat_interleave(len(self.hparams['class_names']), dim=0) for feature in mean_domain_features]
            
            #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
            # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
            text_features = torch.cat([self._get_text_features(feature) for feature in _mean_domain_features])
                
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
            loss = F.cross_entropy(logits_per_image, all_y)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item()}
        
        elif self.hparams['use_caption']:
            all_path = sum([list(data[2]) for data in minibatches], [])

            text_features_ = []
            tokenized_prompts_ = []
            token_prefix = []
            token_suffix = []
            for path in all_path:
                prompts = self.get_prompts(path)    # [7]
                tokenized_prompts = torch.cat(      # [7, 77]
                    [clip.tokenize(p, truncate=True) for p in prompts]
                ).to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)
                                                    # [7, 77, 512]
                
                token_prefix.append(embedding[:, :1, :].unsqueeze(0))       # SOS
                token_suffix.append(embedding[:, self.hparams['num_domain_tokens'] + 1:, :].unsqueeze(0))
                tokenized_prompts_.append(tokenized_prompts.unsqueeze(0))
            tokenized_prompts = torch.cat(tokenized_prompts_)               # [96, 7, 77]
            token_prefix = torch.cat(token_prefix)                          # [96, 7,  1, 512]
            token_suffix = torch.cat(token_suffix)                          # [96, 7, 60, 512]

            #  encode image for each domain.
            image_features = [self.clip_model.encode_image(x) for x in all_x]
            domain_features = [self.network(feature) for feature in image_features]
            # print("LOG:", len(domain_features), domain_features[0].shape) # 3, [32, 8192]   
            image_features = torch.cat(image_features)
            mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]
            # print("LOG:", mean_domain_features[0].shape)                  # [1, 8192]
            
            _mean_domain_features = [feature.repeat_interleave(len(self.hparams['class_names']), dim=0) for feature in mean_domain_features]
            # print("LOG:", _mean_domain_features[0].shape)                 # [7, 8192]
            
            text_features = [self._get_text_features(
                                    feature,                                # [7, 16*512]
                                    tokenized_prompts[(i*32):((i+1)*32)],   # [32, 7, 77]
                                    token_prefix[(i*32):((i+1)*32)],        # [32, 7,  1, 512]
                                    token_suffix[(i*32):((i+1)*32)]         # [32, 7, 60, 512]
                                ) for (i, feature) in enumerate(_mean_domain_features)]
            # print("LOG:", len(text_features), text_features[0].shape)     # 3, [32, 7, 512]
            text_features = torch.cat(text_features)
            # print("LOG:", image_features.shape, text_features.shape)      # [96, 512], [96, 7, 512]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = self.clip_model.logit_scale.exp() * torch.einsum("ab,acb->ac", image_features, text_features)
            loss = F.cross_entropy(logits_per_image, all_y)
            # print("LOG:",logits_per_image.shape, all_y.shape, logits_per_image.argmax(-1), all_y, loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item()}


    def _get_text_features(self, domain_feature, 
                           tokenized_prompts=None, token_prefix=None, token_suffix=None,
                           coop=False):
        if not self.hparams['use_caption']:
            #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
            domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
            
            #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
            domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
            
            #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
            x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = x.permute(1, 0, 2)
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
            
            #  mapping domain_features to text_features.
            text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection      
            return text_features

        elif self.hparams['use_caption']:
            N = tokenized_prompts.shape[0]  # 32

            #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
            domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
            
            #  repeat domain_feature: [7, 16, self.EMBEDDING_DIM] -> [32, 7, 16, self.EMBEDDING_DIM]
            domain_feature = domain_feature.unsqueeze(0).repeat_interleave(N, dim=0)
            # print("LOG:",domain_feature.shape)

            #  reshape domain_feature: [32, 7, 16, self.EMBEDDING_DIM] -> [32, 7, 77, self.EMBEDDING_DIM]
            domain_feature = torch.cat([token_prefix, domain_feature, token_suffix], dim=2)
            # print("LOG:",domain_feature.shape)  # [32, 7, 77, 512]

            #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
            text_features = torch.cat([self._encode_text(feature).unsqueeze(0) for feature in domain_feature])
            # print("LOG:",text_features.shape)   # [32, 7, 77, 512]
            
            text_features = text_features.view(-1,77,self.EMBEDDING_DIM)    # [32*7, 77, self.EMBEDDING_DIM]
            # print("LOG:",text_features.shape, tokenized_prompts.view(-1,77).shape)
            text_features = torch.einsum('bcd,dz->bcz',                     # [32, 7, self.EMBEDDING_DIM]
                                text_features[
                                    torch.arange(text_features.shape[0]),   # [32*7]
                                    tokenized_prompts.view(-1,77).argmax(-1)# [32*7, 77] -> [32*7]
                                ].view(N, -1, self.EMBEDDING_DIM),          # [32, 7, self.EMBEDDING_DIM]
                                self.clip_model.text_projection)            # [self.EMBEDDING_DIM, self.EMBEDDING_DIM]
            # print("LOG:",text_features.shape)   # [32, 7, 512]

            return text_features

    def _encode_text(self, domain_feature): # [7, 77, 512]
        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype) # [7, 77, 512]
        return x

    def predict(self, x, paths):
        if not self.hparams['use_caption']:
            image_feature = self.clip_model.encode_image(x)
            
            domain_feature = self.network(image_feature)
            mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.hparams['class_names']), dim=0)
            text_feature = self._get_text_features(mean_domain_feature)
            
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            return self.clip_model.logit_scale.exp() * image_feature @ text_feature.t()
        elif self.hparams['use_caption']:
            N = x.shape[0]; M = N/3
            text_features_ = []
            tokenized_prompts_ = []
            token_prefix = []
            token_suffix = []
            for path in paths:
                prompts = self.get_prompts(path)    # [7]
                tokenized_prompts = torch.cat(      # self.tokenized_prompts, [7, 77]
                    [clip.tokenize(p, truncate=True) for p in prompts]
                ).to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)
                                                    # [7, 77, 512]
                
                token_prefix.append(embedding[:, :1, :].unsqueeze(0))  # SOS
                token_suffix.append(embedding[:, self.hparams['num_domain_tokens'] + 1:, :].unsqueeze(0))  # CLS, EOS
                tokenized_prompts_.append(tokenized_prompts.unsqueeze(0))
            tokenized_prompts = torch.cat(tokenized_prompts_)               # [N, 7, 77]
            token_prefix = torch.cat(token_prefix)                          # [N, 7,  1, 512]
            token_suffix = torch.cat(token_suffix)                          # [N, 7, 60, 512]

            image_feature = self.clip_model.encode_image(x)
            
            domain_feature = self.network(image_feature)
            mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.hparams['class_names']), dim=0)
            # print("LOG:", mean_domain_feature.shape)    # [7, 8192]
            # text_feature = self._get_text_features(mean_domain_feature)
            text_feature = self._get_text_features(
                                    mean_domain_feature,    # [7, 8192]
                                    tokenized_prompts,      # [N, 7, 77]
                                    token_prefix,           # [N, ...]
                                    token_suffix            # [N, ...]
                                )
            # print("LOG:", image_feature.shape, text_feature.shape)
            
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            return self.clip_model.logit_scale.exp() * torch.einsum("ab,acb->ac", image_feature, text_feature)

class DPLCLIPALL(CLIP):    
    def get_prompts(self, path=None):   # path = "---.jpg"
        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * self.hparams['num_domain_tokens'])
        f = open(str(path)[:-3]+'txt')
        captions = [caption.replace('\n','').strip() for caption in f.readlines() if caption[:7] == "It is a"]
        prompts = [prompt_prefix + ' ' + caption for caption in captions]
        return prompts

    def __init__(self, input_shape, num_classes, num_domains, hparams): 
        super(DPLCLIPALL, self).__init__(input_shape, num_classes, num_domains, hparams)

        # NOTE: CLIPALL
        visual_width = 768       
        textual_width = 512      
        visual_scale = visual_width ** -0.5
        textual_scale = textual_width ** -0.5
        output_dim = self.EMBEDDING_DIM

        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.num_of_visual_encoder_layers = self.clip_model.visual.transformer.layers
        self.num_of_textual_encoder_layers = self.clip_model.transformer.layers
        self.ln_post = self.clip_model.visual.ln_post#.requires_grad_(True)
        self.ln_final = self.clip_model.ln_final#.requires_grad_(True)
        self.visual_projection = nn.Parameter(  # [12, 768, 512]
            torch.stack([
                visual_scale * torch.randn((visual_width, output_dim), dtype=self.dtype).to(self.device)
                for _ in range(self.num_of_visual_encoder_layers - 1)
            ]+[self.clip_model.visual.proj])).requires_grad_(True)
        self.textual_projection = nn.Parameter( # [12, 512, 512]
            torch.stack([
                textual_scale * torch.randn((textual_width, output_dim), dtype=self.dtype).to(self.device)
                for _ in range(self.num_of_textual_encoder_layers - 1)
            ]+[self.clip_model.text_projection])).requires_grad_(True)
        self.score_type = hparams['score_type']
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        print("LOG:",classnames)    # ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

        # NOTE: DPLCLIP
        prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in hparams['class_names']]
        prompts = [prompt_prefix + ' ' + prompt + '.' for prompt in prompts]
        # prompts:  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]

        # to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        self.network = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.network.apply(init_weights)

        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )

    def encode_image(self, image):
        if not self.hparams['use_caption']:
            num_image_layer = self.clip_model.visual.transformer.layers
            image = image.to(self.device)

            out_list = []
            x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)                      # shape = [*, grid ** 2, width]
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + 
                        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)          # NLD -> LND

            for i in range(num_image_layer):
                x = self.clip_model.visual.transformer.resblocks[i](x)
                tmp = x.permute(1, 0, 2)    # LND -> NLD
                tmp = tmp[:, 0, :].detach()
                out_list.append(tmp)

            image_features = torch.stack(out_list)                          # [12, 96, 768]
            # image_features = self.ln_post(image_features)                   # [12, 96, 768]
            # image_features = image_features @ self.visual_projection        # [12, 96, self.EMBEDDING_DIM]

            return image_features
        else:
            # TODO
            pass
    
    def encode_text(self, domain_feature):
        if not self.hparams['use_caption']:
            num_text_layer = self.clip_model.transformer.layers

            #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
            domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
            
            #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
            domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
            
            out_list = []
            x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = x.permute(1, 0, 2)                  # NLD -> LND

            for i in range(num_text_layer):
                x = self.clip_model.transformer.resblocks[i](x)
                tmp = x.permute(1, 0, 2).detach()   # LND -> NLD
                out_list.append(tmp)

            text_features = torch.stack(out_list)
            text_features = self.ln_final(text_features)                    # [12, 7, 77, 512]
            # print("LOG:", text_features.shape, self.tokenized_prompts.shape); raise ValueError

            text_features = torch.einsum('abc,acd->abd',
                                text_features[:,
                                    torch.arange(text_features.shape[1]),   # [ 7]
                                    self.tokenized_prompts.argmax(-1)       # [ 7, 77] -> [ 7]
                                ],                                          # [12,  7, self.EMBEDDING_DIM]
                                self.textual_projection)                    # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]

            return text_features
        else:
            # TODO
            pass
    

    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        
        # self.EMBEDDING_DIM = 512      num_domain_tokens = 16
        if not self.hparams['use_caption']:
            # NOTE: DPLCLIP
            # image_features :          3 * [32, self.EMBEDDING_DIM]
            image_features = [self.clip_model.encode_image(x) for x in all_x]
            # domain_features :         3 * [32, self.EMBEDDING_DIM * num_domain_tokens]
            domain_features = [self.network(feature) for feature in image_features]
            # image_features :              [96, 768]
            image_features = torch.cat(image_features)

            # mean_domain_features :    3 * [ 1, self.EMBEDDING_DIM * num_domain_tokens]
            mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]
            # _mean_domain_features :   3 * [ 7, self.EMBEDDING_DIM * num_domain_tokens]
            _mean_domain_features = [feature.repeat_interleave(len(self.hparams['class_names']), dim=0) for feature in mean_domain_features]
            
            # encode text for each domain.
            text_features = torch.cat([self.encode_text(feature) for feature in _mean_domain_features],
                                      dim=1)                                # [12, 21, 512]
            # print("LOG:", text_features.shape); raise ValueError

            # NOTE: CLIPALL
            image_features_ = []
            for x in all_x:
                image_features = self.encode_image(x)
                image_features_.append(image_features)
            image_features = torch.cat(image_features_, dim=1)              # [12, 96, 768]
            image_features = self.ln_post(image_features)                   # [12, 96, 768]
            image_features = image_features @ self.visual_projection        # [12, 96, 512]

            # text_features = self.ln_final(text_features)                    # [12, 21, 77, 512]
            # print("LOG:", text_features.shape, self.tokenized_prompts.shape); raise ValueError

            # text_features = torch.einsum('abc,acd->abd',
            #                     text_features[:,
            #                         torch.arange(text_features.shape[1]),   # [21]
            #                         self.tokenized_prompts.argmax(-1)       # [21, 77] -> [21]
            #                     ],                                          # [12, 21, self.EMBEDDING_DIM]
            #                     self.textual_projection)                    # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]


            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (12, 96, self.EMBEDDING_DIM)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # (12, 21, self.EMBEDDING_DIM)

            score = 0
            score_tensor = torch.einsum("abc,dec->bead",image_features, text_features)  # (96, 21, 12, 12)
            score_tensor = score_tensor.reshape(*score_tensor.shape[:2],-1)             # (96, 21, 144)

            if self.score_type == 'max':
                score = torch.max(score_tensor, dim=-1)[0]
            elif self.score_type == 'mean':
                score = torch.mean(score_tensor, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * score

            loss = F.cross_entropy(logits, all_y)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item()}

        elif self.hparams['use_caption']:
            # TODO
            pass
            
    def predict(self, x, paths):
        if not self.hparams['use_caption']:
            # NOTE: DPLCLIP
            image_feature = self.clip_model.encode_image(x)
            domain_feature = self.network(image_feature)
            mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.hparams['class_names']), dim=0)
            text_feature = self.encode_text(mean_domain_feature)            # [12,  7, 512]

            # NOTE: CLIPALL
            image_feature = self.encode_image(x)                            # [12, 64, 768]
            image_feature = self.ln_post(image_feature)                     # [12, 64, 768]
            image_feature = image_feature @ self.visual_projection          # [12, 64, 512]

            # text_feature = self.ln_final(text_feature)                      # [12, 7, 77, 512]
            # text_feature = torch.einsum('abc,acd->abd',
            #                     text_feature[:,
            #                         torch.arange(text_feature.shape[1]),    # [7]
            #                         self.tokenized_prompts.argmax(-1)       # [7, 77] -> [7]
            #                     ],                                          # [12, 7, 512]
            #                     self.textual_projection)                    # [12, 512, self.EMBEDDING_DIM]

            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True) # (12, 96, self.EMBEDDING_DIM)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)    # (12,  7, self.EMBEDDING_DIM)

            score = 0
            score_tensor = torch.einsum("abc,dec->bead",image_feature, text_feature) # (96,  7, 12, 12)
            score_tensor = score_tensor.reshape(*score_tensor.shape[:2],-1)          # (96,  7, 144)

        elif self.hparams['use_caption']:
            # encode image for each domain.
            image_features = self.encode_image(x)
            
            # encode text for each domain.
            text_features_ = []
            captions_ = []
            for path in paths:
                try:
                    with open(str(path)[:-3]+'pickle', 'rb') as fr:
                        captions, text_features = pickle.load(fr) 
                    captions_.append(captions)
                    # text_features = torch.cat([text_features_prior, text_features])
                    text_features_.append(text_features)
                except:
                    f = open(str(path)[:-3]+'txt')
                    captions = torch.cat(                                   # [7, 77]
                        [clip.tokenize(caption.replace('\n','').strip(), truncate=True).to(self.device)]
                    )
                    captions_.append(captions.unsqueeze(0))
                    text_features = self.encode_text(captions)
                    text_features_.append(text_features.unsqueeze(1))
                    with open(str(path)[:-3]+'pickle', 'wb') as fw:
                        pickle.dump((captions_[-1], text_features_[-1]), fw)
            text_features = torch.cat(text_features_, dim=1)                # [12, 96, 7, 77, self.EMBEDDING_DIM]
            captions = torch.cat(captions_)                                 # [96, 7, 77]
            num_of_class = captions.shape[1]                                # 7

            image_features = self.ln_post(image_features)                   # [12, 96, 768]
            image_features = image_features @ self.visual_projection        # [12, 96, 512]

            text_features = self.ln_final(text_features)                    # [12, 96, 7, 77, 512]
            text_features = text_features.view(12,-1,77,self.EMBEDDING_DIM) # [12, 96*7, 77, 512]
            text_features = torch.einsum('abcd,adz->abcz',
                                text_features[:,
                                    torch.arange(text_features.shape[1]),   # [96*7]
                                    captions.view(-1,77).argmax(-1)         # [96*7, 77] -> [96*7]
                                ].view(self.num_of_textual_encoder_layers, -1, num_of_class, self.EMBEDDING_DIM),  
                                                                            # [12, 96, 7, 512]
                                self.textual_projection)                    # [12, 512, self.EMBEDDING_DIM]

            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (12, 96, self.EMBEDDING_DIM)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # (12, 96, 7, self.EMBEDDING_DIM)

            score = 0
            score_tensor = torch.einsum("abc,xbzc->bzax",image_features, text_features) 
            score_tensor = score_tensor.reshape(*score_tensor.shape[:2],-1)

        if self.score_type == 'max':
            score = torch.max(score_tensor, dim=-1)[0]   
        elif self.score_type == 'mean':
            score = torch.mean(score_tensor, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score
        return logits
  
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

