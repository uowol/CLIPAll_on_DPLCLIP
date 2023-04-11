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

ALGORITHMS = [
    'ERM',
    'FrozenERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'CLIP',
    'DPLCLIP',
    'CLIPALL',
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


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)


class FrozenERM(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FrozenERM, self).__init__(input_shape, num_classes, num_domains,
                                hparams)

        for param in self.featurizer.parameters():
            param.requires_grad = False
        print('Set self.model.parameters.reguires_grad = False!')

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class CLIP(Algorithm):
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
    
    def predict(self, x):
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)


import pickle

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
        self.visual_projection = nn.Parameter(
            torch.stack([
                visual_scale * torch.randn((visual_width, output_dim), dtype=self.dtype).to(self.device)
                for _ in range(self.num_of_visual_encoder_layers - 1)
            ]+[self.clip_model.visual.proj])).requires_grad_(True)
        self.textual_projection = nn.Parameter(
            torch.stack([
                textual_scale * torch.randn((textual_width, output_dim), dtype=self.dtype).to(self.device)
                for _ in range(self.num_of_textual_encoder_layers - 1)
            ]+[self.clip_model.text_projection])).requires_grad_(True)
        self.score_type = hparams['score_type']
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]

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
        ### 3개의 도메인은 랜덤하게 주어지는가?
        ### len(minibatches) = 3
        ### len(data) = 4, images, labels, paths, labels
        ### shape of all_x = torch.Size([32, 3, 224, 224]) * 3
        ### shape of all_y = torch.Size([96])
        
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
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
                    captions, text_features_prior, text_features = pickle.load(fr)
                    # captions, text_features = pickle.load(fr) # Note: pickles 전부 update 필요
                captions_.append(captions)
                text_features = torch.cat([text_features_prior, text_features])
                text_features_.append(text_features)
            except:
                f = open(str(path)[:-3]+'txt')
                captions = torch.cat(
                    [
                        clip.tokenize(caption.replace('\n','').strip()).to(self.device) 
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

        image_features = self.ln_post(image_features)   # [12, 96, self.EMBEDDING_DIM]
        image_features = image_features @ self.visual_projection

        text_features = self.ln_final(text_features)    # [12, 96, 7, 77, self.EMBEDDING_DIM]
        text_features = text_features.view(12,-1,77,self.EMBEDDING_DIM)# [12, 96*7, 77, self.EMBEDDING_DIM]
        text_features = torch.einsum('abcd,adz->abcz',
                            text_features[:,
                                torch.arange(text_features.shape[1]),   # [96*7]
                                captions.view(-1,77).argmax(-1)         # [96*7, 77] -> [96*7]
                            ].view(self.num_of_textual_encoder_layers, -1, num_of_class, self.EMBEDDING_DIM),  # [12, 96, 7, self.EMBEDDING_DIM]
                            self.textual_projection)                    # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]

        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (1, 96, self.EMBEDDING_DIM)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (1, 96, 7, self.EMBEDDING_DIM)

        score = 0
        score_tensor = torch.einsum("abc,xbzc->bzax",image_features, text_features) 
        score_tensor = score_tensor.reshape(*score_tensor.shape[:2],-1)

        if self.score_type == 'max':
            score = torch.max(score_tensor, dim=-1)[0]   
        elif self.score_type == 'mean':
            score = torch.mean(score_tensor, dim=-1)
        # elif self.score_type == 2:
        #     score = torch.sigmoid(100 * (score_tensor-(self.threshold*self.threshold_weight))).sum(dim=-1)
        # elif self.score_type == 3:
        #     score = score_tensor.max(dim=-1)[0].mean(dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score

        loss = F.cross_entropy(logits, all_y)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
    def predict(self, x, paths):
        # encode image for each domain.
        image_features = self.encode_image(x)
        
        #  encode text for each domain.
        text_features_ = []
        captions_ = []
        for path in paths:
            try:
                with open(str(path)[:-3]+'pickle', 'rb') as fr:
                    captions, text_features_prior, text_features = pickle.load(fr)
                    # captions, text_features = pickle.load(fr) # Note: pickles 전부 update 필요
                captions_.append(captions)
                text_features = torch.cat([text_features_prior, text_features])
                text_features_.append(text_features)
            except:
                f = open(str(path)[:-3]+'txt')
                captions = torch.cat(
                    [
                        clip.tokenize(caption.replace('\n','').strip()).to(self.device) 
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

        image_features = self.ln_post(image_features)   # [12, 96, self.EMBEDDING_DIM]
        image_features = image_features @ self.visual_projection

        text_features = self.ln_final(text_features)    # [12, 96, 7, 77, self.EMBEDDING_DIM]
        text_features = text_features.view(12,-1,77,self.EMBEDDING_DIM)# [12, 96*7, 77, self.EMBEDDING_DIM]
        text_features = torch.einsum('abcd,adz->abcz',
                            text_features[:,
                                torch.arange(text_features.shape[1]),   # [96*7]
                                captions.view(-1,77).argmax(-1)         # [96*7, 77] -> [96*7]
                            ].view(self.num_of_textual_encoder_layers, -1, num_of_class, self.EMBEDDING_DIM),  # [12, 96, 7, self.EMBEDDING_DIM]
                            self.textual_projection)                    # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]

        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (12, 96, self.EMBEDDING_DIM)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (12, 96, 7, self.EMBEDDING_DIM)

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
     

# rename to DPL (Domain Prompt Learning)
class DPLCLIP(CLIP):
    def get_prompts(self, path=None):   # path = "---.jpg"
        if path == None:
            #  initial prompt.
            prompt_prefix = ' '.join(['X'] * self.hparams['num_domain_tokens'])
            
            if self.sentence_prompt:
                print('Using sentence_prompt in DPLCLIP...')
                classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.hparams['class_names']]
            else:
                classnames = [name.replace('_', ' ') for name in self.hparams['class_names']]
            prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        else:
            f = open(str(path)[:-3]+'txt')
            prompts = [caption.replace('\n','').strip() for caption in f.readlines() if caption[:7] == "It is a"]
        return prompts

    def __init__(self, input_shape, num_classes, num_domains, hparams, sentence_prompt=False):
        super(DPLCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.sentence_prompt = sentence_prompt
        prompts = self.get_prompts()
        
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        # [7, 77]
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        # [7, 77, 512]
        
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
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
            
    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_path = sum([list(data[2]) for data in minibatches], [])

    ### BEGIN_EDITED
        text_features_ = []
        tokenized_prompts_ = []
        token_prefix = []
        token_suffix = []
        for path in all_path:
            prompts = self.get_prompts(path)    # [7]
            tokenized_prompts = torch.cat(      # self.tokenized_prompts, [7, 77]
                [clip.tokenize(p) for p in prompts]
            ).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)
                                                # [7, 77, 512]
            
            token_prefix.append(embedding[:, :1, :].unsqueeze(0))  # SOS
            token_suffix.append(embedding[:, self.hparams['num_domain_tokens'] + 1:, :].unsqueeze(0))  # CLS, EOS
            tokenized_prompts_.append(tokenized_prompts.unsqueeze(0))
        tokenized_prompts = torch.cat(tokenized_prompts_)               # [96, 7, 77]
        token_prefix = torch.cat(token_prefix)                          # [96, 7,  1, 512]
        token_suffix = torch.cat(token_suffix)                          # [96, 7, 60, 512]
        # print("LOG:", token_prefix.shape)
        # print("LOG:", token_suffix.shape)
    ### END_EDITED

        ### Note: domain 별로 이미지를 받아 domain_feature를 뽑아냄
        ###       나의 모델의 경우 어떤 도메인의 domain별 이미지를 받아와야 하는 조건이 없음. 어떤 도메인이 있는지도 모름
        ###       DPLCLIP: domain 별 이미지    vs    CLIPALL: 이미지별 caption        

        #  encode image for each domain.
        image_features = [self.clip_model.encode_image(x) for x in all_x]
        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in image_features]
        # print("LOG:", len(domain_features), domain_features[0].shape)   # 3, [32, 8192]
        image_features = torch.cat(image_features)
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
        mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]
        # print("LOG:", mean_domain_features[0].shape)                    # [1, 8192]
        
        #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
        _mean_domain_features = [feature.repeat_interleave(len(self.hparams['class_names']), dim=0) for feature in mean_domain_features]
        # print("LOG:", _mean_domain_features[0].shape)                   # [7, 8192]
        
        #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
        text_features = [self._get_text_features(
                                feature,                                # [7, 8192]
                                tokenized_prompts[(i*32):((i+1)*32)],   # [96, 7, 77]
                                token_prefix[(i*32):((i+1)*32)],        # [96, ...]
                                token_suffix[(i*32):((i+1)*32)]         # [96, ...]
                            ) for (i, feature) in enumerate(_mean_domain_features)]
        # print("LOG:", len(text_features), text_features[0].shape)       # 3, [32, 7, 512]
        text_features = torch.cat(text_features)
        # print("LOG:", image_features.shape, text_features.shape)        # [96, 512], [96, 7, 512]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * torch.einsum("ab,acb->ac", image_features, text_features)
        loss = F.cross_entropy(logits_per_image, all_y)
        # print("LOG:",logits_per_image.shape, all_y.shape, logits_per_image.argmax(-1), all_y, loss)
        ### Note: y_pred = 0~20, y = 0~6, 범위가 다른데 괜찮은지

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def _encode_text(self, domain_feature):
        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype) # [7, 77, 512]
        return x

    def _get_text_features(self, domain_feature, 
                            tokenized_prompts=None, token_prefix=None, token_suffix=None,
                            coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        
        #  repeat domain_feature: [7, 16, self.EMBEDDING_DIM] -> [32, 7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.unsqueeze(0).repeat(tokenized_prompts.shape[0],1,1,1)
        # print("LOG:",domain_feature.shape)

        #  reshape domain_feature: [32, 7, 16, self.EMBEDDING_DIM] -> [32, 7, 77, self.EMBEDDING_DIM]
        if token_prefix == None:
            domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
        else:
            domain_feature = torch.cat([token_prefix, domain_feature, token_suffix], dim=2)
        # print("LOG:",domain_feature.shape)  # [32, 7, 77, 512]

        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        text_features = torch.cat([self._encode_text(feature).unsqueeze(0) for feature in domain_feature])
        # print("LOG:",text_features.shape)   # [32, 7, 77, 512]
        
        text_features = text_features.view(-1,77,self.EMBEDDING_DIM)# [32*7, 77, self.EMBEDDING_DIM]
        # print("LOG:",text_features.shape, tokenized_prompts.view(-1,77).shape)
        text_features = torch.einsum('bcd,dz->bcz',                 # [32, 7, self.EMBEDDING_DIM]
                            text_features[
                                torch.arange(text_features.shape[0]),   # [32*7]
                                tokenized_prompts.view(-1,77).argmax(-1)# [32*7, 77] -> [32*7]
                            ].view(tokenized_prompts.shape[0], -1, self.EMBEDDING_DIM), # [32, 7, self.EMBEDDING_DIM]
                            self.clip_model.text_projection)         # [self.EMBEDDING_DIM, self.EMBEDDING_DIM]
        # print("LOG:",text_features.shape)   # [32, 7, 512]

        return text_features

    def predict(self, x, paths):
        N = x.shape[0]; M = N/3

        text_features_ = []
        tokenized_prompts_ = []
        token_prefix = []
        token_suffix = []
        for path in paths:
            prompts = self.get_prompts(path)    # [7]
            tokenized_prompts = torch.cat(      # self.tokenized_prompts, [7, 77]
                [clip.tokenize(p) for p in prompts]
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


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)

class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}
