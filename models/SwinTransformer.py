#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[4]:


import numpy as np
from thop import profile


# In[6]:


from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath


# In[34]:


#(Shifted) window self-attention module
class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, module_type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.module_type = module_type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias = True)
        self.relative_position_params = nn.Parameter(torch.zeros((2*window_size - 1)*(2*window_size - 1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        
        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size - 1, 2*window_size - 1, self.n_heads).transpose(1,2).transpose(0,1))
        
    def forward(self, x):
        _, H, W, _ = x.shape
        #pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        #do cyclic shift if it is SWMSA
        if self.module_type != 'W': 
            x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims = (1,2))
        #window partition in one line
        #x = window_partition(x, self.window_size)
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_window_num = x.size(1)
        w_window_num = x.size(2)
        
        #window num validation
        #assert h_window_num == w_window_num
        
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (three c) -> three b nw np c', c=self.head_dim).chunk(3, dim=0)
        #compute q*k.T using einsum
        attn = torch.einsum('hbwpc, hbwqc->hbwpq', q, k) * self.scale
        #add relative embedding
        attn = attn + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        #if this module is SWSA
        if self.module_type != 'W':
            attn_mask = self.generate_mask(h_window_num, w_window_num, self.window_size, shift=self.window_size//2)
            attn = attn.masked_fill_(attn_mask, float("-inf"))
        
        attn = nn.functional.softmax(attn, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', attn, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1 = h_window_num, p1 = self.window_size)
        
        #undo cyclic shift if it is SWMSA
        if self.type!='W': 
            output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        if pad_r > 0 or pad_b > 0:
            output = output[:, :H, :W, :].contiguous()
        return output
    
    def generate_mask(self, w1, w2, p, shift):
        """generate SWMSA mask"""
        attn_mask = torch.zeros(w1, w2, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.module_type == 'W':
            return attn_mask
        
        s = p - shift
        #since the attn masks are already given in the offical implementation, we directly construct it here
        #construct attn masks of the last row windows(window2) in two lines
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        #construct attn masks of last colomn windows(window1) in two lines
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        #window3 is automatically constructed by operations above and window0 is already filled by zeros
        #reshape the attention mask to (1 1 nW p^2 p^2)
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask
    
    def relative_embedding(self):
        cord = torch.tensor(np.array([[i,j] for i in range(self.window_size) for j in range(self.window_size)]))
        #use broadcast to calculate relative relation
        relation = cord[:, None, :] - cord[None, : ,:] + self.window_size - 1
        return self.relative_position_params[:, relation[:,:,0], relation[:,:,1]]
                
        


# In[35]:


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path=0, module_type='W', img_size=None):
        """Swin transformer block"""
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert module_type in ['W', 'SW']
        self.module_type = module_type
        if img_size <= window_size:
            self.module_type = 'W'
            
        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.module_type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.module_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4*input_dim),
            nn.GELU(),
            nn.Linear(4*input_dim, output_dim)
        )
        
    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x
              


# In[36]:


class SwinTransformer(nn.Module):
    def __init__(self, num_classes, depths_config=[2,2,6,2], embed_dim=96, head_dim=32, window_size=7, drop_path_rate=0.2, img_size=224):
        super(SwinTransformer, self).__init__()
        self.depths_config = depths_config
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.window_size = window_size
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_config))]
        
        begin = 0
        self.stage1 = [nn.Conv2d(3, embed_dim, kernel_size=4, stride=4),
                      Rearrange('b c h w -> b h w c'),
                      nn.LayerNorm(embed_dim),] + \
                      [Block(embed_dim, embed_dim, head_dim, window_size, dpr[i+begin], 'W' if not i%2 else 'SW', img_size//4)
                      for i in range(depths_config[0])]
        begin += depths_config[0]
        self.stage2 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2),
                      nn.LayerNorm(4*embed_dim), nn.Linear(4*embed_dim, 2*embed_dim, bias=False),] + \
                      [Block(2*embed_dim, 2*embed_dim, head_dim, window_size, dpr[i+begin], 'W' if not i%2 else 'SW', img_size//8)
                      for i in range(depths_config[1])]
        begin += depths_config[1]
        self.stage3 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2),
                      nn.LayerNorm(8*embed_dim), nn.Linear(8*embed_dim, 4*embed_dim, bias=False),] + \
                      [Block(4*embed_dim, 4*embed_dim, head_dim, window_size, dpr[i+begin], 'W' if not i%2 else 'SW', img_size//16)
                      for i in range(depths_config[2])]
        begin += depths_config[2]
        self.stage4 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2),
                      nn.LayerNorm(16*embed_dim), nn.Linear(16*embed_dim, 8*embed_dim, bias=False),] + \
                      [Block(8*embed_dim, 8*embed_dim, head_dim, window_size, dpr[i+begin], 'W' if not i%2 else 'SW', img_size//16)
                      for i in range(depths_config[3])]
        
        self.stage1 = nn.Sequential(*self.stage1)
        self.stage2 = nn.Sequential(*self.stage2)
        self.stage3 = nn.Sequential(*self.stage3)
        self.stage4 = nn.Sequential(*self.stage4)
        
        self.norm = nn.LayerNorm(8*embed_dim)
        self.avgpool = Reduce('b h w c -> b c', reduction='mean')
        self.head = nn.Linear(8*embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
        
    def forward(self, x):
        #padding before patch embedding
        _, _, H, W = x.size()
        if W % 4 != 0:
            x = F.pad(x, (0, 4 - W % 4))
        if H % 4 != 0:
            x = F.pad(x, (0, 0, 0, 4 - H % 4))
        x = self.stage1(x)
        #padding before patch merging
        _, H, W, _ = x.size()
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        print('after patch merging: ', x.size())
        x = self.stage2(x)
        #padding before patch merging
        _, H, W, _ = x.size()
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = self.stage3(x)
        #padding before patch merging
        _, H, W, _ = x.size()
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = self.stage4(x)
        
        x = self.norm(x)
        x = self.avgpool(x)
        x = self.head(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            


# In[37]:


def Swin_T(num_classes, depths_config=[2,2,6,2], embed_dim=96, **kwargs):
    return SwinTransformer(num_classes, depths_config=depths_config, embed_dim=embed_dim, **kwargs)

def Swin_S(num_classes, depths_config=[2,2,18,2], embed_dim=96, **kwargs):
    return SwinTransformer(num_classes, depths_config=depths_config, embed_dim=embed_dim, **kwargs)

def Swin_B(num_classes, depths_config=[2,2,18,2], embed_dim=128, **kwargs):
    return SwinTransformer(num_classes, depths_config=depths_config, embed_dim=embed_dim, **kwargs)

def Swin_L(num_classes, depths_config=[2,2,18,2], embed_dim=192, **kwargs):
    return SwinTransformer(num_classes, depths_config=depths_config, embed_dim=embed_dim, **kwargs)


# In[39]:


if __name__ == '__main__':
    test_model = Swin_T(1000).cuda()
    n_parameters = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(test_model)
    dummy_input = torch.rand(3,3,224,224).cuda()
    output = test_model(dummy_input)
    print(output.size())
    print(n_parameters)


# In[ ]:




