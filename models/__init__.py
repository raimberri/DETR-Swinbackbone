#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(args):
    return build(args)
