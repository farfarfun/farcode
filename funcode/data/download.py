#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/05/23 16:53
# @Author  : niuliangtao
# @Site    : 
# @File    : download.py
# @Software: PyCharm
import os
import pycurl

__all__ = ['download_file']


def download_file(url, path):
    if not os.path.exists(path):
        print("downloading from " + url + " to " + path)
        with open(path, 'wb') as f:
            c = pycurl.Curl()
            c.setopt(pycurl.URL, url)
            c.setopt(pycurl.WRITEDATA, f)
            c.perform()
            c.close()
        print('download success')
