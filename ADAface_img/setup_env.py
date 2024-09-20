#!/usr/bin/env python3

import os

import gdown



def install_packages():

    os.system('pip install -q pytorch-lightning==1.8.6')



def clone_repo():

    if not os.path.exists('AdaFace'):

        os.system('git clone https://github.com/mk-minchul/AdaFace')



def download_model():

    if not os.path.exists('AdaFace/pretrained'):

        os.makedirs('AdaFace/pretrained')

    url = "https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT"

    output = "AdaFace/pretrained/adaface_ir101_webface12m.ckpt"

    gdown.download(url, output, quiet=False)



if __name__ == '__main__':

    install_packages()

    clone_repo()

    download_model()

