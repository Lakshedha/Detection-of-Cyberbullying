from tkinter import Tk
from tkinter import *
import csv
import pandas as pd
import time
import glob
import random
import matplotlib.pyplot as plt
from preprocessing import *
from word_token import *
from glove_word_embed import *
from ensemble_cnn import *
from graph import *
import shutil
import time
import sys
import os
import json
import warnings

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy
from tensorflow.compat.v1 import get_default_graph
from tensorflow.nn import max_pool2d
from tensorflow.compat.v1.train import Optimizer
from tensorflow.compat.v1.ragged import RaggedTensorValue
from classify import *

root = Tk()
root.title("Ensemble CNN with IbI Logics Algorithm (ILA) for Cyberbullying Detection on Social Media (Twitter)")
width = 900
height = 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
root.resizable(0, 0)

        
def readf():
    warnings.filterwarnings('ignore')
    # Ignore deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print ("\n\t\t\t==========================******************* LOADING DATASET *******************==========================")
    file_name='./dataset/cyberbullying_tweets.csv'
    with open(file_name, 'rt',encoding='utf-8') as f:
        original_file = f.read()
        rowsplit_data = original_file.splitlines()    
        
        for row in  rowsplit_data:
            print (row)
            
        dirName1 = './output/Preprocessed'
        if not os.path.exists(dirName1):
            os.makedirs(dirName1)
        else:
            shutil.rmtree(dirName1)
            os.makedirs(dirName1)

        dirName2 = './output/Glove_word_embedded'
        if not os.path.exists(dirName2):
            os.makedirs(dirName2)
        else:
            shutil.rmtree(dirName2)
            os.makedirs(dirName2)

        dirName3 = './output/Ensemble_cnn'
        if not os.path.exists(dirName3):
            os.makedirs(dirName3)
        else:
            shutil.rmtree(dirName3)
            os.makedirs(dirName3)

        dirName4 = './output/Classification'
        if not os.path.exists(dirName4):
            os.makedirs(dirName4)
        else:
            shutil.rmtree(dirName4)
            os.makedirs(dirName4)

                   
    count_rows(file_name);
    num_rows = count_rows(file_name)
    print(f"\n\nThe total number of rows in {file_name} is: {num_rows-1}")        
    

    print ("\n\t\t\t==========================******************* DATA PRE-PROCESSING ******************==========================")    
    time.sleep(2)
    data_cleaning();
    time.sleep(2)
    load_NLTK_resorces();

    print ("\n\t\t\t==========================******************* GLOVE WORD EMBEDDING PROCESS ******************==========================")    
    time.sleep(2)
    word_embedded_technique();
    
    print ("\n\t\t\t==========================******************* CYBER-BULLYING DETECTION USING ENSEMBLE CNN ALGORITHM ******************==========================")    
    time.sleep(2)
    perform_ensemble_cnn_alg(num_models=3)
        
    print ("\n\t\t\t==========================******************* CYBER-BULLYING CLASSIFICATION ******************==========================")    
    time.sleep(2)
    data_classify('./output/Ensemble_cnn/ensemble_cnn_prediction.csv', './output/Classification/bullying_output.csv', './output/Classification/non_bullying_output.csv');
   
    time.sleep(2)
    graph()
       
def count_rows(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = len(list(reader))
    return row_count

def graph():
    root.destroy()
    acc_graph()
    prec_graph()
    recl_graph()
    f1score_graph()
    loss_graph()
    
Top = Frame(root, bg="light green", bd=2,  relief=RIDGE)
Top.pack(side=TOP, fill=X)
Form = Frame(root, bg="light green", height=200)
Form.pack(side=TOP, pady=20)
lbl_title = Label(Top, bg="yellow", text = "ENSEMBLE CNN WITH IBI LOGICS ALGORITHM (ILA) FOR", font=('Arial Bold', 22))
lbl_title.pack(fill=X)
lbl_title = Label(Top, bg="yellow", text = "CYBERBULLYING DETECTIONON SOCIAL MEDIA (Twitter)", font=('Arial Bold', 22))
lbl_title.pack(fill=X)
btn_login = Button(Form, bg="red", text="START SIMULATION", font =('Times New Roman', 22), width=20, height=20, command=readf)
btn_login.pack()
btn_login.bind('<Return>', readf)
btn_log = Button(Form, bg="blue", text="START SIMULATION", font="-weight bold", width=250, height=55)
btn_log.pack()
