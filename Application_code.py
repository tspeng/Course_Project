#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tishun Peng
"""
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.manifold import TSNE
import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np
from tika import parser
import glob

from PIL import ImageTk, Image
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import pandas as pd     
import seaborn as sns
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re


class Application(tk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master)
        
        self.master = master
        self.screenheight=master.winfo_screenheight()
        self.screenwidth=master.winfo_screenwidth()
        master.geometry("{0}x{1}+{2}+{3}".format(
            int(self.screenwidth/2),int(self.screenheight/2),int(self.screenwidth/4),int(self.screenheight/4)))
        
        self.grid()
        self.create_widgets()
        
        self.topic_clusters = tk.Frame(self.master)
    
    def create_widgets(self):
        
        self.no_topics=5
        self.no_features = 1000
        
        fr = tk.Frame(self.master)
        fr.grid(row=0, column=0)
        
        self.train_lda = tk.Button(fr,fg="blue",bg="white",text="Topic Modeling using LDA",
                                   command=self.train_lda_fun)
        self.train_lda.grid(row=0,column=0,sticky='w')
        
        self.train_nmf = tk.Button(fr,fg="blue",bg="white",text="Topic Modeling using NMF",
                                   command=self.train_nmf_fun)
        self.train_nmf.grid(row=0,column=1,sticky='w')
        
        self.predict_lda = tk.Button(fr,fg="blue",bg="white",text="Predict Report Topic using LDA",state='disabled',
                                   command=self.predict_topics)
        self.predict_lda.grid(row=1,column=0,sticky='w')
    
        self.predict_nmf = tk.Button(fr,fg="blue",bg="white",text="Predict Report Topic using NMF",state='disabled',
                                   command=self.predict_topics)
        self.predict_nmf.grid(row=1,column=1,sticky='w')
        
        self.label1=tk.Label(fr, text="True Topic:")
        self.label1.grid(row=2,column=0,sticky='e')
        self.entry1_content=tk.StringVar()
        self.entry1=tk.Entry(fr,state='disabled',textvariable=self.entry1_content)
        self.entry1.grid(row=2,column=1,sticky='w')
        
        
        self.label2=tk.Label(fr, text="Predicted Topic:")
        self.label2.grid(row=3,column=0,sticky='e')
        self.entry2_content=tk.StringVar()
        self.entry2=tk.Entry(fr,state='disabled',textvariable=self.entry2_content)
        self.entry2.grid(row=3,column=1,sticky='w')
        

    def create_corpus(self):
    
        corpus=[]
        reports=glob.glob("training/*.pdf")
        
        for report in reports:
            parsed = parser.from_file(report)
            doc_clean = self.clean(parsed["content"])
            corpus.append(doc_clean)
        
        self.corpus=corpus
        print(corpus)
        return
        
    def clean(self,doc):
        
        #emmatizing,topwords prunning, puncutation prunning, and compound splittiing filtering using NLTK
        
        stop = set(stopwords.words('english'))
        exclude = set(string.punctuation) 
        lemma = WordNetLemmatizer()
        
        doc=re.sub('[0-9]+',' ',doc)
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split() if len(word)>=3)
        
        return normalized   
    
        
    def train_lda_fun(self):
        
        self.test_method="lda"
        
        # TF
        TF_vectorizer = CountVectorizer(max_df=0.95, max_features=self.no_features, stop_words='english')
        self.create_corpus()
        TF = TF_vectorizer.fit_transform(self.corpus)
        self.vocabulary=TF_vectorizer.vocabulary_
        self.feature_names=TF_vectorizer.get_feature_names()
        
        # Topic modeling gusing LDA
        lda_model=LatentDirichletAllocation(n_topics=self.no_topics, max_iter=300, learning_method='batch', learning_offset=10.,random_state=0)
        lda_topics = lda_model.fit_transform(TF)
        
        self.model=lda_model
        self.topics=lda_topics                  #n_doc x n_topics
        self.model_vectorizer=TF_vectorizer
        
        self.topic_word=lda_model.components_  #n_topics x n_feature_names
        
        self.topic_clusters.destroy()
        self.visualize_topics()
        self.plot_topic_hist()
        
        self.predict_lda['state']='normal'
        self.predict_nmf['state']='disabled'
        
        
        self.entry1['state']='disabled'
        self.entry1_content.set('')
        
        self.entry2['state']='disabled'
        self.entry2_content.set('')
        
        return
    
    def train_nmf_fun(self):
        
        self.test_method="nmf"
     
        # TFIDF 
        TFIDF_vectorizer = TfidfVectorizer(use_idf=True,smooth_idf=True, sublinear_tf=False,max_df=0.5, max_features=self.no_features, stop_words='english',ngram_range=(1, 2))
        self.create_corpus()
        TFIDF = TFIDF_vectorizer.fit_transform(self.corpus)
        
        self.vocabulary=TFIDF_vectorizer.vocabulary_              #Extrac the vocabulary, which can be used to create document-term-matrix with the same vocabulary
        self.feature_names=TFIDF_vectorizer.get_feature_names()

        #Topic modelling using NMF
        nmf_model=NMF(n_components=self.no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')
        nmf_topics =nmf_model.fit_transform(TFIDF)

        self.model=nmf_model
        self.topics=nmf_topics                  #n_doc x n_topics
        self.model_vectorizer=TFIDF_vectorizer
        
        self.topic_word=nmf_model.components_  #n_topics x n_feature_names
        
        self.topic_clusters.destroy()
        self.visualize_topics()
        self.plot_topic_hist()
        
        self.predict_nmf['state']='normal'
        self.predict_lda['state']='disabled'
        
        self.entry1['state']='disabled'
        self.entry1_content.set('')
        
        self.entry2['state']='disabled'
        self.entry2_content.set('')
        
        return

    def visualize_topics(self):
                
        model=self.model
        topics=self.topics
        model_vectorizer=self.model_vectorizer
    
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        
        topics_2D_coord = tsne_model.fit_transform(topics)
        
        n_top_words = 5                # number of keywords we show
        
        topic_keys = []
        for i in range(topics.shape[0]):
            topic_keys.append(topics[i].argmax())
            
        topic_summaries = []
        topic_word = model.components_  # all topic words
        vocab = model_vectorizer.get_feature_names()
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)[:-(n_top_words + 1):-1]] # get!
            topic_summaries.append(' '.join(topic_words)) # append!

        #Plot clusters
        sns.set_style('white')
        
        #define a custom palette
        customPalette = ["#ff0000",  "#ff8000", '#630C3A', '#39C8C6', '#0000ff', '#00ff00', "#7f7f7f","#bcbd22", "#17becf"]
        sns.set_palette(customPalette)
        sns.palplot(customPalette)
        
        ndoc=topics_2D_coord.shape[0]
        data = pd.DataFrame(index=range(ndoc), columns=['x','y','label'])
        
        for i in range(ndoc):
            data.loc[i,['x','y']]=topics_2D_coord[i,:]
            data.loc[i,['label']]="Topic-"+ str(topic_keys[i]) + ":" + topic_summaries[topic_keys[i]]
        
        
        #plot data with seaborn
        facet = sns.lmplot(data=data, x='x', y='y', hue='label',fit_reg=False,legend=False,size=4,aspect=1.5)
        
        #add a legend
        leg = facet.ax.legend(bbox_to_anchor=[0, 1],loc=3, title="Topic legend", fancybox=False)
        #change colors of labels
        for i, text in enumerate(leg.get_texts()):
            plt.setp(text, color = customPalette[i])
        
        facet.savefig('topics.jpg')
        
        self.topic_clusters = tk.Frame(self.master)
        self.topic_clusters.grid(row=2,column=0)
        
        img = ImageTk.PhotoImage(Image.open('topics.jpg'))
        
        self.topic_clusters_img = tk.Label(self.topic_clusters,image=img)
        self.topic_clusters_img.image = img
        self.topic_clusters_img.grid(row=2,column=0)
    
    def predict_topics(self):
        
        root1=tk.Tk()
        root1.withdraw()
        file_path = askopenfilename()
        root1.update()
        root1.destroy()
  
        corpus=[]
        parsed = parser.from_file(file_path)
        doc_clean = self.clean(parsed["content"])
        corpus.append(doc_clean)
        
        if (self.test_method=='nmf'):
            # NMF is able to use tf-idf
            test_TFIDF_vectorizer = TfidfVectorizer(use_idf=True,smooth_idf=True, sublinear_tf=False,max_df=0.5, max_features=self.no_features,vocabulary=self.vocabulary,stop_words='english',ngram_range=(1, 2))
            test_TFIDF = test_TFIDF_vectorizer.fit_transform(corpus)
            topics_pred =self.model.transform(test_TFIDF)
            
        elif (self.test_method=='lda'):
            test_TF_vectorizer = CountVectorizer(max_df=0.95, max_features=self.no_features,vocabulary=self.vocabulary, stop_words='english')
            test_TF = test_TF_vectorizer.fit_transform(corpus)
            topics_pred =self.model.transform(test_TF)
            
        temp=np.argsort(topics_pred[0,:])
        
        self.entry1['state']='normal'
        self.entry1_content.set(file_path.split('/')[-1].split('_')[0])
        
        self.entry2['state']='normal'
        self.entry2_content.set('Topic '+str(temp[-1]))
        
        return
    
    def plot_topic_hist(self):
        
        n_top_words=40
        topic_word = self.model.components_                # all topic words
        vocab = self.model_vectorizer.get_feature_names()
        
        topic_dist=topic_word[0,:]
        topic_words = np.array(vocab)[np.argsort(topic_dist)[:-(n_top_words + 1):-1]] # get!
        y=topic_dist[np.argsort(topic_dist)[:-(n_top_words + 1):-1]]
            
        fr2 = tk.Frame(self.master)
        fr2.grid(row=2, column=1,sticky='w')
        f=Figure(figsize=(7,5),tight_layout=1)
        a=f.add_subplot(1,1,1)
        
        a.bar(topic_words,y,align='center') # A bar chart
        a.tick_params(axis='x',labelrotation=90)
        a.set_ylabel('Word distribution for Topic '+ str(0),fontdict={'fontsize':13})
        a.set_xlabel('Top ' + str(n_top_words) +' words',fontdict={'fontsize':13})
        a.set_xticklabels(labels=topic_words,fontdict={'fontsize':13})
        
        canvas=FigureCanvasTkAgg(f,fr2)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        def sel():
            topic_dist=topic_word[var.get(),:]
            topic_words = np.array(vocab)[np.argsort(topic_dist)[:-(n_top_words + 1):-1]] # get!
            y=topic_dist[np.argsort(topic_dist)[:-(n_top_words + 1):-1]]
                
            fr2 = tk.Frame(self.master)
            fr2.grid(row=2, column=1, sticky='w')
            
            f=Figure(figsize=(7,5),tight_layout=1)
            a=f.add_subplot(1,1,1)
            
            a.bar(topic_words,y,align='center') # A bar chart
            a.tick_params(axis='x',labelrotation=90)  #use dir(object) to find attributes
            a.set_ylabel('Word distribution for Topic '+ str(var.get()),fontdict={'fontsize':13})
            a.set_xlabel('Top ' + str(n_top_words) +' words',fontdict={'fontsize':13})
            a.set_xticklabels(labels=topic_words,fontdict={'fontsize':13})
            
            canvas=FigureCanvasTkAgg(f,fr2)
            canvas.draw()
            canvas.get_tk_widget().pack()
   
        
        fr1 = tk.Frame(self.master)
        fr1.grid(row=0, column=1,sticky='w')
        
        var=tk.IntVar()
        for i in range(self.no_topics):
            tk.Radiobutton(fr1, text="Topic "+str(i), variable=var, value=i, command=sel).pack(side='left')

if __name__ == "__main__":  
    root = tk.Tk()
    app = Application(master=root)
    app.master.title("Accident Topic Clustering and Prediction")
    app.mainloop()

#https://github.com/rdspring1/LSH_LDA/blob/master/lda.py
#https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html
#https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
#https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html
#http://effbot.org/tkinterbook/tkinter-index.htm
#https://docs.python.org/2/library/tkinter.html
#https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python


