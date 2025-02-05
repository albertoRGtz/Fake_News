from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import json
import torch
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import json
import re
import numpy as np
import itertools
import tensorflow as tf
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import string
from transformers import pipeline
from tensorflow.keras.models import load_model
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, BertModel, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
import search

pass


class FakeEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.actions_grid=['same_query','next_query','record_return','record_step','stop']
    self.actions_kb=["no_relation","positive","negative"]
    self.noticias= pd.read_csv("/home/alberto/servicio/Fake_News/gym_fake/data/data.csv").drop_duplicates() ##base de noticias
    self.num_noticias=len(self.noticias)-1
    #self.web_data_frame = pd.DataFrame()
    self.noticias_ind=-1
    self.querys_step=[]
    self.episodeDF=[]
    self.queryDF=[]
    self.queryDF_indexes=[]
    self.row_index_query=[]
    self.query_seen=[]
    self.agent_db=[0,0]   #Sera el contador 
    self.realidad=[0]
    self.hist_simil=[]
    self.hist_real=[]
    self.episode_continues=False
    self.num_steps_per_episode=0
    self.state=[0,0]
    self.engine = 0
    # 0-duckduck; 1-scholar; 2-bing; 3-researchgate
    self.env=self
    self.unmasker = pipeline('fill-mask', model='bert-base-uncased')
    self.actionHash={key:selection for key,selection
                      in enumerate(
                        itertools.product(
                          range(len(self.actions_grid)-1),
                          range(len(self.actions_kb)-1)
                        )
                      )
                    }

    self.MODEL_NAME="bert-base-uncased"
    self.model = AutoModel.from_pretrained(self.MODEL_NAME,output_hidden_states=True)
    self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
    self.nlp1= pipeline('ner',model='vblagoje/bert-english-uncased-finetuned-pos', grouped_entities=True)
    self.summarizer = pipeline("summarization")
    self.options = webdriver.FirefoxOptions()
    self.options.add_argument('-headless')
    self.driver = webdriver.Firefox(executable_path=r'/home/alberto/servicio/Fake_News/gym_fake/envs/geckodriver', firefox_options=self.options)
    

    #Space variables
    high = np.array([
           200,
           200])
    low = np.array([
           0,
           0])
    
    self.action_space = spaces.Discrete(len(self.actionHash.keys()))
    self.observation_space = spaces.Box(low, high, dtype=np.int32)

  def _expandAction(self, action):
    action_grid,action_kb=self.actionHash[action]
    return action_grid,action_kb

  
  def step(self, action,agent_db=None):
    """ ejecutamos el paso y regresamos el reward """
    """ debemos regresar un arreglo del tipo:
    respuesta:["estado","reward","booleando(si he termiando o no)"]"""
    self.num_steps_per_episode+=1
    texto1=self.noticias['Headline'][self.noticias_ind]
    action_grid,action_kb=self._expandAction(action)
    query=self._action_grid_selector(action_grid)
    reward=self._get_reward(texto1,query,action_kb)
    self.query_seen[self.query_indexes[self.query_ind]]+=1
    query_seen=self.query_seen[self.query_indexes[self.query_ind]]
    stat=[]
    for i in range(0,768):
      #stat.append=self.state[0].detach().numpy()[i]
      stat.append(self.state[0][i])
    for i in range(0,768):
      #stat.append=self.state[1].detach().numpy()[i]
      stat.append(self.state[0][i])
    stat.append(self.agent_db[0])
    stat.append(self.agent_db[1])
    return [stat,reward,self.episode_continues,{}]


  def get_snippets(self, query):
    if self.engine == 0:
      return search.duckduck_(query, 'name', n=40, verbose=False)
    elif self.engine == 1:
      return search.scholar_(query, 'name', n=40, verbose=False)
    elif self.engine == 2:
      return search.bing_(query, 'name', n=40, verbose=False)
    else:
      return search.researchgate_(query, 'name', n=40, verbose=False)
    


  def reset(self):
    """ aquí avanzamos al siguiente episodio"""
    self.num_steps_per_episode=0
    if self.noticias_ind==self.num_noticias:
      self.noticias_ind=0
    else:
      self.noticias_ind+=1
    self.agent_db=[0,0]
    self.hist_simil=[]
    self.hist_real=[]

     #set stepepisodeDFDF
    Head=self.noticias['Headline'][self.noticias_ind]
    self.episodeDF= pd.DataFrame.from_dict(self.get_snippets(self.noticias['Headline'][self.noticias_ind]))
    self.realidad[0]=self.noticias['Label'][self.noticias_ind]
    #set Query lists and indexes for queryDFs
    self.querys_step=[Head,Head+'True',Head+'False'] #Los querys, se debe cambiar de momento no se necesita
    self.query_indexes=[0,1,2] #Los índices de la lista de querys, depende de como se haga el paso anterior
    self.row_index_query=[0]*len(self.querys_step) #En qué registro está cada query
    self.query_seen=[0]*len(self.querys_step) #Cuantos pasos he dado en ese query
    self.query_ind=0 #En qué query estoy

    #self.querydf=self.episodeDF.loc[self.episodeDF["search"] == self.querys_step[self.query_indexes[self.query_ind]]]
    self.episode_continues=False
    

    return self.render()

  def render(self, mode='human'):

    #return self.querydf.iloc[self.row_index_query[self.query_indexes[self.query_ind]]].tolist(),self.agent_db
    query_seen=self.query_seen[self.query_indexes[self.query_ind]]
    #return [(query,self.num_steps_per_episode,query_seen,agent_db),reward,self.episode_continues]
    return np.array([0]*1538,dtype=np.float64)

  def _action_grid_selector(self,action_grid):
      print(action_grid)
      if action_grid==0:
      #query_return is 0
        self._query_return()
      if action_grid==1:
      #query_step is 1
        self._query_query_step()
      if action_grid==2:
      #record_return is 2
        self._record_return()
      if action_grid==3:
      #record_return is 3  
        self._record_step()
      if action_grid==4:
      #stop is 4
        return self.episodeDF['text'][self.row_index_query[self.query_indexes[self.query_ind]]]
      return self.episodeDF['text'][self.row_index_query[self.query_indexes[self.query_ind]]]


    


    

  def _query_query_step(self):
    if self.query_ind==len(self.querys_step)-1:
      self.query_ind=self.query_ind
    else:
      self.query_ind+=1
    self.episodeDF=pd.DataFrame.from_dict(self.get_snippets(self.querys_step[self.query_ind]))

  def _query_return(self):
    if self.query_ind==0:
      self.query_ind=self.query_indexes[-1]
    else:
      self.query_ind-=1
    self.episodeDF=pd.DataFrame.from_dict(self.get_snippets(self.querys_step[self.query_ind]))

    
  def _record_return(self):
    
    if self.row_index_query[self.query_indexes[self.query_ind]]==0:
      self.row_index_query[self.query_indexes[self.query_ind]]=len(self.episodeDF)-1
    else:
      self.row_index_query[self.query_indexes[self.query_ind]]-=1

  def _record_step(self):
    
    if self.row_index_query[self.query_indexes[self.query_ind]]==len(self.episodeDF)-1:
      if self.query_ind==len(self.querys_step)-1:
        self.episode_continues=True
      else:
        self.row_index_query[self.query_indexes[self.query_ind]]=0
    else:
      self.row_index_query[self.query_indexes[self.query_ind]]+=1

    
    
  

####################################################################################################################################

  def tokeni(self,input):
    tokenized_text=[]
    for s in input['input_ids'].tolist():
      a=self.tokenizer.convert_ids_to_tokens(s)
      tokenized_text.append(a)
    return tokenized_text  


  def token_esp(self,tokens):
    palabras=[]
    toke=[]
    id=0
    i=0
    tokenl=[]
    anterior='token'
    for token in tokens:
      lista=[]
      idx=token.find('##'.lower())
      if id==1 and idx<0:
        palabras.append(anterior)
        result = []
        for item in tokenl:
          if item not in result:
            result.append(item)
        toke.append(result)
        tokenl=[]
      id=0
      if idx==0:
        id=1
        lista.append(anterior)
        lista.append(token)
        anterior=self.tokenizer.convert_tokens_to_string(lista)
        tokenl.append(i-1)
        tokenl.append(i)
      else: 
        anterior=token
      i+=1
 
    return palabras,toke

  def checar(self,palab,tokenized_text):
    i=0
    id=[]
    pal='non'
    for item in tokenized_text:
      idx=item.find(palab.lower())
      if idx==0:
        pal=item
        id=[i]
      i+=1
    if pal=='non':
      res=self.token_esp(tokenized_text)
      j=0
      for item in res[0]:
        idx=item.find(palab.lower())
        if idx==0:
          pal=item
          id=res[1][j]
          j+=1

    return pal,id


  def vec(self,lis,token_vecs_sum):
    token_vecs=[]
    listt=[]
    for i in range(0,len(lis)):
      listt.append(token_vecs_sum[lis[i]].tolist())
    token_vecs=torch.tensor(listt)  

    word_embedding = torch.mean(token_vecs, dim=0)

    return word_embedding

  def calculate_cos_distance(self,a,b):
    cose= np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return 1 - cose

  def separar(self,texto):
    org=self.nlp1(texto)
    Noun=[]
    Verbs=[]
    Adj=[]
    for elem in org:
      if elem['entity_group']=='NOUN':
        Noun.append(elem)
      if elem['entity_group']=='VERB':
        Verbs.append(elem)
      if elem['entity_group']=='ADJ':
        Adj.append(elem)
    return Noun,Verbs,Adj

  def trans(self,texto):
    input = self.tokenizer(texto,return_tensors="pt")
    tokens=self.tokeni(input)
    output  = self.model(**input)
    hidden=output[2]
    X=hidden[-1].detach().numpy()
    X=X.squeeze()
    return X,tokens

  
    
    self.add_general(prom,prom1)

  def add_hist(self,dis):
    self.hist_simil.append(dis)

  def add_histre(self,dis2):
    self.hist_real.append(dis2)
  
  def add_general(self,prom,prom1):
    self.agent_db[0]=prom
    self.agent_db[1]=prom1

  def stados(self,mat1,mat2):
    self.state[0]=mat1
    self.state[1]=mat2

    
  def mask(self,text1,text2):
    palabras=text1.split()
    num=random.randint(0,len(palabras))
    palabras[num]="[MASK]"
    linea=palabras[0]
    for i in range(1,len(palabras)):
      linea+=" "
      linea+=palabras[i]
    if linea.split()[-1]=="[MASK]":
      linea+="."
    texto=text2+"."+" "+linea
    res=self.unmasker(texto)[0]['sequence']
    print(res)
    oraciones=res.split(".")
    print(oraciones)
    resul=oraciones[-1]
    print(resul)
    code=0
    print(text1.lower())
    text1=" "+text1
    if resul.lower()==text1.lower():
        code=1
    return code

    
  def _get_reward(self,texto1,texto2, action):
        
    mat1=self.trans(texto1)
    mat2=self.trans(texto2)
    dis=self.calculate_cos_distance(mat1[0][0],mat2[0][0])
    self.stados(mat1[0][0],mat2[0][0])
  
    if action==0:
      re=dis
    else:
      re=1-dis
  
    res_ori = self.summarizer(texto1, max_length=25, min_length=5, do_sample=False)[0]['summary_text']
    res_nuev = self.summarizer(texto2, max_length=25, min_length=5, do_sample=False)[0]['summary_text']
    token_vecs_ori=self.trans(res_ori)
    token_vecs_nuev=self.trans(res_nuev)
    t_ori=self.token_esp(token_vecs_ori[1][0])
    t_nuev=self.token_esp(token_vecs_nuev[1][0])
    sep_ori=self.separar(res_ori)
    sep_nuev=self.separar(res_nuev)
  
    
    #Adjetivos:
    Ad_ori=self.checar(sep_ori[1][0]['word'],token_vecs_ori[1][0])
    Ad_nuev=self.checar(sep_nuev[1][0]['word'],token_vecs_nuev[1][0])
    vec_ori=self.vec(Ad_ori[1],token_vecs_ori[0])
    vec_nuev=self.vec(Ad_nuev[1],token_vecs_nuev[0])
    
  
    dis2=self.calculate_cos_distance(vec_ori,vec_nuev)
    if action==1:
      re2=1-dis2
    
      re=re+re2
      
    if action==2:
      re2=dis2

      re=re+re2

    self.add_hist(dis)
    prom=np.sum(self.hist_simil)/len(self.hist_simil)  
    self.add_histre(dis2)
    prom1=np.sum(self.hist_real)/len(self.hist_real)
    self.add_general(prom,prom1)

    re3=0.5-prom
    if (1-self.realidad[0])==0:
      re4=0.5-dis2
    if (1-self.realidad[0])==1:
      re4=dis2-0.5
      
    re5=self.mask(texto1,texto2)

    re=re+re3+re4+re5

    return re,state
  """
  @gin.configurable
  def general_agent(step=None,reward=None):
    assert(step!=None)
  """  
