from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import json

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

pass

class FakeExtraHardEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.actions_grid=['query_return','query_query_step','record_return','record_step','stop']
    self.actions_kb=["no_relation","positive","negative"]
    self.noticias= pd.read_csv("data.csv").drop_duplicates() ##base de noticias
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
    self.env=self
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
    self.nlp1= pipeline('ner', grouped_entities=True)
    self.nlp1= pipeline('ner',model='vblagoje/bert-english-uncased-finetuned-pos', grouped_entities=True)

    

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



  def duckduck_(self,query,n=40,verbose=False):
    driver.get('https://duckduckgo.com/')
    search_box = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.NAME, "q")))
    search_box.send_keys(query)
    search_box.submit()

    elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//div[contains(@class,'result__body')]")))
    nxt_page=WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "btn--full")))
    n_=len(elements)
    while nxt_page and n_<n:
        nxt_page.click()
        try:
            nxt_page=WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "btn--full")))
            elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//div[contains(@class,'result__body')]")))
        except TimeoutException:
            nxt_page=None
            elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//div[contains(@class,'result__body')]")))
        n_=len(elements)

    snippets=[]
    for position,ele in enumerate(elements):
        html=ele.get_attribute("outerHTML")
        soup = BeautifulSoup(html,  "html.parser")
        #soup=BeautifulSoup(html, 'lxml')
        title=soup.find_all("h2", class_="result__title")[0]
        href=soup.find_all("a", class_="result__a")[0]
        snippet=soup.find_all("div", class_="result__snippet")[0]
        #print(html)
        if not href['href'].startswith("https://duckduckgo.com/y.js?"):
            snippets.append({"position":position,"title":href.text,"href":href["href"],"text":snippet.text})

    snippets=snippets[:n]

    return snippets

  

  def step(self, action,agent_db=None):
    """ ejecutamos el paso y regresamos el reward """
    """ debemos regresar un arreglo del tipo:
    respuesta:["estado","reward","booleando(si he termiando o no)"]"""
    self.num_steps_per_episode+=1
    action_grid,action_kb=self._expandAction(action)
    query=self._action_grid_selector(action_grid)
    reward=self._get_reward()
    self.query_seen[self.query_indexes[self.query_ind]]+=1
    query_seen=self.query_seen[self.query_indexes[self.query_ind]]
    stat=[]
    for i in range(0,768):
      stat.append=self.state[0].detach().numpy()[i]
    for i in range(0,768):
      stat.append=self.state[1].detach().numpy()[i]
    stat.append(self.agent_db[0])
    stat.append(self.agent_db[1])
    return [stat,reward,self.episode_continues,{}]




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

    self.episodeDF = pd.DataFrame.from_dict(duckduck_(self.noticias['Headline'][self.noticias_ind]))
    
    #set Query lists and indexes for queryDFs
    self.querys_step=list(set(self.episodeDF.search.tolist())) #Los querys
    self.query_indexes=list(range(len(self.querys_step))) #Los índices de la lista de querys
    self.row_index_query=[0]*len(self.querys_step) #En qué registro está cada query
    self.query_seen=[0]*len(self.querys_step) #Cuantos pasos he dado en ese query
    self.query_ind=0 #En qué query estoy

    self.querydf=self.episodeDF.loc[self.episodeDF["search"] == self.querys_step[self.query_indexes[self.query_ind]]]
    self.episode_continues=False

    return self.render()

  def render(self, mode='human'):

    #return self.querydf.iloc[self.row_index_query[self.query_indexes[self.query_ind]]].tolist(),self.agent_db
    query_seen=self.query_seen[self.query_indexes[self.query_ind]]
    #return [(query,self.num_steps_per_episode,query_seen,agent_db),reward,self.episode_continues]
    return [np.array([self.num_steps_per_episode,query_seen],dtype=np.float64)]

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
        return self.querydf.iloc[self.row_index_query[self.query_indexes[self.query_ind]]].tolist()
      return self.querydf.iloc[self.row_index_query[self.query_indexes[self.query_ind]]].tolist()

    


    

  def _query_query_step(self):
    if self.query_ind==len(self.querys_step)-1:
      self.query_ind=self.query_ind
    else:
      self.query_ind+=1
  def _query_return(self):
    if self.query_ind==0:
      self.query_ind=self.query_indexes[-1]
    else:
      self.query_ind-=1
  def _record_return(self):
    
    if self.row_index_query[self.query_indexes[self.query_ind]]==0:
      self.row_index_query[self.query_indexes[self.query_ind]]=len(self.querydf.index)-1
    else:
      self.row_index_query[self.query_indexes[self.query_ind]]-=1

  def _record_step(self):
    
    if self.row_index_query[self.query_indexes[self.query_ind]]==len(self.querydf.index)-1:
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
      a=tokenizer.convert_ids_to_tokens(s)
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
        anterior=tokenizer.convert_tokens_to_string(lista)
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
      res=token_esp(tokenized_text)
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
    cose= dot(a, b)/(norm(a)*norm(b))
    return 1 - cose

  def separar(self,texto):
    org=nlp1(texto)
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
    input = tokenizer(texto,return_tensors="pt")
    tokens=tokeni(input)
    output  = model(**input)
    hidden=output[2]
    X=hidden[-1].detach().numpy()
    X=X.squeeze()
    return X,tokens

  
    
    self.add_general(prom,prom1)

  def add_hist(self,dis):
    self.hist_simil.append(dis)

  def add_histre(self,dis2):
    self.hist_real.append(dis2)
  
  def add_general(prom,prom1):
    self.agent_db[0]=prom
    self.agent_db[1]=prom1

  def stados(mat1,mat2):
    self.state[0]=mat1
    self.state[1]=mat2

    
  def _get_reward(self,texto1,texto2, action):
    mat1=trans(texto1)
    mat2=trans(texto2)
    dis=calculate_cos_distance(mat1[0][0],mat2[0][0])
    stados(mat1[0][0],mat2[0][0])
  
    if action==0:
      re=dis
    else:
      re=1-dis
  
    res_ori = summarizer(texto1, max_length=25, min_length=5, do_sample=False)[0]['summary_text']
    res_nuev = summarizer(texto2, max_length=25, min_length=5, do_sample=False)[0]['summary_text']
    token_vecs_ori=trans(res_ori)
    token_vecs_nuev=trans(res_nuev)
    t_ori=token_esp(token_vecs_ori[1][0])
    t_nuev=token_esp(token_vecs_nuev[1][0])
    sep_ori=separar(res_ori)
    sep_nuev=separar(res_nuev)
  
    
    #Adjetivos:
    Ad_ori=checar(sep_ori[1][0]['word'],token_vecs_ori[1][0])
    Ad_nuev=checar(sep_nuev[1][0]['word'],token_vecs_nuev[1][0])
    vec_ori=vec(Ad_ori[1],token_vecs_ori[0])
    vec_nuev=vec(Ad_nuev[1],token_vecs_nuev[0])
    
  
    dis2=calculate_cos_distance(vec_ori,vec_nuev)
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
    if (1-realidad[0])==0:
      re4=0.5-dis2
    if (1-realidad[0])==1:
      re4=dis2-0.5

    re=re+re3+re4

    return re,state
  """
  @gin.configurable
  def general_agent(step=None,reward=None):
    assert(step!=None)
  """  
