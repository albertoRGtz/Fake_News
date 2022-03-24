import random
import click
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup
import json
import pandas as pd

save_path = r'./searches/'
div_tag = ''
div_class_tag = ''
div_next_tag = ''
division_tag = ''
options = webdriver.FirefoxOptions()
options.add_argument('-headless')
driver = None
# El driver se inicializa en initDriver() para evitar cargarlo de forma
# innecesaria.
# ¡IMPORTANTE!
# En initDriver(), asegurarse que este cargando el geckodriver correspondiente
# El geckodriver dentro del directorio win64 es para windows 64 mientras que el
# otro es para linux.


@click.group()
def cli():
    """Websearch keywords in different search engines"""
    pass

# Datos


def exportar(results, name, path=save_path):
    with open(name+'.json', 'w') as file:
        json.dump(results, file, indent=4)
    
    dataf = pd.DataFrame.from_dict(results)
    dataf.to_csv(path+name+'.csv', index=False)


@cli.command(help='The number of items to process.')
def engines():
    print("duckduck")
    print("scholar")
    print("bing")
    print("researchgate")


@cli.command(help='Collect all search enginees')
@click.option('-n', default=20, help='number of results')
@click.option('--verbose', type=bool, default=False, help='Verbise')
@click.argument('query')
@click.argument('name')
def collect(query, name, n=20, verbose=False):
    results = {}
    print("scholar")
    results["scholar"] = {}
    snippets = scholar_(query, name, n, verbose)
    results["scholar"]['results'] = snippets
    results["scholar"]['total'] = len(snippets)
    print("duckduck")
    results["duckduck"] = {}
    snippets = duckduck_(query, name, n, verbose)
    results["duckduck"]['results'] = snippets
    results["duckduck"]['total'] = len(snippets)
    print("bing")
    results["bing"] = {}
    snippets = bing_(query, name, n, verbose)
    results["bing"]['results'] = snippets
    results["bing"]['total'] = len(snippets)
    print("yahoo")
    results["yahoo"] = {}
    snippets=yahoo_(query, name, n, verbose)
    results["yahoo"]['results'] = snippets
    results["yahoo"]['total'] = len(snippets)
    print("researchgate")
    results["researchgate"] = {}
    snippets=researchgate_(query, name, n, verbose)
    results["researchgate"]['results'] = snippets
    results["researchgate"]['total'] = len(snippets)
    print(results)
    with open(name+'.json', 'w') as file:
        json.dump(results, file, indent=4)


def create_dataset(n=10):
    titles = pd.read_csv("./data/FakeCovid_July2020.csv")['title'].tolist()
    results = {}
    for i in range(n):
    	snippets = duckduck_(titles[i], "./data/FCovDataSet", n=20)
    	results[titles[i]] = {}
    	results[titles[i]]["duckduck"] = {}
    	results[titles[i]]["duckduck"] = snippets
    with open('FCovDataSet.json', 'w') as file:
        json.dump(results, file, indent=4)


def test_search(query, name, n=10, verbose=False):
    words = query.split()
    random.seed()
    for i in range(len(words) // 2):
        words.pop(random.randint(0, len(words)-1))
    query = ' '.join(words)
    return duckduck_(query, name, n)


def scholar_(query, name, n=20, verbose=False):
    div_tag = "//div[contains(@class,'gs_ri')]"
    div_class_tag= "gs_rs"
    div_next_tag = "//div[@id='gs_n']//table//tr//td[last()]//a"
    driver = initDriver()
    division_tag = "div"
    driver.get('https://scholar.google.com.mx/')
    search_box = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "q")))
    search_box.send_keys(query)
    search_box.submit()
    snippets = collectSnippets(n, division_tag, div_tag, div_class_tag, div_next_tag, driver)
    closeDriver(driver)
    exportar(snippets, name)
    return snippets


def bing_(query, name, n=20, verbose=False):
    driver = initDriver()
    division_tag = "div"
    div_tag = "//li[contains(@class,'b_algo')]"
    div_class_tag = "b_caption"
    div_next_tag = "//a[contains(@class,'sb_pagN sb_pagN_bp b_widePag sb_bp')]"
    driver.get('https://www.bing.com/?setlang=es')
    search_box = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "q")))
    search_box.send_keys(query)
    search_box.submit()
    snippets = collectSnippets(n, division_tag, div_tag, div_class_tag, div_next_tag, driver)
    closeDriver(driver)
    exportar(snippets, name)
    return snippets


@cli.command(help='Search on the google scholar enginee')
@click.option('-n', default=20, help='number of results ')
@click.option('--verbose', type=bool, default=False, help='Verbise')
@click.argument('query')
@click.argument('name')
def bing(query, name, n=20, verbose=False):
    bing_(query, name, n, verbose)


@cli.command(help='Search on the google scholar enginee')
@click.option('-n', default=20, help='number of results ')
@click.option('--verbose', type=bool, default=False, help='Verbise')
@click.argument('query')
@click.argument('name')
def scholar(query, name, n=20, verbose=False):
    scholar_(query, name, n, verbose)


def yahoo_(query, name, n=20, verbose=False):
    driver = initDriver()
    division_tag = "p"
    div_tag = "//div[contains(@class,'dd NewsArticle')]"
    div_class_tag = "s-desc"
    div_next_tag = "//a[contains(@class,'next')]"
    driver.get('https://news.yahoo.com/')
    search_box = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "p")))
    search_box.send_keys(query)
    search_box.submit()
    snippets = collectSnippets(n, division_tag, div_tag, div_class_tag, div_next_tag, driver)
    closeDriver(driver)
    exportar(snippets, name)
    return snippets


@cli.command(help='Search on the google scholar enginee')
@click.option('-n', default=20, help='number of results ')
@click.option('--verbose', type=bool, default=False, help='Verbise')
@click.argument('query')
@click.argument('name')
def yahoo(query, name, n=20, verbose=False):
    yahoo_(query, name, n, verbose)


def researchgate_(query, name, n=20, verbose=False):
    driver.get('https://www.researchgate.net/')
    search_box = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
        (By.CLASS_NAME,"index-search-field__input")))
    search_box.send_keys(query)
    search_box = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
        (By.CLASS_NAME,"nova-legacy-e-icon.nova-legacy-e-icon--size-s.nova-"
                       "legacy-e-icon--theme-bare.nova-legacy-e-icon--color-"
                       "grey.nova-legacy-e-icon--luminosity-medium")))
    search_box.click()
    elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located(
        (By.XPATH, "//div[contains(@class,'nova-legacy-o-stack__item')]")))
    n_ = len(elements)
    snippets = []
    position = 0
    while n_ < n:
        for ele in elements:
            html = ele.get_attribute("outerHTML")
            soup = BeautifulSoup(html,  "html.parser")
            # soup=BeautifulSoup(html, 'lxml')
            a = soup.find_all("a")[0]
            title = a.text
            href = a['href']
            snippet = soup.find_all("div", class_="nova-legacy-o-stack__item")[0]
            snippets.append((position, title, href, snippet.text))
            if position == n-1:
                break
            position += 1
        try:
            search_box = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
                (By.CLASS_NAME,"nova-legacy-c-button.nova-legacy-c-button--"
                               "align-center.nova-legacy-c-button--radius-m."
                               "nova-legacy-c-button--size-s.nova-legacy-c-"
                               "button--color-grey.nova-legacy-c-button--theme-"
                               "bare.nova-legacy-c-button--width-full")))
            href = next_page.get_attribute('href')
            driver.get(href)
            elements = WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located(
                (By.XPATH, "//div[contains(@class,'nova-legacy-o-stack__item')]")))
            n_ += len(elements)
        except TimeoutException as e:
            break

    for ele in elements:
        html = ele.get_attribute("outerHTML")
        soup = BeautifulSoup(html,  "html.parser")
        # soup=BeautifulSoup(html, 'lxml')
        a = soup.find_all("a")[0]
        title = a.text
        href = a['href']
        try:
            snippet = soup.find_all("div", class_="gs_rs")[0].text
        except IndexError:
            snippet = ""
        snippets.append({"position": position, "title": title, "href": href,
                         "text": snippet})
        if position == n-1:
            break
        position += 1
    # print(snippets,query,n)
    exportar(snippets, name)
    return snippets


@cli.command(help='Search on the researchgate engine')
@click.option('-n', default=20, help='number of results ')
@click.option('--verbose', type=bool, default=False, help='Verbise')
@click.argument('query')
@click.argument('name')
def researchgate(query,name,n=20,verbose=False):
    researchgate_(query,name,n,verbose)

def duckduck_(query,name,n=20,verbose=False):
    div_tag = "//div[contains(@class,'result__body links_main links_deep')]"
    division_tag = "div"
    div_class_tag = "result__snippet js-result-snippet"
    div_next_tag = "//a[contains(@class,'result--more__btn btn btn--full')]"
    driver = initDriver()
    driver.get('https://duckduckgo.com/')
    search_box = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.NAME, "q")))
    search_box.send_keys(query)
    search_box.submit()
    snippets = collectSnippets(n, division_tag, div_tag, div_class_tag, div_next_tag, driver)
    closeDriver(driver)
    #exportar(snippets, name)
    return snippets



@cli.command(help='Search on the duckduckgo engine')
@click.option('-n', default=20, help='number of results ')
@click.option('--verbose', type=bool, default=False, help='Verbise')
@click.argument('query')
@click.argument('name')
def duckduck(query,name,n=20,verbose=False):
    duckduck_(query,name,n,verbose)
  
def base_(archivo,buscador,name,n=20):
    extension = os.path.splitext(archivo)
    if extension=='.json':
      with open(archivo) as file:
    	   data = json.load(file)
    if extension=='.csv':
      data= pd.read_csv (archivo) 
    if extension=='.xlsx':
      data= pd.read_excel (archivo) 
    base={}
    cont=0
    for i in range(0,len(data)):
        base[cont]={}
        query=data['query'][i]
        base[cont]['query']=query
        snippets=buscador_(query,n,verbose)
        results[cont]['results']=snippets
        cont+=1
    with open(name+'.json', 'w') as file:
        json.dump(base, file, indent=4)
   
    return base
   
@cli.command(help='Search on the data engine')
@click.option('-n', default=20, help='number of results ')
@click.argument('query')
@click.argument('buscador')
@click.argument('name')
def basea(archivo,buscador,name,n=20):
    base_(archivo,buscador,name,n=20)     
    
# Inicializa el driver
def initDriver():
    #driver = webdriver.Firefox(executable_path=r'./win64/geckodriver', options=options)
    driver = webdriver.Firefox(executable_path=r'./gym_fake/envs/geckodriver', options=options)
    return driver

# Recolecta n snippets uitlizando las variable "tag" y el driver
# Regresa una lista de listas que contiene la informacion de los snippets 
def collectSnippets(n, division_tag ,div_tag, div_class_tag, div_next_page_tag, driver):
    collected_elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, div_tag)))
    n_=len(collected_elements)
    snippets=[]
    position=0
    while n_<n:
        for ele in collected_elements:
            html=ele.get_attribute("outerHTML")
            #soup = BeautifulSoup(html,  "html.parser")
            soup=BeautifulSoup(html, "lxml")
            a=soup.find_all("a")[0]
            title=a.text
            href=a['href']
            snippet=soup.find_all(division_tag, class_=div_class_tag)[0]
            snippets.append({"position":position,"title":title,"href":href,"text":snippet.text})
            if position==n-1:
                break
            position+=1
           
        try:
            next_page = driver.find_element_by_xpath(div_next_page_tag)
            href=next_page.get_attribute('href')
            driver.get(href)
            collected_elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, div_tag)))
            n_+=len(collected_elements)
        except TimeoutException as e:
            break
        except NoSuchElementException as e:
            print("No encontre mas elementos")
            break

    for ele in collected_elements:
        html=ele.get_attribute("outerHTML")
        soup = BeautifulSoup(html,  "html.parser")
        #soup=BeautifulSoup(html, 'lxml')
        a=soup.find_all("a")[0]
        title=a.text
        href=a['href']
        try:
            snippet=soup.find_all(division_tag, class_=div_class_tag)[0].text
        except IndexError:
            snippet=""
        snippets.append({"position":position,"title":title,"href":href,"text":snippet})
        if position==n-1:
                break
        position+=1
    snippets = checkOutput(snippets)
    return snippets

def checkOutput(output):
    checkedOutput = []
    # Revisa si el href se encuentra dentro del texto del snippet
    # de ser asi lo elimina del texto.
    for item in output:
        href = item['href']
        text = item['text']
        if(href.endswith('/')):
            href = href[0:-1]
        if(text.startswith(href)):
            item['text'] = text.replace(href, "")
        href = href.replace("http://", "")
        if(text.startswith(href)):
            item['text'] = text.replace(href, "")
        checkedOutput.append(item)
    return checkedOutput

def closeDriver(driver):
    driver.close()


if __name__ == '__main__':
    cli()
    driver.close()

