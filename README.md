## Project Name
>English to german translator local/server using T5 model.



### General info
If you need to  translate  text from english  to german/french 
you could use one of many online translator services. They are convenient and quick.
If you are a company/private and don't want to share, sometimes confident information having own local translator could be better solution.
With a bit more than a 3 lines of code old machine all that is possible. Key word is 'transformers'.
Transformer models have taken the world of natural language processing (NLP) by storm and Hugging Face is a company
which empowered everybody with powerfull tools: open-source libraries, and pretrained models( among them [T5](https://huggingface.co/transformers/v2.7.0/model_doc/t5.html#tft5model)).
There actually more line of code taking care about cosmetic of two used widgets compared to translation.

#### _comparing local and online translator result_ 
![### comparing local and online translator result ](en-ge-t5.JPG)

### There two extra notebooks to make use of T5 and MarianMT model for translation of text files

* [First file:](Part_A_files_preprocess_NTLK_spliting_by_sentence.ipynb) is responsible for text file preprocessing.
* [Second file:](Part_B_files_text_translators_eng_to_de_pl.ipynb) makes text file translation.

![### Using Ipython forms ](file_trans.JPG)

#### _translation to spanish_
![### Using Ipython forms ](text_es.JPG)
### Libraries
 
* Hugging Face transformers [T5](https://huggingface.co/transformers/v2.7.0/model_doc/t5.html#tft5model).
 > model = AutoModelWithLMHead.from_pretrained("t5-base")
 
 I used relativelly moderate base T5 model(850.8MB) as there are also models: large(2.7GB), t5-3b(10.6GB) and t5-11b(42.1GB) for much better accuracy
 
* [torch](https://pytorch.org/)
* [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/)

### Setup

Pytorch
```
conda install -c pytorch pytorch

```
* [ipywidgets installation if not by default in Conda](https://ipywidgets.readthedocs.io/en/stable/user_install.html)

Hugging Face transformers

```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```
Voilà

```
!pip install voila
!jupyter serverextension enable voila —sys-prefix
```
Voilà runs Jupyter notebooks just like the Jupyter notebook server you are using now does, but it also does something very important: it removes all of the cell inputs, and only shows output (including ipywidgets), along with your markdown cells. So what's left is a web application!  You will see the same content as your notebook, but without any of the code cells.

## Code Examples

Examples of usage as given on picture:
```
 voila translate_en_to_ge.ipynb
```

Translator because of resources has set limit max_length=400 to translate at once.
Preparing simple loop and dividing text on sentences using ready NTLK  or re sentence tokeniser or other make it possible to translate text as long as you wish.

``` 
from nltk.tokenize import RegexpTokenizer, sent_tokenize


with open('text_to-translate.txt', 'r') as in_file:
    text = in_file.read()
    sents = nltk.sent_tokenize(text)
``` 

now sents is a table of sentences so you could  give couple sentences at once to transaltor to process and append result from the translator pipeline to another table and then save as translated text

## Status
Project is: _in progress_, 

### Inspiration

 Project inspired by
 [Hugging Face](https://huggingface.co/)
 &&
 [fast.ai ](https://www.fast.ai/)


### Info
Created by [@len-sla]