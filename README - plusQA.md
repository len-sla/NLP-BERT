
## Inference using BERT like models from HuggingFace transformers.pipelines.question_answering collection dockerised and exposed via RESTful API

It is a bit long title but is presenting present way of automatising, making efficient, ready to use environment almost everywhere( where docker is present).
In this particular case docker is installed on Ubuntu18.04 local machine without GPU to test dockerised ML/NLP environments project etc.
The only limiting factor is is the speed  of internet connection and inference time as it is done  locally on CPU based machine. Of course one could go further pay for 
time of computation power of the cloud and execute dockerised environments in the cloud. Then it is also another step of fintunig ready to use models
in this case question_answering from HuggingFace transformers.pipelines.whole procedure is nicelly described by Jarek Szczegielniak and presented in article (https://www.codeproject.com/Articles/5302894/Exposing-Dockerized-AI-Models-via-RESTful-API) 

There couple of things one need to take care when installing  Docker on  Ubuntu machines( details under this address):
(https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04).

To be able to use dockerised environments for non root users in Ubuntu  threre is need to do couple of ajustments. 
Very extensive reasearch on the subject and helpful tips are  on the (https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue#comment101604579_48957722)
especially part presenting solution of Olshansk user( a least for me)
Once done all this you could see :



## Status
Project is: _in progress_, 

### Inspiration

 Project inspired by
 [Hugging Face](https://huggingface.co/)
 &&
 [fast.ai ](https://www.fast.ai/)


### Info
Created by [lencz.sla@gmail.com]
