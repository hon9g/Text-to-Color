# Emotion aware conversational interface - Text to Color
This was supported by [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/) which was organized by [TensorFlow Korea User Group](https://facebook.com/groups/TensorFlowKR/).


    This is a text-to-color demo code.
    To obtain more face-to-face information on the conversational interface,
    It designed as recognizing user's emotion from texts and displaying it in colors. 



<img src="https://github.com/hon9g/Text-to-Color/blob/main/docs/image/1.gif"/>

- I ran this webpage on Google Cloud Platform during the final presentation of Jeju DL Camp 2018. I used this [presentation slide](https://docs.google.com/presentation/d/1KhDNmQDuvWiQ2-A3REiayisGM00CDjCrDB54-qLaIUw/edit?usp=sharing/).
- Now I run [webpage on heroku](https://text-to-color.herokuapp.com/).



## Code Overview

- **deepmoji/**
  contains underlying codes to use deepmoji model.
- **models/**
  contains pretrained model and vocabulary.
- **templates/**
  contains HTML files for the Text to Color demo web page.
- **app.py/**
  main file to run the Text to Color Demo web page.

  
## Dependencies

- Python 3.x
- Emoji 0.5
- Flask 0.12
- Requests 2.14.2
- H5py 2.7.0
- Text-unidecode 1.2
- Keras 2.1.2

I ran this code on
-  Tensorflow (cpu-only) 1.8.0
<br/>and 
- Tensorflow-gpu 1.4.0 & CUDA Toolkit 8.0 & CuDNN v6.0

## How to run

1. Git clone.
2. Run app.py.
3. then you can see message "* Running on http://localhost:5000/ (Press CTRL+C to quit)". access to “http://localhost:5000” on your browser.
4. put the sentence and test it.

## How it works
<img src="https://github.com/hon9g/Text-to-Color/blob/main/docs/image/2.png" width="600" />
The text is classified into emojis(I use it as emotional labels) and emojis are mapped to colors.

### Text to Emoji

I use the DeepMoji model from MIT media lab as emotion classifier.<br/>
It is trained by 1246 million tweets, which is containing one of 64 different common emoticon.<br/>

<img src="https://github.com/hon9g/Text-to-Color/blob/main/docs/image/3.png" width="500" />

There are embedding layer to project each word into a vector space. <br/>
( a hyper tangent activation enforce a constraint of each embedding dimension being within -1~1. )<br/>
two bidirectional LSTM layers to capture the context of each word.<br/>
And an attention layer that lets the model decide the importance of each word for the prediction.<br/>


### Emoji to Color

The color code I use is **rgba**. (a = defines the opacity.)<br/>

<img src="https://github.com/hon9g/Text-to-Color/blob/main/docs/image/4.png"/>

I mapping color(rgb) based on dendrogram, which shows how the model learns to group emojis based on emotional content.<br/> 
The **y-axis is the distance on the correlation matrix** of the model’s predictions. It measured using average linkage.<br/>


<img src="https://github.com/hon9g/Text-to-Color/blob/main/docs/image/array.png" />
The output from the model is the probability of each 64 different emojis.<br/>
I use top 3 probability with normalization for define the opacity of the layers.<br/>
And these 3 layers are overlapped, and then determine the color of the screen.<br/>
<img src="https://github.com/hon9g/Text-to-Color/blob/main/docs/image/5.png" width="500" />

## Citation
```
@inproceedings{felbo2017,
  title={Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm},
  author={Felbo, Bjarke and Mislove, Alan and S{\o}gaard, Anders and Rahwan, Iyad and Lehmann, Sune},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2017}
}
```
