# NMTfor_BD_Regional_languages_RUCSE
This  repository contains our program and dataset used in our work on NMT for Bangladesh Regional languages

-----------------------------------------------------------------------------------------------------
# To run the system following module should be installed:
  1) Tensorflow
  2) Python
  3) Jupyter Notebook(optional)

To use the user interface one should run:

              python3 app.py
              
for app.py file contains in the folder "NMT_System_Interface_Programs" and a port number will be showed on the terminal you should follow the http on your browser to use system interface.

# Dataset:
  In dataset folder Bangla-Chakma and Bangla-Chittagong sentence pairs are provided and the code that is used for preprocessing the datasets. The main file that contains all Bangla-Chakma and Bangla-Chittagonian is:
  1) bangla-chakma.csv -> Bangla and Chakma parallel sentence pairs.
  2) all_bangla_chittagoina.txt -> Bangal and Chittagoinan parallel sentence pairs.

# Code:
The code folder contains Jupyter Notebook files (.ipynb) that are used to build our models. The "TL" (Transfer Learning) notebook file is to show he model's performance for transfer learning procedure. In this experiment, we froze the weights of the upper layers of a pre-trained transformer model, which had been trained on an English-Bangla dataset. We then exclusively trained the linear layer of the transformer architecture using our Bangla-Chakma or Bangla-Chittagonian dataset. 

The "others" file, consisting of "Bangla-Chakma" and "Bangla-Chittagonian", serves the purpose of both model development and performance assessment. Within folder, the "en-to-bn.h5" file represents a saved model that has undergone training on an English-Bangla dataset. In the "Text_to_Text_Bangla_Chakma_TL" file, the previously mentioned model is employed to measuere the performance of Transfer learning.
