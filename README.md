
# Accident classification and prediction
## Introduction

In this project, a classification application is developed to caterorize accident reports published by National Transportation Safety Board (NTSB). Without reading those reports, Natural Language Processing (NLP) is able to extract informative features through some basic NLP tecnniques, such as tokenization, lemmatizing, and compound splitting. The extracted features are integrated to train a classification algorithm for topic modelling. The developed classfier can be used to predit the topic of additional documents. 

## Application development
### Environment setup
The application is developed using Python 3.7. The following packages are required and should be installed
* pip install -U scikit-learn
* pip install numpy
* pip install tika
* pip install glob
* pip insall Pillow
* pip install pandas
* pip install seaborn
* pip install regex
* pip install matplotlib

### Functions
* create_widgets()-------- create buttons to trigger model training, prediction
* create_corpus()--------- Create a list of strings and each string represents a PDF document
* clean()----------------- Apply tokenization, lemmatization, punctuation/stopwords prunning, digits removal
* train_lda_fun()--------- Train LDA model using the accidents reports in /training
* train_nmf_fun()--------- Train NMF model using the accidents reports in /training
* visualize_topics()------ Plot the 2-D visualization of identified topics for all reports
* predict_topics()-------- Predict the topic of a selected PDF document through GUI
* plot_topic_hist()------- Plot the word distribution of each topic

### Data source
The accident reports are download from [NTSB website](https://www.ntsb.gov/investigations/AccidentReports/Pages/AccidentReports.aspx). 47 accident reports (about 10 reports/topic) are selected to train the LDA and NMF topic models. Additional 2 reports for each topic are used to demonstrate the model predictive capability.

## Application implementation







