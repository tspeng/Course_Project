
# Accident classification and prediction
## Introduction

In this project, a topic modeling and prediction application is developed to cluster accident reports published by National Transportation Safety Board (NTSB). Without reading those reports, Natural Language Processing (NLP) is able to extract informative features through some basic NLP tecnniques, such as tokenization, lemmatization, and topword/punctuation prunning. The extracted features are integrated to train machine learning algorithms to identify topics embeded in the accident reports. The trained topic model can be used to predict the topic of additional documents. 

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
* create_widgets()--------Create buttons to trigger model training, prediction
* create_corpus()---------Create a list of strings and each string represents a PDF document
* clean()------------------Apply tokenization, lemmatization, punctuation/stopwords prunning, digits removal
* train_lda_fun()----------Train LDA model using the accident reports in [training data](https://github.com/tspeng/Course_Project/blob/master/training)
* train_nmf_fun()----------Train NMF model using the accident reports in [training data](https://github.com/tspeng/Course_Project/blob/master/training)
* visualize_topics()--------Plot the 2-D visualization of identified topics for all reports
* predict_topics()---------Predict the topic of a selected PDF document in [test data](https://github.com/tspeng/Course_Project/blob/master/test) through GUI
* plot_topic_hist()---------Plot the word distribution of each topic

### Data source
The accident reports are download from [NTSB website](https://www.ntsb.gov/investigations/AccidentReports/Pages/AccidentReports.aspx). 52 accident reports (about 10 reports/topic) are selected to train the LDA and NMF topic models. Additional 2 reports for each topic are used to demonstrate the model's predictive capability.

## Application implementation
In order to facilitate implementation of the developed tool, a Graphical User Interface (GUI) was developed using the python built-in tkinter package. The following features are included in the application user interface:

* Train LDA or NMF topic models
* Project the clustered accident reports in a 2-D graph for visualization
* Predict the report category by loading in another PDF report
* Plot the histogram of word distribution for each topic

The developed GUI is shown below. Please find details regarding interacting with the GUI in the [presentation](https://github.com/tspeng/Course_Project/blob/master/Course_Project_Presentation.pdf).

![GUI](https://github.com/tspeng/Course_Project/blob/master/GUI.png)

## References
* [https://github.com/rdspring1/LSH_LDA/blob/master/lda.py](https://github.com/rdspring1/LSH_LDA/blob/master/lda.py)
* [https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html](https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html)
* [https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)
* [https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html)
* [http://effbot.org/tkinterbook/tkinter-index.htm](http://effbot.org/tkinterbook/tkinter-index.htm)
* [https://docs.python.org/2/library/tkinter.html](https://docs.python.org/2/library/tkinter.html)
* [https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python](https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python)


