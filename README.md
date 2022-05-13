# HockeyDataProject

## About The Project
In this Project we are using a hockey dataset off of Kaggle to predict how far a team makes it in the finals bracket based off their season stats going in. More info on the breakdown of what was measured included in ProjectProposal.ipynb. 

Our class labels included:
- 0: not included in bracket
- 1: lost in round one
- 2: lost in round two
- 3: lost conference finals
- 4: lost in the cup
- 5: won the finals

### For reference for People Who Don't Watch Hockey
Here is a sample playoff bracket with labels 
<img src="https://i.redd.it/we47inh7o6g51.png" style="width:500px;"/>

## Project Structure
### Initial Plans 
Our initial thoughts as well as the breakdown of the data can be seen in our ProjectProposal.ipynb. Please note that, as mentioned above, we later joined a new attribute "FINISH" to the dataset, which became our new clssification label. This attribute tracks how far a team made it into the finals bracket.

### Data Analysis
Prior to training our ML models, we created the EDA.ipynb file which contains various data visualizations to help guide our approach towards our data. We attempted to find correlations between attributes in the dataset which could help us build a more accurate classifier.

### Classification
The results of our classifiers are in ClassificationResults.ipynb. The classifiers we used include a descision tree using entropy based splitting on F random attributes, naive bayes, KNN, and a random forest of the previously mentioned descision trees. We also ran the data through an apriori algorithm to generate any possible rules for additional insights.

### Final Analysis
In our TechnicalReport.ipynb, we recap our EDA and what it meant to our classfication results and explore how they could be improved in the future. 

