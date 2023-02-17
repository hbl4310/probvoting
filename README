# Probabilistic voting schemes

Some code to play around with probabilistic voting schemes. 

`pip install -r requirements.txt` to install relevant packages in chosen environment. 

### Files
- "mev0.py" contains an implementation of the MEV0 voting scheme found in ['Probabilistic electoral methods, representative probability, and maximum entropy'](https://www.votingmatters.org.uk/ISSUE26/I26P3.pdf). 
- "pvoting.py" has a custom ensemble class based on sklearn's voting model ensemble. `ProbabilisticVotingClassifier` supports `voting` schemes "random", "randomseq" and "mev0" corresponding to "random dictator", "sequential random dictator" and "MEV0" in the aforementioned paper. 
    - Each base classifier 'votes' for the class labels for the data, which are aggregated per data-point using the chosen voting scheme.
- "data.py" has method to generate synthetic data.
- "compare.py" compares several ensemble methods against those probabilistic voting schemes mentioned, using a combination of KNN and SVM models as base classifiers. 
    - Data is synthetically generated. 
    - Models are compared based on repeated, stratified, k-fold cross-validation accuracy scores, and printed along with fitting times.
    - "compare.png" is generated graphing boxplots of these scores.

### Example
```
$ python compare.py
     knn-1: mean=0.870 std=0.032 time=10.0945s
     knn-3: mean=0.888 std=0.035 time=0.0469s
     knn-5: mean=0.894 std=0.030 time=0.0404s
     knn-7: mean=0.892 std=0.024 time=0.0443s
     knn-9: mean=0.890 std=0.025 time=0.0472s
   svm-rbf: mean=0.897 std=0.027 time=0.8007s
   svm-lin: mean=0.718 std=0.046 time=6.6017s
  svm-pol1: mean=0.722 std=0.042 time=0.6981s
  svm-pol2: mean=0.892 std=0.028 time=0.5655s
  svm-pol3: mean=0.837 std=0.031 time=0.8038s
  svm-pol4: mean=0.795 std=0.041 time=0.7691s
  svm-pol5: mean=0.728 std=0.037 time=0.9052s
  hardvote: mean=0.899 std=0.025 time=12.3797s
   bagging: mean=0.894 std=0.029 time=0.1481s
   rforest: mean=0.729 std=0.054 time=0.9283s
  adaboost: mean=0.860 std=0.055 time=16.0243s
 gradboost: mean=0.823 std=0.048 time=2.0012s
randomvote: mean=0.841 std=0.033 time=13.0710s
  mev0vote: mean=0.894 std=0.028 time=65.8598s
```
![comparison of various classifiers and ensemble models](compare.png "Comparison")