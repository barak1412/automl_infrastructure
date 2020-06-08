# AutoML Infrastructure
## Library Goal
The automl_infrastructure library's main goal is supplying
wide and easy-to-use infrastructure for classifiers modeling
 and evaluation, including inner cross-validated hyper-param 
 optimization.
 
 ## Main Features
 * Inner cross-validation process using repeated k-fold.
 * Optional hyper-params optimization process.
 * Complex modeling creation (e.g. blending several weak classifiers).
 * Experiment definition with final report that contains:
    * Experiment's information (start time, end time and ect')
    * For each model:
        * Observations (metrics) summary for every class.
        * Visualizations (e.g ConfusionMatrix).
 
 Note that every Visualization/Observation/Metric may be implemented by the user. 

Make sure to look at the [docs](/docs/sphinx/build/html/index.html).
