# Clustering models.
# Customers fraud predictions: Machine Learning Classes.
<p align="center"><img src = "https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg">

Here we have the codes for Spark's Scala API probabilistic clustering methods based on the MLLib Library (Dataframe) API.

On the supervised learning branch for statistical modeling there's always a label feature (response variable) which split the observations into two different groups; this fact provides the building block to the whole methodologies for classification problems. Our main objective in this repository is propose an additional way to response the need of prediction for the customer's fraud class probability, but in this time considering unsupervised methods, **probabilistic clustering methods** particurally.
The key concepts to grab the essence of this API are:

* **Calculate the "optimal" number of groups to distribute the customers who have been previously affected by a fraud circunstance. Those groups are interpreted as the _fraud clusters_.\
Regarding the _nature_ of the features involved, the question about _how many group should I use?_ would have a straightforward solution. The less variability (more zeros or null values) the features have the harder the method fits well.\
Opposite to the supervised learning methods, the procedure here works in the way that reported fraud cases are the only data to construct the cluster structure. There's no need to include non-fraud cases.**

* **After a conscientious analysis for the number of groups to the estimation procedure there's an _object_, called model, whose roll is to _transform_ the data set containing the features of non-affected customers. The summary for this _transformation_ is the addition of new fields, probability and prediction. This information is interpreted as the cluster label and probability each customer belongs to it. The calculation of the mentioned probability is based on the [_Gaussian mixture model_](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm). Understanding the mathematical background of this methods is mandatory to judge how well this approach could be a solution to the fraud prediction problem.**

Be sure you get these two points before moving on. I encourage you to make an effort to completely understand how this approach differs to the supervised learning way.\
The following preserves the way **_Classification_models_Scala_API_** was presented.

### 1. Modules (modules_fraud_model.scala): :books:
  Import the Scala classes needed to transform data and estimate probabilistic clustering method. Here comes the **_Gaussian mixture model_** class. The estimation procedure uses the time-honored **_Expectation-Maximization_** algorithm.
  
### 2. Data set (data_set_fraud_model.scala): :floppy_disk:
  Contains the sql sentence to call data allocated in HDFS. This data contains the features for the two types of customers tailored as an ETL written in Cloudera Impala.
  
### 3. Fraud model (fraud_model.scala): :space_invader:
  The final implementation where the API's **_kernel_** dwells. The user can call the _Gaussian mixture_ method in the usual _object-oriented_ form previous **Modules** and **Data set** call. Inside the method there's a routine performed by the pipeline model; the stages are: _vector assembler_, _minmax scaler_ (it's possible to invoke others), _the machine learning method_ and the _pipeline_ itself. Moreover, there's a writting of the results in form of a Hive table.
  
The last file **_Execution.scala_** contains just the code lines to execute on the **_Spark shell_**.

######  **_Considerations_**:
###### 1. The current way to access Spark is through **_SFTP_** connection. **MobaXterm** is an alternative to doing so, but it has no support, indeed, is has IT restrictions, however, it's our only tool.
###### 2. The source codes in this repository can not be executed inside the GitHub platform.
###### 3. The updates published here are for the good of the version control. The new versions themselves don't migrate directly to the Landing Zone. The user has to copy these new versions into the node using, e.g., WinSPC or FileZilla.
