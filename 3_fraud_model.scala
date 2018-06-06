/*Using the object-oriented capabilities available in the Scala language the next object (gmm_model) performs all calculations that
responds the question: is possible that a probabilistic clustering model predicts the fraud event?.
The gmm_model object contains 4 simple methods, each one workm as te following:
	1. The call_data uses the previous load of the SQL sentence in 2.data_set_model.scala to create an object of
	   org.apache.spark.sql.DataFrame type that brings the feature data set.
	2. Being quite original a lambda method is defined to perform a quiality test for the features imported. It checks the amount
	   of zeros each feature has, this is a useful information to decide whichs features would selected as the input to the
	   EM algorithm.
	3. The main method, the kernel, gmm (Gaussian mixture moddel) performs the usual steps fitting an analytical method within the
	   pipeline stages framework. Assembler, scaler are input data transformations required, then the pipeline gathers them beside
	   the mixture and does the magic taking the estimationData set. Finally the model object shows up and makes the transformation
	   of the prediction data set (estimationData_model).
	   Before the final result is written on HDFS a function to calculate the maximum probability is applied to create a new column
	   "maximum_probability", as its name suggests, returns the max value of the probability vector. These are the cluster
	   probabilities.
	   In the final part the API drops and create tables of results, these are quite similar to those in the supervised methods.*/
object gmm_model {
def call_data(sql: String) : org.apache.spark.sql.DataFrame = {
val data = spark.sql(sql)
return data.cache()
}
def lambda(dataframe : org.apache.spark.sql.DataFrame, feature : String) : Double = {
return (dataframe.filter(dataframe(feature) === 0).count().toDouble / dataframe.count().toDouble) * 100
}
def view_features(dataframe : org.apache.spark.sql.DataFrame) : org.apache.spark.sql.DataFrame = {
val features = dataframe.drop("documento", "documento_virtuales", "indicador").columns
val percentage_zeros = features.map(x => gmm_model.lambda(dataframe, x))
val results = sc.parallelize(features zip percentage_zeros).toDF("features", "percentage_zeros")
return results.orderBy(desc("percentage_zeros"))
}
def gmm(dataframe : org.apache.spark.sql.DataFrame, features: Array[String], K:Int) : Unit = {
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
val gmm = new GaussianMixture().setK(K).setFeaturesCol("scaledFeatures")
val pipeline = new Pipeline().setStages(Array(assembler, scaler, gmm))
val estimationData = dataframe.filter(dataframe("indicador") === "estimación")
estimationData.cache()
val forecastData = dataframe.filter(dataframe("indicador") === "pronóstico")
forecastData.cache()
val model = pipeline.fit(estimationData)
val estimationData_model = model.transform(estimationData)
estimationData_model.cache()
val maximum_probability_udf = udf((v: Vector) => v.toArray.max)
estimationData_model.withColumn("maximum_probability", maximum_probability_udf($"probability")).createTempView("estimationData_gmm")
spark.sql("""drop table if exists proceso_seguridad_externa.gmm_preferencial_estimacion purge""")
spark.sql("""create table if not exists proceso_seguridad_externa.gmm_preferencial_estimacion stored as parquet as 
			 select documento, indicador, prediction as agrupamiento, maximum_probability as probabilidad
			 		from estimationData_gmm""")
val forecastData_model = model.transform(forecastData)
forecastData_model.cache()
forecastData_model.withColumn("maximum_probability", maximum_probability_udf($"probability")).createTempView("forecastData_gmm")
spark.sql("""drop table if exists proceso_seguridad_externa.gmm_preferencial_pronostico purge""")
spark.sql("""create table if not exists proceso_seguridad_externa.gmm_preferencial_pronostico stored as parquet as
			 select documento, indicador, prediction as agrupamiento, maximum_probability as probabilidad
			 from forecastData_gmm""")
}
}
