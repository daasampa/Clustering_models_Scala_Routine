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