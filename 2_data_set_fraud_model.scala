/*This values is just a SQL string sentence that eventually calls, through the spark.sql() method, the input data.*/
val sql_lz = "select * from proceso_seguridad_externa.scoring_variables_preferenciales order by documento"
