/*Scala classes needed to transform data and implement the estimation procedure for the probabilistic clustering methods
using the stage-defined pipeline model.*/
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.VectorAssembler 
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.GaussianMixture
