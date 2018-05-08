package org.homework.tpkmeans

import com.google.common.base.Joiner
import com.panayotis.gnuplot.GNUPlotParameters
import com.panayotis.gnuplot.JavaPlot
import com.panayotis.gnuplot.dataset.FileDataSet
import com.panayotis.gnuplot.layout.StripeLayout
import com.panayotis.gnuplot.plot.AbstractPlot
import com.panayotis.gnuplot.plot.DataSetPlot
import com.panayotis.gnuplot.style.NamedPlotColor
import com.panayotis.gnuplot.style.PlotStyle
import com.panayotis.gnuplot.style.Style
import com.panayotis.gnuplot.utils.Debug
import org.apache.commons.io.FileUtils
import org.apache.commons.lang.ArrayUtils
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.Function
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.util.KMeansDataGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.homework.commons.Gnuplot
import org.homework.commons.SparkFileUtils
import spock.lang.Shared
import spock.lang.Specification

import static org.apache.spark.sql.types.DataTypes.DoubleType

class TPkmeans extends Specification implements Serializable {

    @Shared
    JavaSparkContext sparkContext

    @Shared
    SparkSession session

    File workDir = new File("/tmp/rcp216")
    File kmeansDataGenerated = new File(workDir, "kmeans")
    File kmeansDataGeneratedTxt = new File(workDir, "kmeans.txt")
    File kmeansFinal = new File(workDir, "kmeans-classified.txt")

    def setupSpec() {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("TPkmeansSpecification")
        sparkContext = new JavaSparkContext(conf)
        session = SparkSession.builder().config(conf).getOrCreate()

    }

    def "Generation du jeu de données 5 groupes"() {
        setup:
        FileUtils.deleteDirectory(kmeansDataGenerated)

        when:
        RDD<double[]> rddResult = KMeansDataGenerator.generateKMeansRDD(sparkContext.sc(), 1000, 5, 3, 5.0d, 1)

        then:
        rddResult.toJavaRDD().repartition(1).map(new Function<double[], String>() {
            String call(double[] values) throws Exception {
                return Joiner.on(" ").skipNulls().join(Arrays.asList(ArrayUtils.toObject(values)))
            }
        }).saveAsTextFile(kmeansDataGenerated.getCanonicalPath())
        SparkFileUtils.renameFile(kmeansDataGenerated, kmeansDataGeneratedTxt)
    }


    def "Réalisation des k-means"() {
        setup:
        String[] inputFields = ["c1","c2","c3"]
        assert kmeansDataGeneratedTxt.exists()

        Dataset<Row> dataset = session
                .read().format("csv").option("sep", " ").schema(new StructType().add("c1", DoubleType).add("c2", DoubleType).add("c3", DoubleType))
                .load(kmeansDataGeneratedTxt.getCanonicalPath()).cache()

        dataset.printSchema()

        Dataset<Row> datasetVectorised = new VectorAssembler().setInputCols(inputFields).setOutputCol("features").transform(dataset)
        datasetVectorised.printSchema()

        KMeans kmeans = new KMeans().setK(5).setMaxIter(200).setSeed(1L)
        KMeansModel model = kmeans.fit(datasetVectorised)

        when:
        Dataset<Row> indices = model.transform(datasetVectorised)
        indices.printSchema()

        then:
        SparkFileUtils.save(indices.select("c1", "c2", "c3", "prediction"), kmeansFinal)
        assert true
        /*
        gnuplot> set datafile separator ','
        gnuplot> set palette defined ( 0 "red", 1 "orange", 2 "brown", 3 "green", 4 "blue" )
        gnuplot> splot "data/donneesGnuplot.txt" using 1:2:3:4 with points lc palette
         */
    }

}
