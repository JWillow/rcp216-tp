package org.homework.tpnum

import com.panayotis.gnuplot.JavaPlot
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.Function
import org.apache.spark.api.java.function.PairFunction
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.feature.PCAModel
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.StandardScalerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.RowFactory
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import org.homework.commons.Gnuplot
import scala.Tuple2
import spock.lang.Shared
import spock.lang.Specification

import static org.apache.spark.sql.functions.col
import static org.apache.spark.sql.functions.randn

// wget https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data --no-check-certificate
class TPACPSpecification extends Specification implements Serializable {


    @Shared
    JavaSparkContext sparkContext

    @Shared
    SparkSession session

    @Shared
    Dataset spamDF

    @Shared
    Dataset<Row> subSpamDF

    def setupSpec() {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("TPACPSpecification")
        sparkContext = new JavaSparkContext(conf)
        session = SparkSession.builder().config(conf).getOrCreate()

        StructType structType = new StructType()
        57.times { int i ->
            structType = structType.add("val$i", DataTypes.DoubleType, true)
        }
        structType = structType.add("label", DataTypes.DoubleType, true)
        spamDF = session.read().format("csv").schema(structType).load("src/test/resources/tpacp/spambase.data").cache()

        subSpamDF = spamDF.select("val0", "val1", "label")
    }


    def "Count sur les classes"() {
        when:
        Dataset<Row> result = spamDF.groupBy("label").count()

        then:
        result.show()
    }

    def "Echantillonnage - Simple"() {
        when:
        Dataset<Row> resultFraction1 = subSpamDF.sample(false, 0.5)
        Dataset<Row> resultFraction2 = subSpamDF.sample(false, 0.1)

        then:
        println resultFraction1.count()
        println resultFraction2.count()
    }

    def "Echantillonnage - Simple : comparaison statistique"() {
        when:
        Dataset<Row> resultFraction1 = subSpamDF.sample(false, 0.5)
        Dataset<Row> resultFraction2 = subSpamDF.sample(false, 0.1)

        then:
        subSpamDF.describe().show()
        resultFraction1.describe().show()
        resultFraction2.describe().show()

    }

    def "Echantillonnage stratifié"() {
        setup:
        Map<Double, Double> fractions = [0.0d: 0.5d, 1.0d: 0.5d]

        when:
        Dataset<Row> spamEchStratifie = subSpamDF.stat().sampleBy("label", fractions, 1L)

        then:
        spamEchStratifie.groupBy("label").count().show()
    }

    def "Echantillonnag stratifié plus exacte en passant par les RDD"() {
        setup:
        Map<Double, Double> fractions = [0.0d: 0.5d, 1.0d: 0.5d]

        when:
        JavaPairRDD<Double, Vector> spamEchStratRDD = subSpamDF.toJavaRDD().mapToPair(new TPPairFunction()).sampleByKeyExact(false, fractions)
        then:
        Dataset<Row> spamEchStratifieExact = session.createDataFrame(spamEchStratRDD.map(new TPFunction()), subSpamDF.schema())
        spamEchStratifieExact.groupBy("label").count().show()
    }


    def "Réalisation de l'ACP - sans normalisation"() {
        setup:
        // Transformation du Dataset initial
        String[] inputFields = spamDF.schema().fieldNames().grep { it.startsWith("val") }
        Dataset<Row> spamDFA = new VectorAssembler().setInputCols(inputFields).setOutputCol("features").transform(spamDF)

        when:
        // k : The number of principal components.
        PCAModel pcaModel = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3).fit(spamDFA)

        then:
        println pcaModel.explainedVariance()
        //> [0.9270270117088826,0.07104297298033703,0.001843724769352755]
        println pcaModel.pc()
    }

    def "Réalisation de l'ACP - sans normalisation - avec échantillonnage simple de 10%"() {
        setup:
        // Transformation du Dataset initial
        String[] inputFields = spamDF.schema().fieldNames().grep { it.startsWith("val") }
        Dataset<Row> spamDFA = new VectorAssembler().setInputCols(inputFields).setOutputCol("features").transform(spamDF.sample(false, 0.1))

        when:
        // k : The number of principal components.
        PCAModel pcaModel = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3).fit(spamDFA)

        then:
        println pcaModel.explainedVariance()
        //> [0.9687511876078865,0.030504911502541466,6.025524873548787E-4]
        println pcaModel.pc()
    }

    def "Normalisation des données avant ACP"() {
        setup:
        String[] inputFields = spamDF.schema().fieldNames().grep { it.startsWith("val") }
        Dataset<Row> spamDFA = new VectorAssembler().setInputCols(inputFields).setOutputCol("features").transform(spamDF)
        StandardScaler standardScaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
                .setWithStd(true).setWithMean(true)

        when:
        StandardScalerModel standardScalerModel = standardScaler.fit(spamDFA)
        Dataset<Row> scaledSpamDF = standardScalerModel.transform(spamDFA).select("scaledFeatures")

        then:
        scaledSpamDF.printSchema()
    }

    def "Application d'une ACP sur des données normalisées"() {
        setup:
        String[] inputFields = spamDF.schema().fieldNames().grep { it.startsWith("val") }
        Dataset<Row> spamDFA = new VectorAssembler().setInputCols(inputFields).setOutputCol("features").transform(spamDF)

        // Normalisation des données
        Dataset<Row> spamDFAScaled = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
                .setWithStd(true).setWithMean(true).fit(spamDFA).transform(spamDFA).select("scaledFeatures")

        PCAModel pcaModel = new PCA().setInputCol("scaledFeatures").setOutputCol("pcaScaledFeatures").setK(3).fit(spamDFAScaled)

        when:
        Dataset<Row> resultatCR = pcaModel.transform(spamDFAScaled).select("pcaScaledFeatures")

        then:
        println pcaModel.explainedVariance()
        Gnuplot.display(resultatCR)
    }

    def "ACP sur des données tridimensionnelles obtenues par tirage aléatoire suivant une loi normale isotrope"() {
        setup:
        // Création d'un Dataframe d'identifiants entre 0 et 4600
        Dataset<Long> idsDF = session.range(4601)
        String[] inputFields = ["normale0", "normale1", "normale2"]
        Dataset<Row> randnDF = idsDF.select(col("id"), randn(1).alias("normale0"), randn(2).alias("normale1"), randn(3).alias("normale2"))
        Dataset<Row> randnDF2 = new VectorAssembler().setInputCols(inputFields).setOutputCol("features").transform(randnDF)
        PCAModel pcaModel = new PCA().setInputCol("features").setOutputCol("pcaRndFeatures").setK(3).fit(randnDF2)

        when:
        Dataset<Row> resRnd = pcaModel.transform(randnDF2).select("pcaRndFeatures")

        then:
        Gnuplot.display(resRnd)
        //randnDF.describe().show()
        println pcaModel.explainedVariance()
    }


    def "Test JavaPlot"() {
        when:
        JavaPlot p = new JavaPlot()
        p.addPlot("sin(x)")
        then:
        p.plot()
    }

    class TPFunction implements Function<Tuple2<Double, Vector>, Row>, Serializable {
        @Override
        Row call(Tuple2<Double, Vector> tuple) throws Exception {
            double[] values = tuple._2().toArray()
            return RowFactory.create(values[0], values[1], tuple._1().doubleValue())
        }
    }

    class TPPairFunction implements PairFunction<Row, Double, Vector>, Serializable {
        @Override
        Tuple2<Double, Vector> call(Row row) throws Exception {
            return new Tuple2<>((Double) row.get(2), Vectors.dense((double) row.get(0), (double) row.get(1)))
        }
    }
}
