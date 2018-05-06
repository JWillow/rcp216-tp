package org.homework.tpnum

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import spock.lang.Shared
import spock.lang.Specification

class TPNumSpecification extends Specification {

    @Shared
    JavaSparkContext sparkContext

    @Shared
    SparkSession session

    def setupSpec() {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("TPNumSpecification")
        sparkContext = new JavaSparkContext(conf)
        session = SparkSession.builder().config(conf).getOrCreate()
    }

    def "Produit d'une matrice et d'un vecteur"() {
        setup:
        Vector vectDense = Vectors.dense(1.0,2.0)
        double[] denseMatricContent = [1.2, 3.4, 5.6, 2.3, 4.5, 6.7]
        DenseMatrix denseMatrix = new DenseMatrix(3, 2, denseMatricContent)
        println denseMatrix

        when:
        DenseVector vectorResult = denseMatrix.multiply(vectDense)

        then:
        vectorResult != null
        println vectorResult
    }

    def "Lecture d'un fichier csv"() {
        when:
        def geysercsvdf = session.read().format("csv").option("header",true).load("src/test/resources/tpnum/geyser.csv")

        then:
        geysercsvdf.printSchema()
        geysercsvdf.show(5)
    }

    def "Lecture d'un fichier txt avec séparateur espace"() {
        when:
        def geysercsvdf = session.read().format("csv").option("sep"," ").load("src/test/resources/tpnum/geyser.txt")

        then:
        geysercsvdf.printSchema()
        geysercsvdf.show(5)
    }

    def "Lecture d'un fichier au format libsvm"() {
        when:
        def dataset = session.read().format("libsvm").load("src/test/resources/tpnum/sample_libsvm_data.txt")

        then:
        dataset.printSchema()
        dataset.show(5)
    }

}
