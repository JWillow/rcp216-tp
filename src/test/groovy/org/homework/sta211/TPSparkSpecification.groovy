package org.homework.sta211

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.SparkSession
import spock.lang.Shared
import spock.lang.Specification

class TPSparkSpecification extends Specification {

    File loups = new File("./src/test/resources/loups.txt")

    @Shared
    JavaSparkContext sparkContext

    @Shared
    SparkSession session

    def setupSpec() {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("TPSparkSpecification")
        sparkContext = new JavaSparkContext(conf)
        session = SparkSession.builder().config(conf).getOrCreate()
    }

    def "Comptage des lignes du fichier loups"() {
        setup:
        JavaRDD<String> loupsEtMoutons =  sparkContext.textFile(loups.getPath())

        when:
        long result = loupsEtMoutons.count()

        then:
        result == 4
    }

}
