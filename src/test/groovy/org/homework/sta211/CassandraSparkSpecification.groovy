package org.homework.sta211

import com.datastax.spark.connector.japi.CassandraJavaUtil
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import spock.lang.Shared
import spock.lang.Specification

import static org.apache.spark.sql.functions.col
import static org.apache.spark.sql.functions.avg

class CassandraSparkSpecification extends Specification {


    @Shared
    JavaSparkContext sparkContext

    @Shared
    SparkSession session

    def setupSpec() {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("CassandraSparkSpecification")
                .set("spark.cassandra.connection.host", "127.0.0.1")
                .set("spark.cassandra.connection.port", "9042")

        sparkContext = new JavaSparkContext(conf)
        session = SparkSession.builder().config(conf).getOrCreate()
    }

    def "Comptage des lignes de la table restaurant dans cassandra - RDD"() {
        setup:
        def counter = CassandraJavaUtil.javaFunctions(sparkContext).cassandraTable("resto_ny", "restaurant").count()

        when:
        println counter

        then:
        assert true
    }

    def "Comptage des lignes de la table restaurant dans cassandra - Dataset"() {
        setup:
        Dataset<Row> restaurantDS = session.read()
                .format("org.apache.spark.sql.cassandra")
                .options([table: "restaurant", keyspace: "resto_ny"])
                .load()
        when:
        restaurantDS.show()

        then:
        assert true
    }

    def "Le traitement suivant effectue la moyenne des votes pour les restaurants de Tapas."() {
        setup:
        Dataset<Row> restaurantDS = session.read()
                .format("org.apache.spark.sql.cassandra")
                .options([table: "restaurant", keyspace: "resto_ny"])
                .load()

        Dataset<Row> inspectionDS = session.read()
                .format("org.apache.spark.sql.cassandra")
                .options([table: "inspection", keyspace: "resto_ny"])
                .load()

        when:
        restaurantDS.filter(col("cuisinetype").equalTo("Tapas"))
                .join(inspectionDS, col("idRestaurant").equalTo(col("id")))
                .groupBy(col("name")).agg(avg(col("score"))).explain(true)

        then:
        assert true
    }

}
