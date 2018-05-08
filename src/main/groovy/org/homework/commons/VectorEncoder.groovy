package org.homework.commons

import org.apache.spark.sql.Encoder
import org.apache.spark.sql.types.StructType
import scala.reflect.ClassTag

class VectorEncoder implements Encoder<Vector> {
    @Override
    StructType schema() {
        return null
    }

    @Override
    ClassTag<Vector> clsTag() {
        return null
    }
}
