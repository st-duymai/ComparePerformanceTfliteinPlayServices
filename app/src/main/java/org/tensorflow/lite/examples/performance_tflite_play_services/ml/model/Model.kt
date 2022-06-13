package org.tensorflow.lite.examples.performance_tflite_play_services.ml.model

import org.tensorflow.lite.DataType

data class Model(val name: String, val width: Int, val height: Int, val dataType: DataType) {

    fun getPath() = "models/$name"
}
