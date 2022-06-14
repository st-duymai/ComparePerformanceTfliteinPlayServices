package org.tensorflow.lite.examples.performance_tflite_play_services.model

import org.tensorflow.lite.examples.performance_tflite_play_services.InputDataType

data class Model(val name: String, val width: Int, val height: Int, val dataType: InputDataType) {

    fun getPath() = "models/$name"
}
