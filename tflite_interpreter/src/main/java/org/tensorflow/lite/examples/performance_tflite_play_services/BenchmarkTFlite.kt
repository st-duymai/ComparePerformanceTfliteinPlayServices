package org.tensorflow.lite.examples.performance_tflite_play_services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.performance_tflite_play_services.model.Model
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.pow
import kotlin.math.sqrt

class BenchmarkTFlite(
    private val context: Context,
    private val bitmap: Bitmap,
    private val models: List<Model>
) : Benchmark {

    companion object {
        private const val CPU_THREADS = 4
        private const val NUM_EXECUTE = 50
    }

    private fun prepareImageTensor(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        dataType: DataType
    ): TensorImage? {
        val imageProcessor = ImageProcessor.Builder().apply {
            add(ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
        }.build()
        val tensorImage = TensorImage(dataType)
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
    }

    override fun run() {
        models.forEach { model ->
            val modelBuffer =
                FileUtil.loadMappedFile(context, model.getPath())
            val options = Interpreter.Options().apply {
                numThreads = CPU_THREADS
            }
            val interpreter = Interpreter(modelBuffer, options)
            val dataType =
                if (model.dataType == InputDataType.UINT8) DataType.UINT8 else DataType.FLOAT32

            val inputTensor =
                prepareImageTensor(bitmap, model.height, model.width, dataType)
            val outputShape = interpreter.getOutputTensor(0).shape()
            val outputTensor = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)

            val timeExecute = mutableListOf<Long>()
            (0..NUM_EXECUTE).forEach { _ ->
                val startTime = System.nanoTime()
                interpreter.run(inputTensor?.buffer, outputTensor.buffer.rewind())
                timeExecute.add((System.nanoTime() - startTime) / 1000000)
            }

            // Remove first time execution because first time execution is always take longer time than the others.
            timeExecute.removeAt(0)

            val result = CalculateUtils.calculateStandardDeviation(timeExecute)

            Log.d(
                "Time execute ${model.name}",
                "${result.first.toInt()} ms"
            )
            Log.d(
                "Standard deviation ${model.name}",
                "${result.second.toInt()} ms"
            )
            interpreter.close()
        }
    }
}
