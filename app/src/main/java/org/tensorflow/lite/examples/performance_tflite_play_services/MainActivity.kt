package org.tensorflow.lite.examples.performance_tflite_play_services

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.invoke
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.examples.performance_tflite_play_services.ml.model.Model
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.io.InputStream

class MainActivity : AppCompatActivity() {
    companion object {
        private const val NUM_EXECUTE = 50
        private const val CPU_THREADS = 4
        private val models = listOf(
            Model("efficient_net_lite3.tflite", 512, 512, DataType.UINT8),
            Model("efficient_net_lite4.tflite", 300, 300, DataType.FLOAT32),
            Model("landmarks_classifier_north_america.tflite", 321, 321, DataType.UINT8),
            Model("landmarks_classifier_oceania_antarctica.tflite", 321, 321, DataType.UINT8),
            Model("mobile_food_segmenter_v1.tflite", 513, 513, DataType.UINT8)
        )
    }

    private lateinit var interpreter: InterpreterApi
    private val initializeTask: Task<Void> by lazy { TfLite.initialize(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val btnStartBenchmark = findViewById<Button>(R.id.startBenchMark)

        btnStartBenchmark.setOnClickListener {
            lifecycleScope.launch {
                Dispatchers.IO {
                    benchMark()
                }
            }
        }
    }

    private fun benchMark() {
        initializeTask.addOnSuccessListener {
            getBitmapFromAsset(this, "images/input_image.jpeg")?.let { bitmap ->
                models.forEach { model ->
                    val modelBuffer =
                        FileUtil.loadMappedFile(this, model.getPath())
                    interpreter = InterpreterApi.create(
                        modelBuffer, InterpreterApi.Options().apply {
                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                            numThreads = CPU_THREADS
                        }
                    )
                    val outputShape = interpreter.getOutputTensor(0).shape()
                    val inputTensor =
                        prepareImageTensor(bitmap, model.height, model.width, model.dataType)
                    val outputTensor = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
                    val timeExecutes = mutableListOf<Long>()
                    (0 until NUM_EXECUTE).forEach { _ ->
                        val startTime = System.nanoTime()
                        interpreter.run(inputTensor?.buffer, outputTensor.buffer.rewind())
                        val executeTime = (System.nanoTime() - startTime)
                        timeExecutes.add(executeTime)
                    }
                    val timeExecute = timeExecutes.average() / 1000000
                    Log.d("Time execute ${model.name}", "${timeExecute.toInt()} ms")
                    interpreter.close()
                }
                with(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Execute done", Toast.LENGTH_SHORT).show()
                }
            }
        }.addOnFailureListener { ex ->
            Log.e(
                "Interpreter",
                "Cannot initialize interpreter",
                ex
            )
        }
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

    private fun getBitmapFromAsset(context: Context, filePath: String?): Bitmap? {
        val assetManager: AssetManager = context.assets
        val istr: InputStream
        var bitmap: Bitmap? = null
        try {
            istr = assetManager.open(filePath!!)
            bitmap = BitmapFactory.decodeStream(istr)
        } catch (e: IOException) {
            // handle exception
        }
        return bitmap
    }
}
