package org.tensorflow.lite.examples.performance_tflite_play_services

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.examples.performance_tflite_play_services.model.Model
import java.io.IOException
import java.io.InputStream

class MainActivity : AppCompatActivity() {
    companion object {
        private var TAG = MainActivity::class.java.simpleName
        val models = listOf(
            Model("efficient_net_lite3.tflite", 512, 512, InputDataType.UINT8),
            Model("efficient_net_lite4.tflite", 300, 300, InputDataType.FLOAT32),
            Model("landmarks_classifier_north_america.tflite", 321, 321, InputDataType.UINT8),
            Model("landmarks_classifier_oceania_antarctica.tflite", 321, 321, InputDataType.UINT8),
            Model("mobile_food_segmenter_v1.tflite", 513, 513, InputDataType.UINT8)
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val btnStartBenchmark = findViewById<Button>(R.id.startBenchMark)

        btnStartBenchmark.setOnClickListener {
            getBitmapFromAsset(this, "images/input_image.jpeg")?.let { bitmap ->
                val benchmarkTFlite = BenchmarkTFlite(this, bitmap, models)
                benchmarkTFlite.run()
            }
        }
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
