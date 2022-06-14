package org.tensorflow.lite.examples.tflite_interpreter

import org.junit.Test

import org.junit.Assert.*
import org.tensorflow.lite.examples.performance_tflite_play_services.CalculateUtils

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class CalculateUtilsTest {
    @Test
    fun test_benchmark_result() {
        val population = listOf<Long>(2, 4, 4, 4, 5, 5, 7, 9)
        val average = CalculateUtils.calculateExecuteAverage(population)

        val sd = CalculateUtils.calculateStandardDeviation(population)

        assertEquals(5.0, average, 0.01)

        assertEquals(2.0, sd, 0.01)
    }
}