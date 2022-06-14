package org.tensorflow.lite.examples.performance_tflite_play_services

import kotlin.math.pow
import kotlin.math.sqrt

object CalculateUtils {

    fun calculateExecuteAverage(timeExecute: List<Long>): Double {
        return timeExecute.average()
    }

    /**
     * Return average execute time and standard deviation.
     */
    fun calculateStandardDeviation(timeExecute: List<Long>): Pair<Double, Double> {
        val average = calculateExecuteAverage(timeExecute)
        val variance = timeExecute.map { (it - average).pow(2) }.average()
        return Pair(average, sqrt(variance))
    }
}