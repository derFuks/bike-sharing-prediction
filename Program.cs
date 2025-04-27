using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BikeSharingPrediction
{
    class Program
    {
        private static string _dataPath = "bike_sharing.csv"; // Путь к CSV файлу
        // классы для обработки данных
        public class BikeRentalData
        {
            [LoadColumn(0)]
            public float Season { get; set; }

            [LoadColumn(1)]
            public float Month { get; set; }

            [LoadColumn(2)]
            public float Hour { get; set; }

            [LoadColumn(3)]
            public float Holiday { get; set; }

            [LoadColumn(4)]
            public float Weekday { get; set; }

            [LoadColumn(5)]
            public float WorkingDay { get; set; }

            [LoadColumn(6)]
            public float WeatherCondition { get; set; }

            [LoadColumn(7)]
            public float Temperature { get; set; }

            [LoadColumn(8)]
            public float Humidity { get; set; }

            [LoadColumn(9)]
            public float Windspeed { get; set; }

            [LoadColumn(10)]
            public bool RentalType { get; set; } // 0 = краткосрочная, 1 = долгосрочная
        }

        public class RentalTypePrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedRentalType { get; set; }

            public float Probability { get; set; }

            public float Score { get; set; }
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Предсказание типа аренды велосипедов.");

            var mlContext = new MLContext(seed: 0);

            // загружаю данные
            var data = mlContext.Data.LoadFromTextFile<BikeRentalData>(
                path: _dataPath,
                hasHeader: true,
                separatorChar: ',');
            
            // разделяю выборку на обучающую и тестовую в 20%
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            Console.WriteLine("Данные загружены, разделены (0_o)");

            // делаю пайплайн обработки признаков
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", nameof(BikeRentalData.RentalType))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[]
            {
                new InputOutputColumnPair("SeasonEncoded", nameof(BikeRentalData.Season)),
                new InputOutputColumnPair("WeatherConditionEncoded", nameof(BikeRentalData.WeatherCondition))
            }))
            
            .Append(mlContext.Transforms.NormalizeMinMax(nameof(BikeRentalData.Temperature)))
            .Append(mlContext.Transforms.NormalizeMinMax(nameof(BikeRentalData.Humidity)))
            .Append(mlContext.Transforms.NormalizeMinMax(nameof(BikeRentalData.Windspeed)))
            .Append(mlContext.Transforms.Concatenate("Features",
            "SeasonEncoded",
            "WeatherConditionEncoded",
            nameof(BikeRentalData.Month),
            nameof(BikeRentalData.Hour),
            nameof(BikeRentalData.Holiday),
            nameof(BikeRentalData.Weekday),
            nameof(BikeRentalData.WorkingDay),
            nameof(BikeRentalData.Temperature),
            nameof(BikeRentalData.Humidity),
            nameof(BikeRentalData.Windspeed)
            )
            )
            ;
            Console.WriteLine("Пайплайн готов =^_^= ");

            //Обучаю по алгоритмам FastTree, LightGBM, Logistic Regression. Лучшего определим по AUC/F1-Score
            //FastTree
            var trainerFastTree = mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipelineFastTree = dataProcessPipeline.Append(trainerFastTree);
            // Обучение модели
            var modelFastTree = trainingPipelineFastTree.Fit(trainData);
            // Оценка модели на тестовых данных
            var predictionsFastTree = modelFastTree.Transform(testData);
            var metricsFastTree = mlContext.BinaryClassification.Evaluate(
                predictionsFastTree,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                probabilityColumnName: "Probability",
                predictedLabelColumnName: "PredictedLabel"
            );

            Console.WriteLine($"Точность FastTree: {metricsFastTree.Accuracy:P2}, AUC: {metricsFastTree.AreaUnderRocCurve:P2}, F1 Score: {metricsFastTree.F1Score:P2}");

            // LightGBM !!! Работаю на Mac, и тут очень сложно напрямую использовать LightGBM из ML.NET без пересборки исходников. Просто закоментирую, может потом, если будет время покопаю в глубь.
            // var trainerLightGbm = mlContext.BinaryClassification.Trainers.LightGbm(
            //     labelColumnName: "Label", featureColumnName: "Features");
            // var trainingPipelineLightGbm = dataProcessPipeline.Append(trainerLightGbm);
            // var modelLightGbm = trainingPipelineLightGbm.Fit(trainData);
            // var predictionsLightGbm = modelLightGbm.Transform(testData);
            // var metricsLightGbm = mlContext.BinaryClassification.Evaluate(predictionsLightGbm, labelColumnName: "Label");

            // Console.WriteLine($"Точность LightGBM: {metricsLightGbm.Accuracy:P2}, AUC: {metricsLightGbm.AreaUnderRocCurve:P2}, F1 Score: {metricsLightGbm.F1Score:P2}");

            // Logistic Regression
            var trainerLogisticRegression = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipelineLogistic = dataProcessPipeline.Append(trainerLogisticRegression);
            var modelLogistic = trainingPipelineLogistic.Fit(trainData);
            var predictionsLogistic = modelLogistic.Transform(testData);
            var metricsLogistic = mlContext.BinaryClassification.Evaluate(
                predictionsLogistic,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                probabilityColumnName: "Probability",
                predictedLabelColumnName: "PredictedLabel"
            );

            Console.WriteLine($"Точность Logistic Regression: {metricsLogistic.Accuracy:P2}, AUC: {metricsLogistic.AreaUnderRocCurve:P2}, F1 Score: {metricsLogistic.F1Score:P2}");

            // результаты обучения отвратительны. Мало, слишком мало данных. Попробую еще Averaged Perceptron
            var trainerPerceptron = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipelinePerceptron = dataProcessPipeline.Append(trainerPerceptron);
            var modelPerceptron = trainingPipelinePerceptron.Fit(trainData);
            var predictionsPerceptron = modelPerceptron.Transform(testData);
            var metricsPerceptron = mlContext.BinaryClassification.EvaluateNonCalibrated( // ловлю исключения, тк ML.NET ищет перегрузку Evaluate, где ожидается все 4 параметра включая probabilityColumnName. Нашел EvaluateNonCalibrated
                predictionsPerceptron,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                predictedLabelColumnName: "PredictedLabel" // <= без probability для perceptron, иначе крашится
            );

            Console.WriteLine($"Точность Averaged Perceptron: {metricsPerceptron.Accuracy:P2}, AUC: {metricsPerceptron.AreaUnderRocCurve:P2}, F1 Score: {metricsPerceptron.F1Score:P2}");

            Console.WriteLine("Жмяк любую клавишу для выхода...");
            Console.ReadKey();
        }
    }
}
