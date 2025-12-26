using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text.RegularExpressions;

namespace YamnetRealtime;

/// <summary>
/// Classification result from YAMNet
/// </summary>
public record ClassificationResult(string ClassName, float Score, int ClassIndex);

/// <summary>
/// Input schema for YAMNet model
/// </summary>
public class YamnetInput {
    [VectorType(15600)]
    [ColumnName("waveform")]
    public float[] Waveform { get; set; } = Array.Empty<float>();
}

/// <summary>
/// Output schema for YAMNet model (521 AudioSet classes)
/// </summary>
public class YamnetOutput {
    [VectorType(521)]
    [ColumnName("scores")]
    public float[] Scores { get; set; } = Array.Empty<float>();
}

/// <summary>
/// YAMNet audio classifier using ML.NET with TensorFlow backend
/// </summary>
public class YamnetClassifier : IDisposable {
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private PredictionEngine<YamnetInput, YamnetOutput>? _predictionEngine;
    private Dictionary<int, string> _classMap = new();

    public YamnetClassifier() {
        _mlContext = new MLContext(seed: 0);
    }

    /// <summary>
    /// Initializes the classifier by loading model and class map
    /// </summary>
    public async Task InitializeAsync(string modelPath = "yamnet_model") {
        Console.WriteLine("Loading YAMNet model...");

        if (!Directory.Exists(modelPath)) {
            throw new DirectoryNotFoundException(
                $"Model directory not found: {modelPath}\n" +
                "Please download the YAMNet SavedModel first. See README.md for instructions.");
        }

        // Load TensorFlow SavedModel
        var tensorFlowModel = _mlContext.Model.LoadTensorFlowModel(modelPath);

        // Inspect and display model schema
        var schema = tensorFlowModel.GetModelSchema();
        Console.WriteLine("\n   Model tensors found:");

        string? inputTensorName = null;
        string? outputTensorName = null;

        foreach (var col in schema) {
            var typeName = col.Type.ToString();
            Console.WriteLine($"   - {col.Name}: {typeName}");

            // Find input tensor (contains "waveform" in name)
            if (col.Name.Contains("waveform", StringComparison.OrdinalIgnoreCase)) {
                inputTensorName = col.Name;
            }

            // Find output tensor (contains "scores" or is a vector of 521)
            if (col.Name.Contains("scores", StringComparison.OrdinalIgnoreCase) ||
                typeName.Contains("521")) {
                outputTensorName = col.Name;
            }
        }

        // Fallback: try common TensorFlow Hub naming patterns
        if (inputTensorName == null) {
            var inputCandidates = new[] {
                "serving_default_waveform",
                "waveform",
                "input",
                "input_1",
                "args_0"
            };
            inputTensorName = inputCandidates.FirstOrDefault(c =>
                schema.Any(s => s.Name.Equals(c, StringComparison.OrdinalIgnoreCase)));
        }

        if (outputTensorName == null) {
            var outputCandidates = new[] {
                "scores",
                "output_0",
                "StatefulPartitionedCall",
                "Identity",
                "PartitionedCall"
            };
            outputTensorName = outputCandidates.FirstOrDefault(c =>
                schema.Any(s => s.Name.Equals(c, StringComparison.OrdinalIgnoreCase)));
        }

        if (inputTensorName == null || outputTensorName == null) {
            Console.WriteLine("\n⚠️ Could not auto-detect tensor names.");
            Console.WriteLine("Available tensors:");
            foreach (var col in schema) {
                Console.WriteLine($"   {col.Name}");
            }
            throw new InvalidOperationException("Could not find input/output tensors. Please update code with correct names.");
        }

        Console.WriteLine($"\n   Using input tensor:  {inputTensorName}");
        Console.WriteLine($"   Using output tensor: {outputTensorName}");

        // Build pipeline with correct tensor names
        // First rename our input column to match TensorFlow's expected name
        var pipeline = _mlContext.Transforms.CopyColumns(
                outputColumnName: inputTensorName,
                inputColumnName: "waveform")
            .Append(tensorFlowModel.ScoreTensorFlowModel(
                outputColumnNames: new[] { outputTensorName },
                inputColumnNames: new[] { inputTensorName },
                addBatchDimensionInput: false))
            .Append(_mlContext.Transforms.CopyColumns(
                outputColumnName: "scores",
                inputColumnName: outputTensorName));

        // Fit on empty data
        var emptyData = _mlContext.Data.LoadFromEnumerable(Array.Empty<YamnetInput>());
        _model = pipeline.Fit(emptyData);

        // Create prediction engine
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<YamnetInput, YamnetOutput>(_model);

        Console.WriteLine("\n✅ YAMNet model loaded successfully");

        // Load class labels
        await LoadClassMapAsync();
    }

    /// <summary>
    /// Downloads and loads the AudioSet class map (521 classes)
    /// </summary>
    private async Task LoadClassMapAsync() {
        const string url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv";
        const string localPath = "yamnet_class_map.csv";

        string csv;

        if (File.Exists(localPath)) {
            Console.WriteLine("Loading class map from local file...");
            csv = await File.ReadAllTextAsync(localPath);
        }
        else {
            Console.WriteLine("Downloading class map...");
            using var http = new HttpClient();
            http.Timeout = TimeSpan.FromSeconds(30);
            csv = await http.GetStringAsync(url);
            await File.WriteAllTextAsync(localPath, csv);
        }

        // Parse CSV: index, mid, display_name
        foreach (var line in csv.Split('\n').Skip(1)) {
            var match = Regex.Match(line, @"^(\d+),([^,]+),(.+)$");
            if (match.Success) {
                var index = int.Parse(match.Groups[1].Value);
                var displayName = match.Groups[3].Value.Trim().Trim('"');
                _classMap[index] = displayName;
            }
        }

        Console.WriteLine($"✅ Loaded {_classMap.Count} class labels");
    }

    /// <summary>
    /// Classifies audio waveform and returns top K predictions
    /// </summary>
    public List<ClassificationResult> Classify(float[] waveform, int topK = 5) {
        if (_predictionEngine == null)
            throw new InvalidOperationException("Model not loaded. Call InitializeAsync first.");

        if (waveform.Length != 15600) {
            Console.WriteLine($"⚠️ Warning: Expected 15600 samples, got {waveform.Length}");
        }

        var input = new YamnetInput { Waveform = waveform };
        var output = _predictionEngine.Predict(input);

        if (output.Scores == null || output.Scores.Length == 0) {
            return new List<ClassificationResult>();
        }

        return output.Scores
            .Select((score, index) => new ClassificationResult(
                _classMap.GetValueOrDefault(index, $"Class {index}"),
                score,
                index))
            .OrderByDescending(r => r.Score)
            .Take(topK)
            .ToList();
    }

    /// <summary>
    /// Gets the class name for a given index
    /// </summary>
    public string GetClassName(int index) => _classMap.GetValueOrDefault(index, $"Unknown ({index})");

    public void Dispose() {
        _predictionEngine?.Dispose();
    }
}
