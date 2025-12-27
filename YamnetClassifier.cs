using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;

namespace YamnetRealtime;

/// <summary>
/// Classification result from YAMNet
/// </summary>
public record ClassificationResult(string ClassName, float Score, int ClassIndex);

/// <summary>
/// YAMNet audio classifier using ONNX Runtime
/// </summary>
public class YamnetClassifier : IDisposable {
    private InferenceSession? _session;
    private Dictionary<int, string> _classMap = [];
    private string? _inputName;
    private string? _outputName;

    /// <summary>
    /// Initializes the classifier by loading ONNX model and class map
    /// </summary>
    public async Task InitializeAsync(string modelPath = "yamnet.onnx") {
        Console.WriteLine("Loading YAMNet ONNX model...");

        if (!File.Exists(modelPath)) {
            throw new FileNotFoundException(
                $"ONNX model not found: {modelPath}\n" +
                "Please run 'node setup.mjs' to download the model, or download manually.\n" +
                "See README.md for instructions.");
        }

        var options = new SessionOptions {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        _session = new InferenceSession(modelPath, options);

        // Get input/output names from model
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();

        Console.WriteLine($"   Input:  {_inputName} {FormatShape(_session.InputMetadata[_inputName].Dimensions)}");
        Console.WriteLine($"   Output: {_outputName} {FormatShape(_session.OutputMetadata[_outputName].Dimensions)}");
        Console.WriteLine("✅ YAMNet ONNX model loaded");

        // Load class labels
        await LoadClassMapAsync();
    }

    private static string FormatShape(int[] dimensions) {
        return $"[{string.Join(", ", dimensions)}]";
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
    /// <param name="waveform">Audio samples (16kHz, mono, ~0.975s = 15600 samples)</param>
    /// <param name="topK">Number of top predictions to return</param>
    public List<ClassificationResult> Classify(float[] waveform, int topK = 5) {
        if (_session == null || _inputName == null || _outputName == null)
            throw new InvalidOperationException("Model not loaded. Call InitializeAsync first.");

        var inputTensor = new DenseTensor<float>(waveform, [waveform.Length]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        using var results = _session.Run(inputs);

        var scoresOutput = results.First(r => r.Name == _outputName);
        var scoresTensor = scoresOutput.AsTensor<float>();

        var avgScores = AverageScores(scoresTensor);

        return avgScores
            .Select((score, index) => new ClassificationResult(
                _classMap.GetValueOrDefault(index, $"Class {index}"),
                score,
                index))
            .OrderByDescending(r => r.Score)
            .Take(topK)
            .ToList();
    }

    /// <summary>
    /// Averages scores across frames (if model returns multiple frames)
    /// </summary>
    private float[] AverageScores(Tensor<float> scores) {
        var dimensions = scores.Dimensions.ToArray();

        // If 1D tensor, return as-is
        if (dimensions.Length == 1) {
            return [.. scores];
        }

        // If 2D tensor [frames, classes], average across frames
        int numFrames = dimensions[0];
        int numClasses = dimensions[1];
        var avg = new float[numClasses];

        for (int frame = 0; frame < numFrames; frame++) {
            for (int cls = 0; cls < numClasses; cls++) {
                avg[cls] += scores[frame, cls];
            }
        }

        for (int cls = 0; cls < numClasses; cls++) {
            avg[cls] /= numFrames;
        }

        return avg;
    }

    /// <summary>
    /// Gets the class name for a given index
    /// </summary>
    public string GetClassName(int index) => _classMap.GetValueOrDefault(index, $"Unknown ({index})");

    /// <summary>
    /// Gets all available class names
    /// </summary>
    public IReadOnlyDictionary<int, string> GetAllClasses() => _classMap;

    public void Dispose() {
        _session?.Dispose();
    }
}
