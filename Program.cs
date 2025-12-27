using YamnetRealtime;

// ═══════════════════════════════════════════════════════════════
//  YAMNet Real-Time Audio Classification
//  Using ML.NET with TensorFlow backend
// ═══════════════════════════════════════════════════════════════

Console.OutputEncoding = System.Text.Encoding.UTF8;
Console.Clear();

PrintBanner();

// List available audio devices
AudioCapture.ListDevices();
Console.WriteLine();

// Initialize classifier
using var classifier = new YamnetClassifier();

try {
    await classifier.InitializeAsync();
}
catch (Exception ex) {
    Console.WriteLine($"\n❌ Failed to load model: {ex.Message}");
    Console.WriteLine("\nMake sure you have downloaded the YAMNet model:");
    Console.WriteLine("  1. Download from: https://tfhub.dev/google/yamnet/1");
    Console.WriteLine("  2. Extract to 'yamnet_model' directory");
    Console.WriteLine("\nSee README.md for detailed instructions.");
    return;
}

Console.WriteLine();

// Initialize audio capture
// YAMNet requires: 16kHz sample rate, 15600 samples (~0.975 seconds)
using var audioCapture = new AudioCapture(sampleRate: 16000, samplesNeeded: 15600);

// Optional: Set specific device (uncomment and change index)
// audioCapture.SetDevice(0);

// Track processing state to avoid overlapping classifications
var isProcessing = false;
var lastResults = new List<ClassificationResult>();

// Handle audio chunks
audioCapture.OnAudioReady += waveform => {
    if (isProcessing) return;
    isProcessing = true;

    try {
        var results = classifier.Classify(waveform, topK: 9);
        lastResults = results;
        DisplayResults(results);
    }
    catch (Exception ex) {
        Console.WriteLine($"\n❌ Classification error: {ex.Message}");
    }
    finally {
        isProcessing = false;
    }
};

// Start recording
try {
    audioCapture.Start();
}
catch {
    Console.WriteLine("\n❌ Could not start audio capture.");
    Console.WriteLine("   Make sure you have a working microphone connected.");
    return;
}

Console.WriteLine("\n" + new string('─', 60));
Console.WriteLine("Press [Enter] to stop, [S] to save last result, [Q] to quit");
Console.WriteLine(new string('─', 60) + "\n");

while (true) {
    if (Console.KeyAvailable) {
        var key = Console.ReadKey(intercept: true);

        if (key.Key == ConsoleKey.Enter || key.Key == ConsoleKey.Q) {
            break;
        }

        if (key.Key == ConsoleKey.S && lastResults.Count > 0) {
            SaveResults(lastResults);
        }
    }

    await Task.Delay(50);
}

audioCapture.Stop();
Console.WriteLine("\n✅ Recording stopped.");

// ═══════════════════════════════════════════════════════════════
//  Helper Methods
// ═══════════════════════════════════════════════════════════════

void PrintBanner() {
    Console.WriteLine("╔═══════════════════════════════════════════════════════════╗");
    Console.WriteLine("║       🎤 YAMNet Real-Time Audio Classification            ║");
    Console.WriteLine("║          ML.NET + TensorFlow | 521 Sound Classes          ║");
    Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
    Console.WriteLine();
}

void DisplayResults(List<ClassificationResult> results) {
    // Save cursor position
    var currentRow = Console.CursorTop;

    // Move to display area
    Console.SetCursorPosition(0, Console.CursorTop);

    Console.WriteLine("┌─────────────────────────────────────────────────────────────┐");
    Console.WriteLine("│  Current Classifications                                    │");
    Console.WriteLine("├─────────────────────────────────────────────────────────────┤");

    foreach (var (result, index) in results.Select((r, i) => (r, i))) {
        var rank = (index + 1).ToString();
        var name = TruncateString(result.ClassName, 22);
        var percentage = (result.Score * 100).ToString("F1").PadLeft(5);
        var barLength = Math.Min((int)(result.Score * 25), 25);
        var bar = new string('█', barLength) + new string('░', 25 - barLength);

        Console.WriteLine($"│  {rank}. {name.PadRight(22)} {percentage}% {bar} │");
    }

    Console.WriteLine("├─────────────────────────────────────────────────────────────┤");
    Console.WriteLine($"│  {DateTime.Now:HH:mm:ss} | Press Enter to stop                             │");
    Console.WriteLine("└─────────────────────────────────────────────────────────────┘");
}

void SaveResults(List<ClassificationResult> results) {
    var filename = $"detection_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
    var lines = new List<string>
    {
        $"YAMNet Detection - {DateTime.Now:yyyy-MM-dd HH:mm:ss}",
        new string('-', 50)
    };

    foreach (var r in results) {
        lines.Add($"{r.ClassName}: {r.Score * 100:F2}%");
    }

    File.WriteAllLines(filename, lines);
    Console.WriteLine($"\n💾 Saved to {filename}");
}

string TruncateString(string str, int maxLength) {
    if (str.Length <= maxLength) return str;
    return str[..(maxLength - 2)] + "..";
}
