using YamnetRealtime;

Console.OutputEncoding = System.Text.Encoding.UTF8;

PrintBanner();

AudioCapture.ListDevices();
Console.WriteLine();

using var classifier = new YamnetClassifier();

try {
    await classifier.InitializeAsync();
}
catch (Exception ex) {
    Console.WriteLine($"\n❌ Failed to load model: {ex.Message}");
    return;
}

Console.WriteLine();

// YAMNet requires: 16kHz sample rate, 15600 samples (~0.975 seconds)
using var audioCapture = new AudioCapture(sampleRate: 16000, samplesNeeded: 15600);


var isProcessing = false;
var lastResults = new List<ClassificationResult>();
float[] prevBuffer = [];

audioCapture.OnAudioReady += waveform => {
    if (isProcessing) return;
    isProcessing = true;

    try {
        if (prevBuffer.Length > 0) {
            var shiftedWave = prevBuffer.Skip(prevBuffer.Length / 2).Concat(waveform.Take(waveform.Length / 2)).ToArray();
            var results2 = classifier.Classify(shiftedWave, topK: 9);
            DisplayResults(results2);
        }
        var results = classifier.Classify(waveform, topK: 9);
        prevBuffer = waveform;
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

try {
    audioCapture.Start();
}
catch {
    Console.WriteLine("\n❌ Could not start audio capture.");
    Console.WriteLine("   Make sure you have a working microphone connected.");
    return;
}

Console.WriteLine("\n" + new string('─', 60));
Console.WriteLine("Press [Enter] to stop, [Q] to quit");
Console.WriteLine(new string('─', 60) + "\n");

while (true) {
    if (Console.KeyAvailable) {
        var key = Console.ReadKey(intercept: true);

        if (key.Key == ConsoleKey.Enter || key.Key == ConsoleKey.Q) {
            break;
        }
    }

    await Task.Delay(50);
}

audioCapture.Stop();
Console.WriteLine("\n✅ Recording stopped.");

void PrintBanner() {
    Console.WriteLine("╔═══════════════════════════════════════════════════════════╗");
    Console.WriteLine("║       🎤 YAMNet Real-Time Audio Classification            ║");
    Console.WriteLine("║          ML.NET + TensorFlow | 521 Sound Classes          ║");
    Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
    Console.WriteLine();
}

void DisplayResults(List<ClassificationResult> results) {
    Console.WriteLine("┌─────────────────────────────────────────────────────────────┐");
    Console.WriteLine($"│  { DateTime.Now:HH:mm:ss} | Current Classifications                         │");
    Console.WriteLine("├─────────────────────────────────────────────────────────────┤");

    foreach (var (result, index) in results.Select((r, i) => (r, i))) {
        var rank = (index + 1).ToString();
        var name = TruncateString(result.ClassName, 22);
        var percentage = (result.Score * 100).ToString("F1").PadLeft(5);
        var barLength = Math.Min((int)(result.Score * 25), 25);
        var bar = new string('█', barLength) + new string('░', 25 - barLength);

        Console.WriteLine($"│  {rank}. {name,-22} {percentage}% {bar} │");
    }

    Console.WriteLine("└─────────────────────────────────────────────────────────────┘");
}

string TruncateString(string str, int maxLength) {
    if (str.Length <= maxLength) return str;
    return str[..(maxLength - 2)] + "..";
}
