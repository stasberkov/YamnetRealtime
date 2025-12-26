# YAMNet Real-Time Audio Classification

Real-time audio classification using YAMNet model with ML.NET and TensorFlow.

Classifies audio into **521 sound categories** from the [AudioSet](https://research.google.com/audioset/) ontology including:
- Speech, Music, Singing
- Animals (Dog, Cat, Bird, etc.)
- Vehicles (Car, Train, etc.)
- Environment (Rain, Wind, etc.)
- And many more...

## Requirements

- .NET 8.0 SDK
- Windows (NAudio for audio capture)
- Microphone

## Quick Start

### 1. Clone/Download the project

```bash
cd YamnetRealtime
```

### 2. Download YAMNet Model

**Option A: Using curl/wget**

```bash
# Download compressed model from TensorFlow Hub
curl -L "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed" -o yamnet.tar.gz

# Create directory and extract
mkdir yamnet_model
tar -xzf yamnet.tar.gz -C yamnet_model
```

**Option B: Using PowerShell (Windows)**

```powershell
# Download
Invoke-WebRequest -Uri "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed" -OutFile yamnet.tar.gz

# Extract (requires 7-Zip or tar)
mkdir yamnet_model
tar -xzf yamnet.tar.gz -C yamnet_model
```

**Option C: Manual Download**

1. Go to: https://tfhub.dev/google/yamnet/1
2. Click "Download" button
3. Extract the downloaded archive to `yamnet_model` folder

### 3. Verify Model Structure

After extraction, you should have:

```
YamnetRealtime/
├── yamnet_model/
│   ├── saved_model.pb
│   ├── variables/
│   │   ├── variables.data-00000-of-00001
│   │   └── variables.index
│   └── assets/ (may be empty)
├── YamnetRealtime.csproj
├── Program.cs
├── AudioCapture.cs
├── YamnetClassifier.cs
└── README.md
```

### 4. Build and Run

```bash
dotnet restore
dotnet build
dotnet run
```

## Usage

1. Run the application
2. Allow microphone access if prompted
3. Make sounds near your microphone
4. Watch real-time classifications

### Controls

- **Enter** - Stop recording and exit
- **S** - Save current detection to file
- **Q** - Quit

## Output Example

```
╔═══════════════════════════════════════════════════════════╗
║       🎤 YAMNet Real-Time Audio Classification            ║
║          ML.NET + TensorFlow | 521 Sound Classes          ║
╚═══════════════════════════════════════════════════════════╝

Available audio input devices:
  [0] Microphone (Realtek Audio) (Channels: 2)

Loading YAMNet model...
✅ YAMNet model loaded successfully
✅ Loaded 521 class labels

🎤 Recording from: Microphone (Realtek Audio)
   Format: 16000Hz, 16-bit, Mono

┌─────────────────────────────────────────────────────────┐
│  Current Classifications                                │
├─────────────────────────────────────────────────────────┤
│  1. Speech                  92.3% █████████████████████████ │
│  2. Narration, monologue    45.2% ███████████░░░░░░░░░░░░░░ │
│  3. Conversation            38.1% █████████░░░░░░░░░░░░░░░░ │
│  4. Male speech             22.4% █████░░░░░░░░░░░░░░░░░░░░ │
│  5. Music                    8.3% ██░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────────────────────────────────────────────┤
│  14:32:15 | Press Enter to stop                         │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Model not loading

```
❌ Failed to load model: Model directory not found
```

**Solution:** Make sure `yamnet_model` directory exists and contains `saved_model.pb`

### No audio devices found

```
Available audio input devices:
  No audio input devices found!
```

**Solution:** 
- Check microphone is connected
- Check Windows sound settings
- Try a different microphone

### TensorFlow DLL errors

```
DllNotFoundException: tensorflow
```

**Solution:** Add explicit TensorFlow native package:

```bash
dotnet add package SciSharp.TensorFlow.Redist --version 2.16.0
```

### Wrong tensor names

If you see errors about tensor names, the model export may use different names. Check the console output for detected schema and update `YamnetClassifier.cs` accordingly.

## Customization

### Filter specific sounds

```csharp
// In Program.cs, modify the OnAudioReady handler:
audioCapture.OnAudioReady += waveform =>
{
    var results = classifier.Classify(waveform, topK: 10);
    
    // Filter for specific sounds
    var coughDetected = results.Any(r => 
        r.ClassName.Contains("Cough", StringComparison.OrdinalIgnoreCase) && 
        r.Score > 0.3f);
    
    if (coughDetected)
    {
        Console.WriteLine("🔔 COUGH DETECTED!");
        // Trigger action...
    }
};
```

### Adjust sensitivity

Change the `samplesNeeded` parameter for different window sizes:
- 15600 samples = ~0.975s (default, best for YAMNet)
- Smaller values = faster but less accurate
- Larger values = more context but higher latency

### Log to file

```csharp
// Add to classification results
File.AppendAllText("detections.log", 
    $"{DateTime.Now:O}|{results[0].ClassName}|{results[0].Score:F3}\n");
```

## Technical Details

- **Model:** YAMNet (Yet Another Mobile Network)
- **Input:** 16kHz mono audio, ~0.975 seconds (15600 samples)
- **Output:** 521 class probabilities
- **Architecture:** MobileNet v1 backbone
- **Training Data:** AudioSet (2+ million YouTube clips)

## License

YAMNet model is released under Apache 2.0 license by Google.

## References

- [YAMNet on TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
- [AudioSet](https://research.google.com/audioset/)
- [YAMNet Paper](https://arxiv.org/abs/1609.04243)
