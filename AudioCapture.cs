using NAudio.Wave;

namespace YamnetRealtime;

/// <summary>
/// Captures audio from microphone and emits chunks for processing
/// </summary>
public class AudioCapture : IDisposable {
    private readonly WaveInEvent _waveIn;
    private readonly List<float> _buffer = new();
    private readonly object _lock = new();
    private readonly int _samplesNeeded;

    /// <summary>
    /// Fired when enough audio samples are collected for classification
    /// </summary>
    public event Action<float[]>? OnAudioReady;

    /// <summary>
    /// Creates audio capture instance
    /// </summary>
    /// <param name="sampleRate">Sample rate in Hz (YAMNet requires 16000)</param>
    /// <param name="samplesNeeded">Number of samples per chunk (YAMNet needs 15600 for ~0.975s)</param>
    public AudioCapture(int sampleRate = 16000, int samplesNeeded = 15600) {
        _samplesNeeded = samplesNeeded;

        _waveIn = new WaveInEvent {
            WaveFormat = new WaveFormat(sampleRate, 16, 1),
            BufferMilliseconds = 100
        };

        _waveIn.DataAvailable += OnDataAvailable;
        _waveIn.RecordingStopped += OnRecordingStopped;
    }

    /// <summary>
    /// Lists all available audio input devices
    /// </summary>
    public static void ListDevices() {
        Console.WriteLine("Available audio input devices:");

        if (WaveInEvent.DeviceCount == 0) {
            Console.WriteLine("  No audio input devices found!");
            return;
        }

        for (int i = 0; i < WaveInEvent.DeviceCount; i++) {
            var caps = WaveInEvent.GetCapabilities(i);
            Console.WriteLine($"  [{i}] {caps.ProductName} (Channels: {caps.Channels})");
        }
    }

    /// <summary>
    /// Sets the audio input device by index
    /// </summary>
    public void SetDevice(int deviceNumber) {
        if (deviceNumber < 0 || deviceNumber >= WaveInEvent.DeviceCount) {
            throw new ArgumentException($"Invalid device number. Must be 0-{WaveInEvent.DeviceCount - 1}");
        }
        _waveIn.DeviceNumber = deviceNumber;
    }

    /// <summary>
    /// Starts audio capture
    /// </summary>
    public void Start() {
        try {
            _waveIn.StartRecording();
            var deviceName = WaveInEvent.GetCapabilities(_waveIn.DeviceNumber).ProductName;
            Console.WriteLine($"🎤 Recording from: {deviceName}");
            Console.WriteLine($"   Format: {_waveIn.WaveFormat.SampleRate}Hz, {_waveIn.WaveFormat.BitsPerSample}-bit, Mono");
        }
        catch (Exception ex) {
            Console.WriteLine($"❌ Failed to start recording: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Stops audio capture
    /// </summary>
    public void Stop() {
        _waveIn.StopRecording();
    }

    private void OnDataAvailable(object? sender, WaveInEventArgs e) {
        // Convert 16-bit PCM bytes to float samples [-1.0, 1.0]
        var samples = new float[e.BytesRecorded / 2];
        for (int i = 0; i < samples.Length; i++) {
            short sample = BitConverter.ToInt16(e.Buffer, i * 2);
            samples[i] = sample / 32768f;
        }

        lock (_lock) {
            _buffer.AddRange(samples);

            // Emit chunks when we have enough samples
            while (_buffer.Count >= _samplesNeeded) {
                var chunk = _buffer.Take(_samplesNeeded).ToArray();
                _buffer.RemoveRange(0, _samplesNeeded);
                OnAudioReady?.Invoke(chunk);
            }
        }
    }

    private void OnRecordingStopped(object? sender, StoppedEventArgs e) {
        if (e.Exception != null) {
            Console.WriteLine($"❌ Recording error: {e.Exception.Message}");
        }
    }

    public void Dispose() {
        _waveIn.StopRecording();
        _waveIn.Dispose();
    }
}
