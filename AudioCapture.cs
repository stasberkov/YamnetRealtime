using System.Diagnostics;
using System.Runtime.InteropServices;

namespace YamnetRealtime;

/// <summary>
/// Cross-platform audio capture using SoX (Windows, macOS, Linux)
/// </summary>
public class AudioCapture : IDisposable
{
    private readonly int _sampleRate;
    private readonly int _samplesNeeded;
    private readonly List<float> _buffer = new();
    private readonly object _lock = new();
    
    private Process? _soxProcess;
    private bool _isRunning;

    /// <summary>
    /// Fired when enough audio samples are collected for classification
    /// </summary>
    public event Action<float[]>? OnAudioReady;

    /// <summary>
    /// Creates audio capture instance
    /// </summary>
    /// <param name="sampleRate">Sample rate in Hz (YAMNet requires 16000)</param>
    /// <param name="samplesNeeded">Number of samples per chunk (YAMNet needs 15600 for ~0.975s)</param>
    public AudioCapture(int sampleRate = 16000, int samplesNeeded = 15600)
    {
        _sampleRate = sampleRate;
        _samplesNeeded = samplesNeeded;
    }

    /// <summary>
    /// Lists all available audio input devices using SoX
    /// </summary>
    public static void ListDevices()
    {
        Console.WriteLine("Audio capture using SoX");
        Console.WriteLine();
        
        // Check if SoX is installed
        if (!IsSoxInstalled())
        {
            Console.WriteLine("❌ SoX is not installed!");
            Console.WriteLine();
            PrintInstallInstructions();
            return;
        }

        Console.WriteLine("✅ SoX is installed");
        Console.WriteLine();
        Console.WriteLine("Default audio input device will be used.");
        Console.WriteLine("To see available devices, run: sox --help-device");
    }

    private static bool IsSoxInstalled()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "sox",
                Arguments = "--version",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);
            if (process != null)
            {
                process.WaitForExit(5000);
                return process.ExitCode == 0;
            }
        }
        catch { }
        
        return false;
    }

    private static void PrintInstallInstructions()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Console.WriteLine("Install SoX on Windows:");
            Console.WriteLine("  1. Download from: https://sourceforge.net/projects/sox/files/sox/");
            Console.WriteLine("  2. Run the installer");
            Console.WriteLine("  3. Add SoX to PATH or install to C:\\Program Files (x86)\\sox-*");
            Console.WriteLine();
            Console.WriteLine("Or use Chocolatey:");
            Console.WriteLine("  choco install sox");
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            Console.WriteLine("Install SoX on Linux:");
            Console.WriteLine("  Ubuntu/Debian: sudo apt install sox libsox-fmt-all");
            Console.WriteLine("  Fedora:        sudo dnf install sox");
            Console.WriteLine("  Arch:          sudo pacman -S sox");
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            Console.WriteLine("Install SoX on macOS:");
            Console.WriteLine("  brew install sox");
        }
    }

    /// <summary>
    /// Starts audio capture using SoX
    /// </summary>
    public void Start()
    {
        if (!IsSoxInstalled())
        {
            Console.WriteLine("❌ SoX is not installed!");
            PrintInstallInstructions();
            throw new InvalidOperationException("SoX is required for audio capture. Please install it first.");
        }

        _isRunning = true;

        // SoX command: capture from default device, output raw 16-bit signed PCM
        // Platform-specific input:
        //   Windows: -t waveaudio -d (Windows audio device)
        //   macOS:   -t coreaudio default (Core Audio)
        //   Linux:   -t alsa default (ALSA) or -t pulseaudio default
        // Output args:
        //   -r = sample rate
        //   -c 1 = mono
        //   -b 16 = 16-bit
        //   -e signed-integer = signed integer encoding
        //   -q = quiet mode
        //   -p = output to pipe (stdout as raw audio)
        
        string inputArgs = GetPlatformInputArgs();
        string arguments = $"{inputArgs} -r {_sampleRate} -c 1 -b 16 -e signed-integer -q -t raw -";
        
        var psi = new ProcessStartInfo
        {
            FileName = "sox",
            Arguments = arguments,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        try
        {
            Console.WriteLine($"   Command: sox {arguments}");
            
            _soxProcess = Process.Start(psi);

            if (_soxProcess == null)
            {
                throw new InvalidOperationException("Failed to start SoX process");
            }

            // Log errors from SoX
            _soxProcess.ErrorDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data) && _isRunning)
                {
                    Console.WriteLine($"⚠️ SoX: {e.Data}");
                }
            };
            _soxProcess.BeginErrorReadLine();

            Console.WriteLine($"🎤 Recording started");
            Console.WriteLine($"   Format: {_sampleRate}Hz, 16-bit, Mono");

            StartReadingAudio();
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Could not start audio capture: {ex.Message}\n" +
                "Make sure SoX is installed and a microphone is connected.");
        }
    }

    /// <summary>
    /// Gets platform-specific SoX input arguments
    /// </summary>
    private static string GetPlatformInputArgs()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return "-t waveaudio -d";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return "-t coreaudio default";
        }
        else // Linux
        {
            return "-t alsa default";
        }
    }

    private void StartReadingAudio()
    {
        if (_soxProcess == null) return;

        Task.Run(async () =>
        {
            var stream = _soxProcess.StandardOutput.BaseStream;
            var buffer = new byte[4096];

            while (_isRunning && !_soxProcess.HasExited)
            {
                try
                {
                    var bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length);

                    if (bytesRead > 0)
                    {
                        ProcessAudioData(buffer, bytesRead);
                    }
                    else if (bytesRead == 0)
                    {
                        // End of stream
                        await Task.Delay(10);
                    }
                }
                catch (Exception ex)
                {
                    if (_isRunning)
                    {
                        Console.WriteLine($"❌ Audio read error: {ex.Message}");
                    }
                    break;
                }
            }
        });
    }

    private void ProcessAudioData(byte[] buffer, int bytesRead)
    {
        // Convert 16-bit PCM bytes to float samples [-1.0, 1.0]
        var samples = new float[bytesRead / 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short sample = BitConverter.ToInt16(buffer, i * 2);
            samples[i] = sample / 32768f;
        }

        lock (_lock)
        {
            _buffer.AddRange(samples);

            // Emit chunks when we have enough samples
            while (_buffer.Count >= _samplesNeeded)
            {
                var chunk = _buffer.Take(_samplesNeeded).ToArray();
                _buffer.RemoveRange(0, _samplesNeeded);
                OnAudioReady?.Invoke(chunk);
            }
        }
    }

    /// <summary>
    /// Stops audio capture
    /// </summary>
    public void Stop()
    {
        _isRunning = false;

        if (_soxProcess != null && !_soxProcess.HasExited)
        {
            try
            {
                // Send Ctrl+C signal on Unix, kill on Windows
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    _soxProcess.Kill();
                }
                else
                {
                    // Try graceful shutdown first
                    Process.Start("kill", $"-SIGINT {_soxProcess.Id}")?.WaitForExit(100);
                }
                
                _soxProcess.WaitForExit(1000);
                
                if (!_soxProcess.HasExited)
                {
                    _soxProcess.Kill();
                }
            }
            catch { }
        }

        _soxProcess?.Dispose();
        _soxProcess = null;
    }

    public void Dispose()
    {
        Stop();
    }
}
