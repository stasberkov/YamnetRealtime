#!/usr/bin/env node

/**
 * setup.mjs - Download YAMNet model and setup .NET project
 * Run: node setup.mjs
 */

import { execSync } from 'child_process';
import { createWriteStream, existsSync, mkdirSync, unlinkSync } from 'fs';
import https from 'https';
import path from 'path';

const MODEL_DIR = 'yamnet_model';
const MODEL_ARCHIVE = 'yamnet.tar.gz';
const MODEL_URL = 'https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed';
const CLASS_MAP_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv';
const CLASS_MAP_FILE = 'yamnet_class_map.csv';

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  gray: '\x1b[90m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logSuccess(message) {
  console.log(`  ${colors.green}✅ ${message}${colors.reset}`);
}

function logError(message) {
  console.log(`  ${colors.red}❌ ${message}${colors.reset}`);
}

function logWarning(message) {
  console.log(`  ${colors.yellow}⚠️ ${message}${colors.reset}`);
}

/**
 * Execute shell command and return output
 */
function execCommand(command, silent = false) {
  try {
    const output = execSync(command, { encoding: 'utf8', stdio: silent ? 'pipe' : 'inherit' });
    return { success: true, output };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Download file with progress indication
 */
function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    log(`  URL: ${url}`, 'gray');
    
    const file = createWriteStream(destPath);
    
    const request = https.get(url, { 
      headers: { 'User-Agent': 'Mozilla/5.0' }
    }, (response) => {
      // Handle redirects
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        file.close();
        unlinkSync(destPath);
        return downloadFile(response.headers.location, destPath).then(resolve).catch(reject);
      }
      
      if (response.statusCode !== 200) {
        file.close();
        unlinkSync(destPath);
        return reject(new Error(`HTTP ${response.statusCode}`));
      }
      
      const totalSize = parseInt(response.headers['content-length'], 10);
      let downloadedSize = 0;
      
      response.on('data', (chunk) => {
        downloadedSize += chunk.length;
        if (totalSize) {
          const percent = ((downloadedSize / totalSize) * 100).toFixed(1);
          process.stdout.write(`\r  Downloading: ${percent}% (${(downloadedSize / 1024 / 1024).toFixed(1)} MB)`);
        }
      });
      
      response.pipe(file);
      
      file.on('finish', () => {
        file.close();
        console.log('');
        resolve();
      });
    });
    
    request.on('error', (err) => {
      file.close();
      if (existsSync(destPath)) unlinkSync(destPath);
      reject(err);
    });
    
    request.setTimeout(60000, () => {
      request.destroy();
      reject(new Error('Download timeout'));
    });
  });
}

/**
 * Extract tar.gz archive using system command
 */
function extractTarGz(archivePath, destDir) {
  if (!existsSync(destDir)) {
    mkdirSync(destDir, { recursive: true });
  }
  
  // Use system tar command (available on Windows 10+, macOS, Linux)
  const result = execCommand(`tar -xzf "${archivePath}" -C "${destDir}"`, true);
  
  if (!result.success) {
    throw new Error(result.error || 'tar extraction failed');
  }
}

/**
 * Main setup function
 */
async function main() {
  console.log('');
  log('═══════════════════════════════════════════════════════════', 'cyan');
  log('  YAMNet Real-Time Audio Classification - Setup Script', 'cyan');
  log('═══════════════════════════════════════════════════════════', 'cyan');
  console.log('');

  // Check .NET SDK
  log('Checking .NET SDK...', 'yellow');
  const dotnetResult = execCommand('dotnet --version', true);
  
  if (dotnetResult.success) {
    logSuccess(`NET SDK found: ${dotnetResult.output.trim()}`);
  } else {
    logError('.NET SDK not found');
    console.log('  Please install from: https://dotnet.microsoft.com/download');
    process.exit(1);
  }

  // Restore NuGet packages
  console.log('');
  log('Restoring NuGet packages...', 'yellow');
  const restoreResult = execCommand('dotnet restore');
  
  if (restoreResult.success) {
    logSuccess('Packages restored successfully');
  } else {
    logError('Failed to restore packages');
    process.exit(1);
  }

  // Download YAMNet model
  console.log('');
  if (existsSync(MODEL_DIR) && existsSync(path.join(MODEL_DIR, 'saved_model.pb'))) {
    log('Model directory already exists. Skipping download.', 'yellow');
  } else {
    log('Downloading YAMNet model from TensorFlow Hub...', 'yellow');
    
    try {
      await downloadFile(MODEL_URL, MODEL_ARCHIVE);
      logSuccess('Download complete');
      
      // Extract
      console.log('');
      log('Extracting model...', 'yellow');
      
      try {
        extractTarGz(MODEL_ARCHIVE, MODEL_DIR);
        logSuccess('Extraction complete');
        
        // Clean up archive
        unlinkSync(MODEL_ARCHIVE);
      } catch (extractErr) {
        logError(`Extraction failed: ${extractErr.message}`);
        console.log('  Please extract manually:');
        console.log(`    tar -xzf ${MODEL_ARCHIVE} -C ${MODEL_DIR}`);
        process.exit(1);
      }
    } catch (downloadErr) {
      logError(`Download failed: ${downloadErr.message}`);
      console.log('');
      log('Please download manually:', 'yellow');
      console.log('  1. Go to: https://tfhub.dev/google/yamnet/1');
      console.log('  2. Click "Download"');
      console.log(`  3. Extract to '${MODEL_DIR}' folder`);
      process.exit(1);
    }
  }

  // Verify model files
  console.log('');
  log('Verifying model files...', 'yellow');
  
  const requiredFiles = [
    path.join(MODEL_DIR, 'saved_model.pb')
  ];
  
  let allFilesExist = true;
  for (const file of requiredFiles) {
    if (existsSync(file)) {
      logSuccess(file);
    } else {
      logError(`${file} (missing)`);
      allFilesExist = false;
    }
  }
  
  if (!allFilesExist) {
    console.log('');
    logError('Some model files are missing. Please re-download the model.');
    process.exit(1);
  }

  // Download class map
  if (!existsSync(CLASS_MAP_FILE)) {
    console.log('');
    log('Downloading class map...', 'yellow');
    
    try {
      await downloadFile(CLASS_MAP_URL, CLASS_MAP_FILE);
      logSuccess('Class map downloaded');
    } catch (err) {
      logWarning('Could not download class map (will be downloaded at runtime)');
    }
  }

  // Build project
  console.log('');
  log('Building project...', 'yellow');
  const buildResult = execCommand('dotnet build --configuration Release');
  
  if (buildResult.success) {
    logSuccess('Build successful');
  } else {
    logError('Build failed');
    process.exit(1);
  }

  // Done
  console.log('');
  log('═══════════════════════════════════════════════════════════', 'cyan');
  log('  Setup complete! Run the application with:', 'cyan');
  log('    dotnet run', 'reset');
  log('═══════════════════════════════════════════════════════════', 'cyan');
  console.log('');
}

// Run
main().catch((err) => {
  logError(`Setup failed: ${err.message}`);
  process.exit(1);
});
