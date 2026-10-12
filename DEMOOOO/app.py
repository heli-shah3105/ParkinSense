"""
=============================================================================
VocalAid AI — Parkinson's Dysarthric Speech-to-Text Interactive Web Platform
=============================================================================
A local interactive web platform for transcribing and analyzing speech from
individuals with Parkinson's Disease and dysarthria.

Features:
  - Real-time transcription using Wav2Vec2 + LoRA (Generalized & Personalized)
  - Automatic Smart Adapter Routing (Unseen patient vs. Personal adapter)
  - Word-level timestamps with clickable audio seeking
  - Clinical Speech Pathology Analytics (WPM, pause ratio, articulation score)
  - Interactive Microphone recording and Audio file upload
  - Built-in dataset sample player to test without external files
  - Premium dark-mode glassmorphic UI with responsive accessibility

Usage:
  python app.py                 # Launches server at http://localhost:8080
  python app.py --port 8000     # Custom port
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import base64
import cgi
import glob
import io
import json
import logging
import mimetypes
import urllib.parse
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from config import (
    BASE_MODEL_NAME, SAMPLE_RATE, ADAPTER_DIR,
    MANIFEST_DIR, PROCESSED_AUDIO_DIR, SPEAKER_SEVERITY,
    LOGS_DIR
)
from step6_inference import DysarthricSpeechTranscriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global model instance (lazy loaded)
_transcriber: Optional[DysarthricSpeechTranscriber] = None


def get_transcriber() -> DysarthricSpeechTranscriber:
    global _transcriber
    if _transcriber is None:
        logger.info("Initializing DysarthricSpeechTranscriber for Web Platform...")
        _transcriber = DysarthricSpeechTranscriber()
    return _transcriber


# ─────────────────────────────────────────────────────────────────────────────
# HTML / CSS / JS Frontend
# ─────────────────────────────────────────────────────────────────────────────

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VocalAid AI — Parkinson's Dysarthric Speech Recognition</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0e17;
      --bg-surface: #111827;
      --bg-card: rgba(17, 24, 39, 0.75);
      --bg-card-hover: rgba(31, 41, 55, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(56, 189, 248, 0.3);
      --primary: #38bdf8;
      --primary-gradient: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
      --accent-glow: rgba(14, 165, 233, 0.15);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.08) 0%, transparent 40%);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Navigation Bar */
    header {
      background: rgba(10, 14, 23, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .brand-icon {
      width: 40px;
      height: 40px;
      border-radius: var(--radius-sm);
      background: var(--primary-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
      box-shadow: 0 0 20px rgba(14, 165, 233, 0.35);
    }

    .brand-title {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-badge {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2rem 0.6rem;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid var(--border-accent);
      color: var(--primary);
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .status-indicator {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 10px var(--success);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.9); }
    }

    /* Layout */
    .app-container {
      max-width: 1380px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 2rem;
      flex: 1;
    }

    @media (max-width: 980px) {
      .app-container {
        grid-template-columns: 1fr;
      }
    }

    /* Cards */
    .card {
      background: var(--bg-card);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 1.75rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
      position: relative;
      overflow: hidden;
    }

    .card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0; height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
    }

    .card-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Form Controls */
    .form-group {
      margin-bottom: 1.25rem;
    }

    .form-label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }

    select, input[type="file"] {
      width: 100%;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      color: var(--text-main);
      padding: 0.75rem 1rem;
      font-family: var(--font-sans);
      font-size: 0.9rem;
      outline: none;
      transition: all 0.2s ease;
    }

    select:focus, input[type="file"]:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2);
    }

    /* Severity badges */
    .badge {
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .badge-severe { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-moderate { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-mild { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-control { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

    /* Audio Source Tabs */
    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 0.35rem;
      background: rgba(15, 23, 42, 0.6);
      padding: 0.3rem;
      border-radius: var(--radius-sm);
      margin-bottom: 1.25rem;
      border: 1px solid var(--border-subtle);
    }

    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.5rem;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: center;
    }

    .tab-btn.active {
      background: var(--primary-gradient);
      color: #fff;
      box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
    }

    /* Record Button Area */
    .record-box {
      border: 2px dashed var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.5rem;
      text-align: center;
      transition: all 0.2s ease;
      background: rgba(15, 23, 42, 0.3);
    }

    .record-btn {
      width: 64px;
      height: 64px;
      border-radius: 50%;
      background: rgba(239, 68, 68, 0.15);
      border: 2px solid var(--danger);
      color: var(--danger);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s ease;
      margin-bottom: 0.75rem;
    }

    .record-btn.recording {
      background: var(--danger);
      color: #fff;
      box-shadow: 0 0 24px rgba(239, 68, 68, 0.6);
      animation: record-pulse 1s infinite alternate;
    }

    @keyframes record-pulse {
      0% { transform: scale(1); }
      100% { transform: scale(1.1); }
    }

    /* Primary Action Button */
    .btn-primary {
      width: 100%;
      background: var(--primary-gradient);
      color: #fff;
      border: none;
      border-radius: var(--radius-sm);
      padding: 0.9rem 1.5rem;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3);
    }

    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(14, 165, 233, 0.45);
    }

    .btn-primary:active {
      transform: translateY(0);
    }

    .btn-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    /* Audio Player */
    audio {
      width: 100%;
      margin: 1rem 0;
      border-radius: var(--radius-sm);
      filter: invert(0.85) hue-rotate(180deg);
    }

    /* Output Results Area */
    .transcription-box {
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.5rem;
      min-height: 140px;
      font-size: 1.35rem;
      font-weight: 500;
      letter-spacing: -0.01em;
      color: #fff;
      position: relative;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
    }

    .transcription-placeholder {
      color: var(--text-dim);
      font-size: 1rem;
      font-style: italic;
    }

    /* Metrics Grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    @media (max-width: 768px) {
      .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    .metric-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1rem;
      text-align: center;
      transition: all 0.2s ease;
    }

    .metric-card:hover {
      border-color: var(--border-accent);
      background: rgba(15, 23, 42, 0.8);
    }

    .metric-value {
      font-size: 1.5rem;
      font-weight: 800;
      font-family: var(--font-mono);
      background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.25rem;
    }

    .metric-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    /* Word Timestamps Explorer */
    .word-cloud {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      padding: 1rem;
      background: rgba(15, 23, 42, 0.5);
      border-radius: var(--radius-md);
      border: 1px solid var(--border-subtle);
      max-height: 220px;
      overflow-y: auto;
    }

    .word-pill {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.85rem;
      font-family: var(--font-mono);
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .word-pill:hover {
      background: rgba(56, 189, 248, 0.2);
      border-color: var(--primary);
      transform: translateY(-2px);
    }

    .word-pill .time {
      font-size: 0.7rem;
      color: var(--text-dim);
    }

    /* Spinner */
    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-top: 3px solid var(--primary);
      border-radius: 50%;
      width: 20px;
      height: 20px;
      animation: spin 0.8s linear infinite;
      display: none;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">🎙️</div>
      <div>
        <div class="brand-title">VocalAid AI</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">Parkinson's Dysarthric Speech Recognition</div>
      </div>
      <div class="brand-badge">Wav2Vec2 + LoRA</div>
    </div>
    <div class="status-indicator">
      <div class="status-dot"></div>
      <span id="systemStatus">Engine Ready (16 kHz Mono)</span>
    </div>
  </header>

  <div class="app-container">
    
    <!-- LEFT PANEL: CONTROLS -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">⚙️ Adapter & Audio Control</div>
      </div>

      <!-- Adapter Selector -->
      <div class="form-group">
        <label class="form-label">Speaker / Adapter Target</label>
        <select id="speakerSelect" onchange="onSpeakerChange()">
          <option value="">Auto-Route (Generalized Adapter - Unseen Speaker)</option>
        </select>
        <div id="speakerInfo" style="margin-top: 0.4rem; font-size: 0.8rem; color: var(--text-dim);"></div>
      </div>

      <!-- Audio Source Selection Tabs -->
      <div class="tabs">
        <button class="tab-btn active" onclick="setTab('dataset')">Dataset</button>
        <button class="tab-btn" onclick="setTab('upload')">Upload</button>
        <button class="tab-btn" onclick="setTab('mic')">Mic Record</button>
      </div>

      <!-- TAB 1: DATASET SAMPLES -->
      <div id="tabDataset" class="tab-content">
        <div class="form-group">
          <label class="form-label">Preprocessed Sample Audio</label>
          <select id="sampleSelect" onchange="onSampleChange()">
            <option value="">Loading samples...</option>
          </select>
        </div>
      </div>

      <!-- TAB 2: UPLOAD FILE -->
      <div id="tabUpload" class="tab-content" style="display: none;">
        <div class="form-group">
          <label class="form-label">Select Audio File (.wav, .mp3)</label>
          <input type="file" id="audioFileInput" accept="audio/*" onchange="onFileSelected(event)">
        </div>
      </div>

      <!-- TAB 3: MICROPHONE RECORD -->
      <div id="tabMic" class="tab-content" style="display: none;">
        <div class="record-box">
          <button id="micBtn" class="record-btn" onclick="toggleRecording()">
            🎤
          </button>
          <div id="micStatus" style="font-size: 0.85rem; font-weight: 600;">Click to Record</div>
          <div style="font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem;">Speak clearly into your microphone</div>
        </div>
      </div>

      <!-- Audio Player -->
      <audio id="audioPlayer" controls style="display: none;"></audio>

      <!-- Submit Button -->
      <button id="transcribeBtn" class="btn-primary" onclick="transcribeAudio()">
        <span class="spinner" id="loadingSpinner"></span>
        <span id="btnText">Transcribe Dysarthric Speech</span>
      </button>

      <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--border-subtle); font-size: 0.75rem; color: var(--text-dim);">
        <strong>Architecture:</strong> Pretrained Wav2Vec2 base with LoRA rank-16 adapters injected in self-attention & feedforward projection layers.
      </div>
    </div>

    <!-- RIGHT PANEL: TRANSCRIPTION & CLINICAL DASHBOARD -->
    <div>
      
      <!-- Primary Transcription Card -->
      <div class="card" style="margin-bottom: 2rem;">
        <div class="card-header">
          <div class="card-title">📝 Decoded Transcription</div>
          <button style="background: transparent; border: 1px solid var(--border-subtle); color: var(--text-muted); padding: 0.3rem 0.75rem; border-radius: 6px; font-size: 0.75rem; cursor: pointer;" onclick="copyTranscription()">
            Copy Text
          </button>
        </div>

        <div class="transcription-box" id="transcriptionDisplay">
          <span class="transcription-placeholder">Transcription output will appear here after analysis...</span>
        </div>
        <div id="rawBox" style="display: none; margin-top: 0.6rem; padding: 0.5rem 0.8rem; background: rgba(0,0,0,0.25); border-radius: 6px; font-size: 0.8rem; color: var(--text-dim);">
          <span style="font-weight: 600; color: var(--warning);">⚡ Raw CTC Output:</span> <span id="rawTranscriptionDisplay" style="font-family: var(--font-mono); color: var(--text-main);"></span>
        </div>

        <!-- Clinical Speech Analytics Grid -->
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-value" id="valConfidence">--%</div>
            <div class="metric-label">Confidence</div>
          </div>
          <div class="metric-card">
            <div class="metric-value" id="valDuration">--s</div>
            <div class="metric-label">Duration</div>
          </div>
          <div class="metric-card">
            <div class="metric-value" id="valWpm">--</div>
            <div class="metric-label">Words / Min</div>
          </div>
          <div class="metric-card">
            <div class="metric-value" id="valAdapter">--</div>
            <div class="metric-label">Adapter Used</div>
          </div>
        </div>

        <!-- Word-Level Timestamps -->
        <div class="card-title" style="margin-bottom: 0.75rem; font-size: 0.95rem;">
          ⏱️ Word Boundaries & Acoustic Timestamps
        </div>
        <div class="word-cloud" id="wordCloud">
          <span style="color: var(--text-dim); font-size: 0.85rem;">Word timestamps will appear here. Click any word to jump audio directly to that syllable.</span>
        </div>
      </div>

    </div>

  </div>

  <script>
    let activeTab = 'dataset';
    let mediaRecorder = null;
    let audioChunks = [];
    let currentAudioBlob = null;
    let speakersList = [];
    let samplesList = [];

    // Load available speakers & samples on load
    window.addEventListener('DOMContentLoaded', () => {
      fetchAdapters();
      fetchSamples();
    });

    function setTab(tab) {
      activeTab = tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');

      document.getElementById('tabDataset').style.display = tab === 'dataset' ? 'block' : 'none';
      document.getElementById('tabUpload').style.display = tab === 'upload' ? 'block' : 'none';
      document.getElementById('tabMic').style.display = tab === 'mic' ? 'block' : 'none';
    }

    async function fetchAdapters() {
      try {
        const res = await fetch('/api/adapters');
        const data = await res.json();
        const select = document.getElementById('speakerSelect');

        select.innerHTML = `
          <option value="">Auto-Route (Generalized Adapter - Unseen Dysarthric Speaker)</option>
          <option value="__base__">Base Wav2Vec2 (Standard / Healthy Control Speech)</option>
        `;

        if (data.speaker_adapters) {
          for (const [spk, meta] of Object.entries(data.speaker_adapters)) {
            const opt = document.createElement('option');
            opt.value = spk;
            opt.textContent = `${spk} [Severity: ${meta.severity}]`;
            select.appendChild(opt);
          }
        }
      } catch (err) {
        console.error('Failed to load adapters:', err);
      }
    }

    async function fetchSamples() {
      try {
        const res = await fetch('/api/sample_audios');
        samplesList = await res.json();
        const select = document.getElementById('sampleSelect');
        select.innerHTML = '';

        if (samplesList.length === 0) {
          select.innerHTML = '<option value="">No samples found</option>';
          return;
        }

        samplesList.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.path;
          opt.textContent = `[${s.speaker}] ${s.filename} (${s.severity || 'N/A'})`;
          select.appendChild(opt);
        });

        onSampleChange();
      } catch (err) {
        console.error('Failed to load samples:', err);
      }
    }

    function onSpeakerChange() {
      const spk = document.getElementById('speakerSelect').value;
      const info = document.getElementById('speakerInfo');
      if (spk === '__base__') {
        info.innerHTML = `Base Wav2Vec2-960h active — recommended when recording standard / healthy speech.`;
      } else if (spk) {
        info.innerHTML = `Personalized LoRA adapter selected for <strong>${spk}</strong>.`;
      } else {
        info.innerHTML = `Generalized adapter active — optimal for Parkinson's dysarthric speech.`;
      }
    }

    function onSampleChange() {
      const samplePath = document.getElementById('sampleSelect').value;
      if (!samplePath) return;

      const player = document.getElementById('audioPlayer');
      player.src = `/api/audio?path=${encodeURIComponent(samplePath)}`;
      player.style.display = 'block';
    }

    function onFileSelected(e) {
      const file = e.target.files[0];
      if (!file) return;

      currentAudioBlob = file;
      currentFileName = file.name;
      const player = document.getElementById('audioPlayer');
      player.src = URL.createObjectURL(file);
      player.style.display = 'block';
    }

    // Web Audio API State for 16kHz PCM WAV Recording
    let audioCtx = null;
    let micStream = null;
    let processorNode = null;
    let inputSource = null;
    let pcmBuffers = [];
    let recordingLength = 0;
    let isRecording = false;
    let currentFileName = null;

    async function toggleRecording() {
      const btn = document.getElementById('micBtn');
      const status = document.getElementById('micStatus');

      if (isRecording) {
        stopRecording();
        btn.classList.remove('recording');
        status.textContent = 'Processing Recording...';
        return;
      }

      try {
        await startRecording();
        btn.classList.add('recording');
        status.textContent = 'Recording... Click to Stop';
      } catch (err) {
        alert('Microphone access denied: ' + err.message);
      }
    }

    async function startRecording() {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContextClass();
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      inputSource = audioCtx.createMediaStreamSource(micStream);

      processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
      pcmBuffers = [];
      recordingLength = 0;

      processorNode.onaudioprocess = (e) => {
        if (!isRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        pcmBuffers.push(new Float32Array(channelData));
        recordingLength += channelData.length;
      };

      inputSource.connect(processorNode);
      processorNode.connect(audioCtx.destination);
      isRecording = true;
    }

    function stopRecording() {
      isRecording = false;
      const status = document.getElementById('micStatus');

      if (processorNode) { processorNode.disconnect(); processorNode = null; }
      if (inputSource) { inputSource.disconnect(); inputSource = null; }
      if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }

      const origRate = audioCtx ? audioCtx.sampleRate : 16000;
      if (audioCtx && audioCtx.state !== 'closed') { audioCtx.close(); audioCtx = null; }

      if (recordingLength === 0) {
        status.textContent = 'No audio detected. Click to Record.';
        return;
      }

      const merged = new Float32Array(recordingLength);
      let offset = 0;
      for (const b of pcmBuffers) {
        merged.set(b, offset);
        offset += b.length;
      }

      // Encode at native sample rate. Python librosa will perform band-limited, anti-aliased resampling.
      currentAudioBlob = encodeWAV(merged, origRate);
      currentFileName = 'mic_recording.wav';

      const player = document.getElementById('audioPlayer');
      player.src = URL.createObjectURL(currentAudioBlob);
      player.style.display = 'block';
      status.textContent = 'Recording Complete. Ready to Transcribe!';
    }

    function downsampleBuffer(buffer, sampleRate, outSampleRate) {
      if (outSampleRate === sampleRate) return buffer;
      const sampleRatio = sampleRate / outSampleRate;
      const newLength = Math.round(buffer.length / sampleRatio);
      const result = new Float32Array(newLength);
      let offsetResult = 0;
      let offsetBuffer = 0;
      while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
          accum += buffer[i];
          count++;
        }
        result[offsetResult] = count > 0 ? accum / count : 0;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
      }
      return result;
    }

    function encodeWAV(samples, sampleRate) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);

      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(view, 8, 'WAVE');

      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);

      writeString(view, 36, 'data');
      view.setUint32(40, samples.length * 2, true);

      let offset = 44;
      for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
      }

      return new Blob([view], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }

    async function transcribeAudio() {
      const btn = document.getElementById('transcribeBtn');
      const spinner = document.getElementById('loadingSpinner');
      const btnText = document.getElementById('btnText');
      const speakerId = document.getElementById('speakerSelect').value;

      btn.disabled = true;
      spinner.style.display = 'inline-block';
      btnText.textContent = 'Decoding Dysarthric Speech...';

      try {
        let res;

        if (activeTab === 'dataset') {
          const samplePath = document.getElementById('sampleSelect').value;
          if (!samplePath) {
            alert('Please select a dataset sample first.');
            return;
          }
          res = await fetch('/api/transcribe_sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_path: samplePath, speaker_id: speakerId })
          });
        } else {
          if (!currentAudioBlob) {
            alert('Please select or record an audio file first.');
            return;
          }

          const base64Audio = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
              const result = reader.result;
              const commaIdx = result.indexOf(',');
              resolve(commaIdx >= 0 ? result.slice(commaIdx + 1) : result);
            };
            reader.onerror = () => reject(new Error('Failed to read audio data'));
            reader.readAsDataURL(currentAudioBlob);
          });

          res = await fetch('/api/transcribe_upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audio_base64: base64Audio,
              speaker_id: speakerId,
              filename: currentFileName || 'audio.wav'
            })
          });
        }

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || `Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        renderResults(data);
      } catch (err) {
        alert('Transcription failed: ' + err.message);
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Transcribe Dysarthric Speech';
      }
    }

    function renderResults(data) {
      document.getElementById('transcriptionDisplay').textContent = data.transcription || '[No speech recognized]';
      document.getElementById('valConfidence').textContent = `${Math.round(data.confidence * 100)}%`;
      document.getElementById('valDuration').textContent = `${data.duration_sec}s`;
      document.getElementById('valAdapter').textContent = data.adapter_used || 'Base Model';

      // Raw vs Rescored display
      const rawBox = document.getElementById('rawBox');
      const rawDisplay = document.getElementById('rawTranscriptionDisplay');
      if (rawBox && rawDisplay) {
        if (data.raw_transcription && data.raw_transcription.toLowerCase() !== (data.transcription || '').toLowerCase()) {
          rawBox.style.display = 'block';
          rawDisplay.textContent = data.raw_transcription;
        } else {
          rawBox.style.display = 'none';
        }
      }

      // Compute WPM
      const wordCount = (data.transcription || '').trim().split(/\\s+/).filter(Boolean).length;
      const minutes = (data.duration_sec || 1) / 60;
      const wpm = Math.round(wordCount / (minutes || 1));
      document.getElementById('valWpm').textContent = wpm || '0';

      // Word Timestamps
      const cloud = document.getElementById('wordCloud');
      cloud.innerHTML = '';

      if (data.words && data.words.length > 0) {
        data.words.forEach(w => {
          const pill = document.createElement('div');
          pill.className = 'word-pill';
          pill.innerHTML = `<strong>${w.word}</strong> <span class="time">${w.start_time}s</span>`;
          pill.onclick = () => seekAudio(w.start_time);
          cloud.appendChild(pill);
        });
      } else {
        cloud.innerHTML = '<span style="color: var(--text-dim);">No word boundary breakdown available for this utterance.</span>';
      }
    }

    function seekAudio(seconds) {
      const player = document.getElementById('audioPlayer');
      player.currentTime = Math.max(0, seconds - 0.1);
      player.play();
    }

    function copyTranscription() {
      const text = document.getElementById('transcriptionDisplay').textContent;
      navigator.clipboard.writeText(text);
      alert('Transcription copied to clipboard!');
    }
  </script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Request Handler
# ─────────────────────────────────────────────────────────────────────────────

class DysarthriaASRServer(BaseHTTPRequestHandler):

    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=200)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # 1. Serve Web Application UI
        if path in ["/", "/index.html"]:
            self._set_headers(content_type="text/html", status=200)
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            return

        # 2. API: List Adapters
        if path == "/api/adapters":
            transcriber = get_transcriber()
            adapters_info = transcriber.list_available_adapters()
            self._set_headers()
            self.wfile.write(json.dumps(adapters_info).encode("utf-8"))
            return

        # 3. API: List Dataset Samples
        if path == "/api/sample_audios":
            samples = []
            if os.path.exists(PROCESSED_AUDIO_DIR):
                for spk_dir in sorted(os.listdir(PROCESSED_AUDIO_DIR)):
                    full_spk_dir = os.path.join(PROCESSED_AUDIO_DIR, spk_dir)
                    if os.path.isdir(full_spk_dir):
                        wavs = sorted(glob.glob(os.path.join(full_spk_dir, "*.wav")))
                        for wav in wavs[:3]:  # Top 3 per speaker
                            samples.append({
                                "speaker": spk_dir,
                                "filename": os.path.basename(wav),
                                "path": wav,
                                "severity": SPEAKER_SEVERITY.get(spk_dir, "Unknown"),
                            })

            self._set_headers()
            self.wfile.write(json.dumps(samples).encode("utf-8"))
            return

        # 4. API: Stream Audio File
        if path == "/api/audio":
            file_path = params.get("path", [None])[0]
            if file_path and os.path.exists(file_path):
                self._set_headers(content_type="audio/wav")
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers(status=404)
                self.wfile.write(json.dumps({"error": "Audio file not found"}).encode("utf-8"))
            return

        # Unknown route
        self._set_headers(status=404)
        self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""

            # 1. Transcribe Preprocessed Dataset Sample
            if path == "/api/transcribe_sample":
                data = json.loads(body.decode("utf-8"))
                audio_path = data.get("audio_path")
                speaker_id = data.get("speaker_id") or None

                if not audio_path or not os.path.exists(audio_path):
                    self._set_headers(status=400)
                    self.wfile.write(json.dumps({"error": f"Invalid audio path: {audio_path}"}).encode("utf-8"))
                    return

                transcriber = get_transcriber()
                transcriber.switch_adapter(speaker_id=speaker_id)
                result = transcriber.transcribe_with_timestamps(audio_path)
                self._set_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
                return

            # 2. Transcribe Uploaded File / Microphone Blob
            if path == "/api/transcribe_upload":
                content_type = self.headers.get("Content-Type", "")
                speaker_id = None
                audio_bytes = None

                if "application/json" in content_type:
                    data = json.loads(body.decode("utf-8"))
                    speaker_id = data.get("speaker_id") or None
                    b64_str = data.get("audio_base64", "")
                    if not b64_str:
                        self._set_headers(status=400)
                        self.wfile.write(json.dumps({"error": "No audio payload provided"}).encode("utf-8"))
                        return
                    audio_bytes = base64.b64decode(b64_str)

                elif "multipart/form-data" in content_type:
                    form = cgi.FieldStorage(
                        fp=io.BytesIO(body),
                        headers=self.headers,
                        environ={
                            "REQUEST_METHOD": "POST",
                            "CONTENT_TYPE": content_type,
                            "CONTENT_LENGTH": str(len(body)),
                        }
                    )
                    if "audio" in form:
                        audio_item = form["audio"]
                        speaker_id = form.getvalue("speaker_id", None) or None
                        audio_bytes = audio_item.file.read()

                else:
                    audio_bytes = body

                if not audio_bytes:
                    self._set_headers(status=400)
                    self.wfile.write(json.dumps({"error": "Empty audio data received"}).encode("utf-8"))
                    return

                os.makedirs(LOGS_DIR, exist_ok=True)
                temp_path = os.path.join(LOGS_DIR, "_temp_upload.wav")
                with open(temp_path, "wb") as f:
                    f.write(audio_bytes)

                transcriber = get_transcriber()
                transcriber.switch_adapter(speaker_id=speaker_id)

                result = transcriber.transcribe_with_timestamps(temp_path)
                self._set_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
                return

            self._set_headers(status=404)
            self.wfile.write(json.dumps({"error": "Route not found"}).encode("utf-8"))

        except Exception as e:
            logger.error(f"Error handling POST {self.path}: {e}", exc_info=True)
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"error": f"Transcription error: {str(e)}"}).encode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Server Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_server(port: int = 8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DysarthriaASRServer)
    logger.info("=" * 70)
    logger.info(f"VocalAid AI Platform running at: http://localhost:{port}")
    logger.info("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down VocalAid server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VocalAid AI Web Platform")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    args = parser.parse_args()
    run_server(port=args.port)
