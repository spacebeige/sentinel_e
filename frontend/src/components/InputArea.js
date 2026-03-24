import React, { useState, useRef } from 'react';
import { Paperclip, X, CornerDownLeft, Mic, MicOff, Loader, AlertCircle } from 'lucide-react';
import awaazService from '../services/awaazService';
import { analyzeAndRoute } from '../engines/nlpRouter';
import { analyzeLanguageAndSelectModel } from '../engines/languageDetector';

const InputArea = ({ onSend, loading, mode, subMode, onModeChange, onSubModeChange }) => {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [isFocused, setIsFocused] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [nlpAnalysis, setNlpAnalysis] = useState(null);
  const [showNLPHint, setShowNLPHint] = useState(false);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const recordingIntervalRef = useRef(null);

  const handleSend = () => {
    if (!text.trim() && !file) return;
    onSend({ text: text.trim(), file });
    setText('');
    setFile(null);
    if (textareaRef.current) textareaRef.current.style.height = '52px';
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const autoResize = (e) => {
    setText(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  };

  const startRecording = async () => {
    try {
      setIsRecording(true);
      setRecordingTime(0);
      await awaazService.initializeAudio();
      awaazService.startRecording();

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      console.error('Failed to start recording:', error);
      setIsRecording(false);
      alert(`Recording failed: ${error.message}`);
    }
  };

  const stopRecording = async () => {
    try {
      setIsRecording(false);
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }

      setIsTranscribing(true);
      const audioBlob = await awaazService.stopRecording();

      if (!audioBlob) {
        setIsTranscribing(false);
        return;
      }

      // Upload audio for transcription
      const uploadResult = await awaazService.uploadAudio(audioBlob);
      const jobId = uploadResult.job_id;

      // Poll for transcription result
      const transcriptionResult = await awaazService.pollJobWithBackoff(
        jobId,
        'transcription',
        (id) => awaazService.getTranscriptionStatus(id),
        30000
      );

      const transcribedText = transcriptionResult.transcribed_text || '';
      const detectedLanguage = transcriptionResult.detected_language || 'en';

      // ─────────────────────────────────────────────────────────
      // NLP ANALYSIS & ROUTING
      // ─────────────────────────────────────────────────────────
      const routingDecision = analyzeAndRoute(transcribedText, detectedLanguage);
      const languageSelection = analyzeLanguageAndSelectModel(
        transcribedText,
        detectedLanguage,
        'standard'
      );

      // Update mode/subMode based on NLP analysis
      if (onModeChange && routingDecision.recommended_mode) {
        onModeChange(routingDecision.recommended_mode);
      }

      if (onSubModeChange && routingDecision.recommended_subMode) {
        onSubModeChange(routingDecision.recommended_subMode);
      }

      // Store NLP analysis for display
      setNlpAnalysis({
        routing: routingDecision,
        language: languageSelection,
        transcriptionConfidence: transcriptionResult.confidence || 0,
      });

      // Show NLP hint briefly
      setShowNLPHint(true);
      setTimeout(() => setShowNLPHint(false), 5000);

      // Auto-populate textarea with transcribed text
      setText(transcribedText);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.style.height = Math.min(
          textareaRef.current.scrollHeight,
          200
        ) + 'px';
      }

      setIsTranscribing(false);
    } catch (error) {
      console.error('Transcription failed:', error);
      setIsTranscribing(false);
      alert(`Transcription error: ${error.message}`);
    } finally {
      awaazService.cleanup();
    }
  };

  const handleMicClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const modeLabel =
    mode === 'experimental' ? `Omega · ${subMode || 'debate'}` : 'Sentinel';

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full">
      <div
        className="rounded-2xl transition-all"
        style={{
          backgroundColor: 'var(--bg-input)',
          border: isFocused
            ? '1px solid var(--border-focus)'
            : '1px solid var(--border-primary)',
          boxShadow: isFocused
            ? 'var(--shadow-md)'
            : 'var(--shadow-sm)',
        }}
      >
        {/* File preview */}
        {file && (
          <div className="px-4 pt-3 flex items-center">
            <div
              className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full"
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                border: '1px solid var(--border-secondary)',
                color: 'var(--text-primary)',
              }}
            >
              <Paperclip
                className="w-3 h-3"
                style={{ color: 'var(--accent-blue)' }}
              />
              <span className="truncate max-w-[200px]">{file.name}</span>
              <button
                onClick={() => setFile(null)}
                className="ml-1 transition-colors"
                style={{ color: 'var(--accent-red)' }}
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}

        {/* Recording indicator */}
        {isRecording && (
          <div className="px-4 pt-2 flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full animate-pulse"
              style={{ backgroundColor: 'var(--accent-red)' }}
            />
            <span
              className="text-xs font-semibold"
              style={{ color: 'var(--accent-red)' }}
            >
              Recording: {formatTime(recordingTime)}
            </span>
          </div>
        )}

        {/* NLP Analysis Hint */}
        {showNLPHint && nlpAnalysis && (
          <div className="px-4 py-2 flex items-start gap-2 rounded-lg mb-2"
            style={{
              backgroundColor: 'var(--bg-tertiary)',
              border: '1px solid var(--border-secondary)',
            }}>
            <AlertCircle className="w-4 h-4 mt-0.5" style={{ color: 'var(--accent-blue)' }} />
            <div className="text-xs flex-1">
              <p style={{ color: 'var(--text-primary)' }}>
                <strong>NLP Analysis:</strong> {nlpAnalysis.routing.detected_grievance.category_name}
              </p>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                Intent: <strong>{nlpAnalysis.routing.detected_intent}</strong> • 
                Language: <strong>{nlpAnalysis.language.languageInfo.name}</strong> • 
                Mode: <strong>{nlpAnalysis.routing.recommended_mode}{nlpAnalysis.routing.recommended_subMode ? ` (${nlpAnalysis.routing.recommended_subMode})` : ''}</strong>
              </p>
            </div>
          </div>
        )}

        {/* Transcribing indicator */}
        {isTranscribing && (
          <div className="px-4 pt-2 flex items-center gap-2">
            <Loader
              className="w-4 h-4 animate-spin"
              style={{ color: 'var(--accent-blue)' }}
            />
            <span
              className="text-xs font-semibold"
              style={{ color: 'var(--accent-blue)' }}
            >
              Transcribing and analyzing audio...
            </span>
          </div>
        )}

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={autoResize}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onKeyDown={handleKeyPress}
          placeholder={`Message ${modeLabel}...`}
          className="w-full bg-transparent focus:outline-none resize-none px-4 py-3.5 text-sm leading-relaxed scrollbar-thin"
          style={{
            color: 'var(--text-primary)',
            minHeight: '52px',
            maxHeight: '200px',
          }}
          rows={1}
          disabled={isRecording}
        />

        {/* Toolbar */}
        <div className="flex items-center justify-between px-2 pb-2">
          <div className="flex items-center gap-1">
            {/* File attachment button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isRecording || isTranscribing}
              className="p-2 rounded-lg transition-colors"
              style={{
                color: 'var(--text-tertiary)',
                cursor:
                  isRecording || isTranscribing
                    ? 'not-allowed'
                    : 'pointer',
                opacity: isRecording || isTranscribing ? 0.5 : 1,
              }}
              onMouseEnter={(e) => {
                if (!isRecording && !isTranscribing) {
                  e.target.style.color = 'var(--text-primary)';
                }
              }}
              onMouseLeave={(e) => {
                if (!isRecording && !isTranscribing) {
                  e.target.style.color = 'var(--text-tertiary)';
                }
              }}
            >
              <Paperclip className="w-5 h-5" />
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) =>
                e.target.files[0] && setFile(e.target.files[0])
              }
              className="hidden"
            />

            {/* Microphone button */}
            <button
              onClick={handleMicClick}
              disabled={isTranscribing || loading}
              className="p-2 rounded-lg transition-colors"
              style={{
                color: isRecording
                  ? 'var(--accent-red)'
                  : 'var(--text-tertiary)',
                cursor:
                  isTranscribing || loading
                    ? 'not-allowed'
                    : 'pointer',
                opacity: isTranscribing || loading ? 0.5 : 1,
              }}
              onMouseEnter={(e) => {
                if (!isTranscribing && !loading && !isRecording) {
                  e.target.style.color = 'var(--text-primary)';
                }
              }}
              onMouseLeave={(e) => {
                if (!isRecording && !isTranscribing && !loading) {
                  e.target.style.color = 'var(--text-tertiary)';
                }
              }}
              title={
                isRecording
                  ? 'Stop recording'
                  : 'Start voice transcription'
              }
            >
              {isRecording ? (
                <MicOff className="w-5 h-5" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </button>

            <span
              className="text-[9px] font-mono uppercase tracking-wider px-2 py-0.5 rounded"
              style={{
                color: 'var(--text-tertiary)',
                backgroundColor: 'var(--bg-tertiary)',
              }}
            >
              {modeLabel}
            </span>
          </div>

          <button
            onClick={handleSend}
            disabled={loading || (!text.trim() && !file) || isRecording}
            className="p-2 rounded-lg transition-all flex items-center justify-center"
            style={{
              backgroundColor:
                text.trim() || file ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
              color:
                text.trim() || file ? '#fff' : 'var(--text-tertiary)',
              cursor:
                loading || (!text.trim() && !file) || isRecording
                  ? 'not-allowed'
                  : 'pointer',
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <CornerDownLeft className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      <div className="text-center mt-2">
        <p
          className="text-[10px]"
          style={{ color: 'var(--text-tertiary)' }}
        >
          Sentinel-E may produce inaccurate results. Verify critical
          information independently.
        </p>
      </div>
    </div>
  );
};

export default InputArea;
