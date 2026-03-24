/**
 * ============================================================
 * useVoiceTranscription Hook — Voice Input Integration
 * ============================================================
 *
 * Encapsulates:
 *   - Audio recording management
 *   - Speech-to-text transcription
 *   - NLP analysis & routing
 *   - Mode/SubMode selection
 *   - Language detection
 */

import { useState, useRef, useCallback } from 'react';
import awaazService from '../services/awaazService';
import { analyzeAndRoute } from '../engines/nlpRouter';
import { analyzeLanguageAndSelectModel } from '../engines/languageDetector';

export function useVoiceTranscription() {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [nlpAnalysis, setNlpAnalysis] = useState(null);
  const [transcribedText, setTranscribedText] = useState('');
  const [error, setError] = useState(null);
  const recordingIntervalRef = useRef(null);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setIsRecording(true);
      setRecordingTime(0);
      await awaazService.initializeAudio();
      awaazService.startRecording();

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setError(err.message);
      setIsRecording(false);
    }
  }, []);

  const stopRecording = useCallback(async () => {
    return new Promise(async (resolve, reject) => {
      try {
        setIsRecording(false);
        if (recordingIntervalRef.current) {
          clearInterval(recordingIntervalRef.current);
        }

        setIsTranscribing(true);
        const audioBlob = await awaazService.stopRecording();

        if (!audioBlob) {
          setIsTranscribing(false);
          resolve(null);
          return;
        }

        // Upload audio for transcription
        const uploadResult = await awaazService.uploadAudio(audioBlob);
        const jobId = uploadResult.job_id;

        // Poll for transcription result
        const transcriptionResult =
          await awaazService.pollJobWithBackoff(
            jobId,
            'transcription',
            (id) => awaazService.getTranscriptionStatus(id),
            30000
          );

        const text = transcriptionResult.transcribed_text || '';
        const language = transcriptionResult.detected_language || 'en';

        // ─────────────────────────────────────────────────────────
        // NLP ANALYSIS & ROUTING
        // ─────────────────────────────────────────────────────────
        const routingDecision = analyzeAndRoute(text, language);
        const languageSelection = analyzeLanguageAndSelectModel(
          text,
          language,
          'standard'
        );

        // Store analysis results
        setTranscribedText(text);
        setNlpAnalysis({
          routing: routingDecision,
          language: languageSelection,
          transcriptionConfidence: transcriptionResult.confidence || 0,
        });

        setIsTranscribing(false);
        setError(null);

        // Resolve with complete analysis
        resolve({
          text,
          language,
          routing: routingDecision,
          languageSelection,
          confidence: transcriptionResult.confidence || 0,
        });
      } catch (err) {
        setIsTranscribing(false);
        setError(err.message);
        reject(err);
      } finally {
        awaazService.cleanup();
      }
    });
  }, []);

  const reset = useCallback(() => {
    setIsRecording(false);
    setIsTranscribing(false);
    setRecordingTime(0);
    setNlpAnalysis(null);
    setTranscribedText('');
    setError(null);
  }, []);

  return {
    // State
    isRecording,
    isTranscribing,
    recordingTime,
    nlpAnalysis,
    transcribedText,
    error,

    // Methods
    startRecording,
    stopRecording,
    reset,
  };
}

export default useVoiceTranscription;
