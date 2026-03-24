/**
 * ============================================================
 * AWAAZ Service — Voice Integration Layer
 * ============================================================
 * 
 * Handles:
 *   - Audio recording from microphone
 *   - Speech-to-text (STT) transcription
 *   - NLP routing & language detection
 *   - AI processing with model selection
 *   - Text-to-speech (TTS) synthesis
 */

import axios from 'axios';

// Use backend AWAAZ API endpoint
const AWAAZ_API_BASE = process.env.REACT_APP_AWAAZ_API || 'http://localhost:8000/api/v1';

export class AwaazService {
  constructor() {
    this.mediaRecorder = null;
    this.audioContext = null;
    this.stream = null;
    this.audioChunks = [];
    this.isRecording = false;
  }

  /**
   * Initialize audio context and request microphone access
   */
  async initializeAudio() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      this.audioContext = audioContext;
      
      return true;
    } catch (error) {
      console.error('Failed to access microphone:', error);
      throw new Error('Microphone access denied. Please check your permissions.');
    }
  }

  /**
   * Start recording audio
   */
  startRecording() {
    if (!this.stream) {
      throw new Error('Audio not initialized. Call initializeAudio() first.');
    }

    this.audioChunks = [];
    this.isRecording = true;

    this.mediaRecorder = new MediaRecorder(this.stream, {
      mimeType: 'audio/webm;codecs=opus',
    });

    this.mediaRecorder.ondataavailable = (e) => {
      this.audioChunks.push(e.data);
    };

    this.mediaRecorder.start();
  }

  /**
   * Stop recording and return audio blob
   */
  stopRecording() {
    return new Promise((resolve) => {
      if (!this.mediaRecorder) {
        resolve(null);
        return;
      }

      this.mediaRecorder.onstop = () => {
        this.isRecording = false;
        const audioBlob = new Blob(this.audioChunks, {
          type: 'audio/webm;codecs=opus',
        });
        this.audioChunks = [];
        resolve(audioBlob);
      };

      this.mediaRecorder.stop();
    });
  }

  /**
   * Upload audio file for transcription
   * @param {Blob} audioBlob - Audio data as blob
   * @returns {Promise} - Job details including job_id
   */
  async uploadAudio(audioBlob) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');

    try {
      const response = await axios.post(
        `${AWAAZ_API_BASE}/transcription/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      return response.data;
    } catch (error) {
      console.error('Audio upload failed:', error);
      throw new Error(`Upload failed: ${error.message}`);
    }
  }

  /**
   * Poll transcription job status
   * @param {string} jobId - Job ID from upload
   * @returns {Promise} - Transcription job details
   */
  async getTranscriptionStatus(jobId) {
    try {
      const response = await axios.get(
        `${AWAAZ_API_BASE}/transcription/status/${jobId}`
      );
      return response.data;
    } catch (error) {
      console.error('Failed to get transcription status:', error);
      throw error;
    }
  }

  /**
   * Get full pipeline result (transcription + AI processing)
   * @param {string} jobId - Job ID
   * @returns {Promise} - Complete pipeline result with AI response
   */
  async getPipelineResult(jobId) {
    try {
      const response = await axios.get(
        `${AWAAZ_API_BASE}/pipeline/result/${jobId}`
      );
      return response.data;
    } catch (error) {
      console.error('Failed to get pipeline result:', error);
      throw error;
    }
  }

  /**
   * Process NLP for text (language detection + grievance classification)
   * @param {string} text - Transcribed text
   * @param {string} language - Detected language code (optional)
   * @returns {Promise} - NLP analysis including grievance category and intent
   */
  async processNLP(text, language = null) {
    try {
      const response = await axios.post(
        `${AWAAZ_API_BASE}/nlp/analyze`,
        {
          text,
          language,
        }
      );
      return response.data;
    } catch (error) {
      console.error('NLP processing failed:', error);
      throw error;
    }
  }

  /**
   * Send transcribed text to AI model with NLP context
   * @param {string} text - Transcribed text
   * @param {string} language - Detected language
   * @param {string} mode - Model mode (standard/experimental)
   * @param {object} nlpContext - NLP analysis results
   * @returns {Promise} - AI processing job details
   */
  async processWithAI(text, language, mode = 'standard', nlpContext = null) {
    try {
      const response = await axios.post(
        `${AWAAZ_API_BASE}/ai/process`,
        {
          text,
          language,
          mode,
          nlp_context: nlpContext,
        }
      );
      return response.data;
    } catch (error) {
      console.error('AI processing failed:', error);
      throw error;
    }
  }

  /**
   * Generate speech from text (TTS)
   * @param {string} text - Text to synthesize
   * @param {string} language - Target language
   * @returns {Promise} - Audio file URL
   */
  async synthesizeText(text, language) {
    try {
      const response = await axios.post(
        `${AWAAZ_API_BASE}/tts/synthesize`,
        {
          text,
          language,
        }
      );
      return response.data;
    } catch (error) {
      console.error('TTS synthesis failed:', error);
      throw error;
    }
  }

  /**
   * Detect language from text
   * @param {string} text - Text to analyze
   * @returns {Promise} - Language detection result
   */
  async detectLanguage(text) {
    try {
      const response = await axios.post(
        `${AWAAZ_API_BASE}/language/detect`,
        {
          text,
        }
      );
      return response.data;
    } catch (error) {
      console.error('Language detection failed:', error);
      // Return default (English) on failure
      return { language: 'en', confidence: 0 };
    }
  }

  /**
   * Poll job with exponential backoff
   * @param {string} jobId - Job ID to poll
   * @param {string} jobType - Type of job (transcription/ai_processing/tts)
   * @param {function} getStatusFn - Function to call for status
   * @param {number} maxWaitMs - Maximum wait time
   * @returns {Promise} - Completed job result
   */
  async pollJobWithBackoff(jobId, jobType, getStatusFn, maxWaitMs = 60000) {
    const initialWaitTime = 500;
    const maxWaitTime = 3000;
    let waitTime = initialWaitTime;
    let totalWaited = 0;

    // eslint-disable-next-line no-constant-condition
    while (true) {
      if (totalWaited >= maxWaitMs) {
        throw new Error(`${jobType} job timed out after ${maxWaitMs}ms`);
      }

      try {
        const result = await getStatusFn(jobId);

        if (result.status === 'completed') {
          return result;
        }

        if (result.status === 'failed') {
          throw new Error(result.error_message || `${jobType} job failed`);
        }

        // Wait before polling again with constant waitTime value
        const delay = waitTime;
        // eslint-disable-next-line no-await-in-loop
        await new Promise((resolve) => {
          setTimeout(resolve, delay);
        });
        totalWaited += delay;
        waitTime = Math.min(waitTime * 1.5, maxWaitTime); // Cap at 3 seconds
      } catch (error) {
        if (totalWaited >= maxWaitMs) {
          throw new Error(`${jobType} job timed out after ${maxWaitMs}ms`);
        }
        throw error;
      }
    }
  }

  /**
   * End audio stream
   */
  cleanup() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }
    if (this.audioContext) {
      this.audioContext.close();
    }
    this.mediaRecorder = null;
    this.stream = null;
    this.audioContext = null;
  }
}

const awaazServiceInstance = new AwaazService();
export default awaazServiceInstance;
