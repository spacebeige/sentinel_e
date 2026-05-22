import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../services/api'; // Assuming api service is available

/**
 * Hook to consume the Sentinel-E Runtime Event Bus SSE stream.
 * Includes throttling to prevent excessive re-renders during high-volume debate phases.
 */
export function useRuntimeStream(runId) {
  const [events, setEvents] = useState([]);
  const [currentPhase, setCurrentPhase] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, connecting, active, complete, error
  const [error, setError] = useState(null);

  const eventBufferRef = useRef([]);
  const updateTimeoutRef = useRef(null);

  const flushBuffer = useCallback(() => {
    if (eventBufferRef.current.length > 0) {
      const newEvents = [...eventBufferRef.current];
      
      setEvents((prev) => [...prev, ...newEvents]);
      
      // Update current phase if a transition occurred
      const transitions = newEvents.filter(e => e.event_type === 'phase_transition');
      if (transitions.length > 0) {
        setCurrentPhase(transitions[transitions.length - 1].phase);
      }
      
      // Check for completion
      if (newEvents.some(e => e.event_type === 'orchestration_completed' || e.event_type === 'orchestration_failed')) {
        setStatus('complete');
      }

      eventBufferRef.current = [];
    }
    updateTimeoutRef.current = null;
  }, []);

  useEffect(() => {
    if (!runId) {
      setStatus('idle');
      setEvents([]);
      return;
    }

    setStatus('connecting');
    setError(null);
    setEvents([]);
    eventBufferRef.current = [];

    // Assuming api.client gives access to the base URL
    const url = `${api.client.defaults.baseURL || '/api'}/orchestration/${runId}/events`;
    
    // We append authorization if needed, but EventSource doesn't support headers well.
    // Assuming cookie auth or token in URL if needed. Here we just use basic EventSource.
    const token = localStorage.getItem('sentinel_token');
    const esUrl = token ? `${url}?token=${token}` : url;

    const eventSource = new EventSource(esUrl);

    eventSource.onopen = () => {
      setStatus('active');
    };

    eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.event_type === 'heartbeat') return;
        if (data.event_type === 'stream_end') {
          eventSource.close();
          setStatus('complete');
          return;
        }

        eventBufferRef.current.push(data);

        // Throttle updates to ~60fps (16ms) or slightly slower (50ms)
        if (!updateTimeoutRef.current) {
          updateTimeoutRef.current = setTimeout(flushBuffer, 50);
        }
      } catch (err) {
        console.error('Failed to parse SSE message', err);
      }
    };

    eventSource.onerror = (err) => {
      console.error('SSE Error', err);
      eventSource.close();
      setStatus('error');
      setError('Connection lost or stream failed');
    };

    return () => {
      eventSource.close();
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
      flushBuffer();
    };
  }, [runId, flushBuffer]);

  return { events, currentPhase, status, error };
}
