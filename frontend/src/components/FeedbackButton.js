import React, { useState } from "react";
import axios from "axios";
import { ThumbsUp, ThumbsDown, Loader2, Star } from "lucide-react";

/**
 * FeedbackButton — v4.5
 * Thumbs up/down, 1-5 star, optional reason. Uses CSS variables.
 */
const FeedbackButton = ({ runId, mode, subMode, onFeedbackSent }) => {
  const [feedback, setFeedback] = useState(null);
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [reason, setReason] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState(null);
  const [showRating, setShowRating] = useState(false);

  const submitFeedback = async (type, starRating) => {
    setIsSubmitting(true);
    setError(null);
    try {
      const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
      const formData = new FormData();
      formData.append("run_id", runId);
      formData.append("feedback", type);
      if (starRating) formData.append("rating", starRating);
      if (mode) formData.append("mode", mode);
      if (subMode) formData.append("sub_mode", subMode);
      if (type === "down" && reason) formData.append("reason", reason.substring(0, 500));

      const response = await axios.post(`${BASE_URL}/feedback`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.status === 200) {
        setSubmitted(true);
        setFeedback(null);
        setReason("");
        setShowRating(false);
        if (onFeedbackSent && response.data.feedback_id) onFeedbackSent(response.data.feedback_id);
        setTimeout(() => setSubmitted(false), 2500);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleVote = async (type) => {
    if (type === "up") {
      await submitFeedback("up", rating || null);
    } else {
      if (!feedback) { setFeedback("down"); setShowRating(true); return; }
      await submitFeedback("down", rating || null);
    }
  };

  const handleStarClick = async (star) => {
    setRating(star);
    if (!feedback || feedback === "up") await submitFeedback("up", star);
  };

  if (submitted) {
    return <div className="text-xs flex items-center gap-1" style={{ color: 'var(--accent-green)' }}>
      ✓ Thanks{rating > 0 && ` (${rating}★)`}
    </div>;
  }

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <div className="flex gap-1.5">
        <button onClick={() => handleVote("up")} disabled={isSubmitting}
          className="p-1.5 rounded-md transition-all"
          style={{
            border: feedback === "up" ? '1px solid var(--accent-green)' : '1px solid var(--border-secondary)',
            backgroundColor: feedback === "up" ? 'rgba(34,197,94,0.1)' : 'transparent',
            color: feedback === "up" ? 'var(--accent-green)' : 'var(--text-tertiary)',
          }} title="Helpful">
          <ThumbsUp size={14} strokeWidth={2} />
        </button>
        <button onClick={() => handleVote("down")} disabled={isSubmitting}
          className="p-1.5 rounded-md transition-all"
          style={{
            border: feedback === "down" ? '1px solid var(--accent-red)' : '1px solid var(--border-secondary)',
            backgroundColor: feedback === "down" ? 'rgba(239,68,68,0.1)' : 'transparent',
            color: feedback === "down" ? 'var(--accent-red)' : 'var(--text-tertiary)',
          }} title="Not Helpful">
          <ThumbsDown size={14} strokeWidth={2} />
        </button>
      </div>

      <div className="flex items-center gap-0.5">
        {[1, 2, 3, 4, 5].map(star => (
          <button key={star} onClick={() => handleStarClick(star)}
            onMouseEnter={() => setHoverRating(star)} onMouseLeave={() => setHoverRating(0)}
            disabled={isSubmitting} className="p-0.5 transition-all">
            <Star size={14} strokeWidth={2}
              className="transition-colors"
              style={{
                color: star <= (hoverRating || rating) ? 'var(--accent-yellow)' : 'var(--text-tertiary)',
                fill: star <= (hoverRating || rating) ? 'var(--accent-yellow)' : 'transparent',
              }} />
          </button>
        ))}
      </div>

      {feedback === "down" && showRating && (
        <div className="flex items-center gap-2">
          <input type="text" value={reason} onChange={(e) => setReason(e.target.value)}
            placeholder="What could improve?"
            className="rounded-md px-2.5 py-1 text-xs outline-none focus:ring-1 w-44"
            style={{
              backgroundColor: 'var(--bg-input)',
              border: '1px solid var(--border-primary)',
              color: 'var(--text-primary)',
            }}
            onKeyDown={(e) => { if (e.key === 'Enter') handleVote("down"); }} />
          <button onClick={() => handleVote("down")} disabled={isSubmitting}
            className="text-[10px] px-2.5 py-1 rounded font-bold uppercase transition-colors"
            style={{ backgroundColor: 'var(--accent-red)', color: '#fff' }}>
            {isSubmitting ? "..." : "Send"}
          </button>
        </div>
      )}

      {isSubmitting && <Loader2 size={14} className="animate-spin" style={{ color: 'var(--text-tertiary)' }} />}
      {error && <span className="text-[10px]" style={{ color: 'var(--accent-red)' }}>{error}</span>}
    </div>
  );
};

export { FeedbackButton };
export default FeedbackButton;
