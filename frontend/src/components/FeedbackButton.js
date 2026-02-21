import React, { useState } from "react";
import axios from "axios";
import { ThumbsUp, ThumbsDown, Loader2 } from "lucide-react";

/**
 * FeedbackButton Component
 *
 * Minimal, low-friction feedback UI inspired by ChatGPT.
 *
 * Features:
 * - Two buttons: Thumbs-up (👍) and Thumbs-down (👎)
 * - Optional text field on negative feedback
 * - Non-blocking, appears after output
 * - No re-execution or output modification
 * - Sends feedback to /feedback endpoint
 */

export const FeedbackButton = ({ runId, onFeedbackSent }) => {
  const [feedback, setFeedback] = useState(null);
  const [reason, setReason] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState(null);

  const handleFeedback = async (type) => {
    if (feedback === type && type === "down" && reason === "") {
      // Show reason field for down vote
      setFeedback(type);
      return;
    }

    if (type === "down" && reason === "") {
      // Require reason for down vote
      setFeedback(type);
      return;
    }

    // Submit feedback
    setIsSubmitting(true);
    setError(null);

    try {
      const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
      const formData = new FormData();
      formData.append("run_id", runId);
      formData.append("feedback", type);
      if (type === "down" && reason) {
        formData.append("reason", reason.substring(0, 500));
      }

      // Use dynamic base URL
      const response = await axios.post(
        `${BASE_URL}/feedback`,
        formData,
        {
          headers: {
            // FormData usually sets Content-Type boundary automatically.
            'Content-Type': 'multipart/form-data' 
          },
        }
      );

      if (response.status === 200) {
        setSubmitted(true);
        setFeedback(null);
        setReason("");

        if (onFeedbackSent && response.data.feedback_id) {
          onFeedbackSent(response.data.feedback_id);
        }

        // Auto-hide after 2 seconds
        setTimeout(() => {
          setSubmitted(false);
        }, 2000);
      }
    } catch (err) {
      setError(
        err.response?.data?.detail || err.message || "Failed to submit feedback"
      );
      console.error("Feedback submission error:", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const styles = {
    container: {
      display: "flex",
      alignItems: "center",
      gap: "12px",
      marginTop: "12px",
      fontSize: "0.875rem",
      color: "#94a3b8", // slate-400
    },
    buttonGroup: {
      display: "flex",
      gap: "8px",
    },
    button: (active, type) => ({
      padding: "6px",
      borderRadius: "6px",
      border: "1px solid",
      borderColor: active ? (type === "up" ? "#10b981" : "#ef4444") : "rgba(255, 255, 255, 0.1)", // Subtle border
      backgroundColor: active ? (type === "up" ? "rgba(16, 185, 129, 0.1)" : "rgba(239, 68, 68, 0.1)") : "rgba(255, 255, 255, 0.05)", // Subtle bg
      color: active ? (type === "up" ? "#10b981" : "#ef4444") : "#94a3b8",
      cursor: "pointer",
      transition: "all 0.2s",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    }),
    input: {
      backgroundColor: "rgba(255, 255, 255, 0.05)",
      border: "1px solid rgba(255, 255, 255, 0.1)",
      borderRadius: "6px",
      padding: "6px 10px",
      color: "#e2e8f0",
      fontSize: "0.875rem",
      outline: "none",
      width: "200px",
    },
    error: {
      color: "#ef4444",
      fontSize: "0.75rem",
      marginLeft: "8px",
    },
    success: {
      color: "#10b981",
      fontSize: "0.875rem",
      display: "flex",
      alignItems: "center",
      gap: "6px",
    }
  };

  if (submitted) {
     return <div className="text-xs text-emerald-400 flex items-center gap-1">✓ Thanks for your feedback</div>;
  }

  return (
    <div className="flex items-center gap-2">
      <div className="flex gap-2">
        <button
          onClick={() => handleFeedback("up")}
          disabled={isSubmitting}
          style={styles.button(feedback === "up", "up")}
          title="Helpful"
          className="text-gray-400 hover:text-white transition-all p-1"
        >
          <ThumbsUp size={18} strokeWidth={2} />
        </button>
        <button
          onClick={() => handleFeedback("down")}
          disabled={isSubmitting}
          style={styles.button(feedback === "down", "down")}
          title="Not Helpful"
          className="text-gray-400 hover:text-white transition-all p-1"
        >
          <ThumbsDown size={18} strokeWidth={2} />
        </button>
      </div>

      {feedback === "down" && (
         <div className="flex items-center gap-2 animate-in fade-in slide-in-from-left-2 duration-200 ml-2">
            <input
                type="text"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="What could improve?"
                style={styles.input}
                onKeyDown={(e) => {
                    if (e.key === 'Enter') handleFeedback("down");
                }}
                className="focus:ring-1 focus:ring-red-500/50 text-xs"
            />
            <button 
                onClick={() => handleFeedback("down")}
                className="text-xs px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded transition-colors disabled:opacity-50"
                disabled={isSubmitting}
            >
                {isSubmitting ? "..." : "Send"}
            </button>
         </div>
      )}

      {isSubmitting && <Loader2 size={16} className="animate-spin text-slate-500" />}
      {error && <span className="text-xs text-red-400">{error}</span>}
    </div>
  );
};

export default FeedbackButton;
