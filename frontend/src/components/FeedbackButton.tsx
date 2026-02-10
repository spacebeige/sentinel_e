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

import React, { useState } from "react";
import axios from "axios";

interface FeedbackButtonProps {
  runId: string;
  onFeedbackSent?: (feedbackId: string) => void;
}

export const FeedbackButton: React.FC<FeedbackButtonProps> = ({
  runId,
  onFeedbackSent,
}) => {
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);
  const [reason, setReason] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFeedback = async (type: "up" | "down") => {
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
      const formData = new FormData();
      formData.append("run_id", runId);
      formData.append("feedback", type);
      if (type === "down" && reason) {
        formData.append("reason", reason.substring(0, 500));
      }

      const response = await axios.post(
        `${process.env.REACT_APP_API_URL || "http://localhost:8000"}/feedback`,
        formData,
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
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
        err instanceof Error ? err.message : "Failed to submit feedback"
      );
      console.error("Feedback submission error:", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="sentiment-feedback-container" style={styles.container}>
      {submitted ? (
        <div style={styles.successMessage}>✓ Thank you for your feedback</div>
      ) : (
        <>
          <div style={styles.label}>Was this response helpful?</div>

          <div style={styles.buttonGroup}>
            {/* Thumbs Up Button */}
            <button
              onClick={() => handleFeedback("up")}
              disabled={isSubmitting}
              style={{
                ...styles.button,
                ...(feedback === "up" ? styles.buttonActive : {}),
              }}
              title="This response was helpful"
              aria-label="Thumbs up - helpful"
            >
              👍
            </button>

            {/* Thumbs Down Button */}
            <button
              onClick={() => setFeedback("down")}
              disabled={isSubmitting}
              style={{
                ...styles.button,
                ...(feedback === "down" ? styles.buttonActive : {}),
              }}
              title="This response was not helpful"
              aria-label="Thumbs down - not helpful"
            >
              👎
            </button>
          </div>

          {/* Reason Field (visible when 👎 clicked) */}
          {feedback === "down" && (
            <div style={styles.reasonContainer}>
              <textarea
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="What could be improved? (optional)"
                maxLength={500}
                disabled={isSubmitting}
                style={styles.reasonInput}
              />
              <div style={styles.charCounter}>
                {reason.length}/500
              </div>

              <div style={styles.reasonButtonGroup}>
                <button
                  onClick={() => handleFeedback("down")}
                  disabled={isSubmitting || !reason.trim()}
                  style={{
                    ...styles.submitButton,
                    ...(isSubmitting || !reason.trim()
                      ? styles.submitButtonDisabled
                      : {}),
                  }}
                >
                  {isSubmitting ? "Sending..." : "Send Feedback"}
                </button>

                <button
                  onClick={() => {
                    setFeedback(null);
                    setReason("");
                    setError(null);
                  }}
                  disabled={isSubmitting}
                  style={styles.cancelButton}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div style={styles.errorMessage}>
              ⚠️ {error}
            </div>
          )}
        </>
      )}
    </div>
  );
};

// ============================================
// STYLES
// ============================================

const styles: Record<string, React.CSSProperties> = {
  container: {
    marginTop: "16px",
    padding: "12px",
    borderRadius: "8px",
    backgroundColor: "#f9f9f9",
    border: "1px solid #e0e0e0",
    fontSize: "14px",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  },

  label: {
    marginBottom: "8px",
    fontSize: "12px",
    fontWeight: 500,
    color: "#666",
  },

  buttonGroup: {
    display: "flex",
    gap: "8px",
    marginBottom: "8px",
  },

  button: {
    padding: "6px 12px",
    border: "1px solid #d0d0d0",
    borderRadius: "6px",
    backgroundColor: "#fff",
    cursor: "pointer",
    fontSize: "16px",
    transition: "all 0.2s ease",
    flex: 1,
    textAlign: "center" as const,
  },

  buttonActive: {
    backgroundColor: "#e8f4f8",
    borderColor: "#0078d4",
    boxShadow: "0 0 0 1px #0078d4",
  },

  reasonContainer: {
    marginTop: "8px",
    paddingTop: "8px",
    borderTop: "1px solid #e0e0e0",
  },

  reasonInput: {
    width: "100%",
    padding: "8px",
    border: "1px solid #d0d0d0",
    borderRadius: "6px",
    fontFamily: "inherit",
    fontSize: "13px",
    resize: "vertical" as const,
    minHeight: "60px",
    boxSizing: "border-box" as const,
  },

  charCounter: {
    fontSize: "11px",
    color: "#999",
    textAlign: "right" as const,
    marginTop: "4px",
    marginBottom: "8px",
  },

  reasonButtonGroup: {
    display: "flex",
    gap: "8px",
  },

  submitButton: {
    flex: 1,
    padding: "6px 12px",
    backgroundColor: "#0078d4",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "13px",
    fontWeight: 500,
    transition: "background-color 0.2s ease",
  },

  submitButtonDisabled: {
    backgroundColor: "#c0c0c0",
    cursor: "not-allowed",
  },

  cancelButton: {
    padding: "6px 12px",
    backgroundColor: "#f0f0f0",
    color: "#333",
    border: "1px solid #d0d0d0",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "13px",
    transition: "background-color 0.2s ease",
  },

  successMessage: {
    color: "#107c10",
    fontSize: "13px",
    fontWeight: 500,
    textAlign: "center" as const,
    padding: "8px",
  },

  errorMessage: {
    color: "#d83b01",
    fontSize: "12px",
    marginTop: "8px",
    padding: "8px",
    backgroundColor: "#fff4ce",
    borderRadius: "4px",
  },
};

export default FeedbackButton;
