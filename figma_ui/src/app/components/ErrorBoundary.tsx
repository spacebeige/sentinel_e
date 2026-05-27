import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

interface Props {
  children?: ReactNode;
  fallbackMessage?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 m-4 rounded-3xl backdrop-blur-2xl bg-red-500/5 dark:bg-red-500/10 border border-red-500/20 shadow-2xl">
          <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mb-6">
            <AlertCircle className="w-8 h-8 text-red-500 animate-pulse" />
          </div>
          <h2 className="text-xl font-bold tracking-tight text-zinc-900 dark:text-white mb-2">
            Cinematic Core Fault
          </h2>
          <p className="text-zinc-600 dark:text-zinc-400 text-center max-w-md mb-6 font-medium text-sm">
            {this.props.fallbackMessage || "A rendering exception occurred in this layer of the machine. The cognitive core remains active."}
            <br/><br/>
            <span className="opacity-50 text-xs font-mono">{this.state.error?.message}</span>
          </p>
          <button
            onClick={this.handleReset}
            className="flex items-center gap-2 px-6 py-2.5 rounded-full bg-red-500 text-white font-semibold text-sm tracking-wide hover:bg-red-600 transition-colors shadow-lg shadow-red-500/25"
          >
            <RefreshCw className="w-4 h-4" />
            Re-initialize Layer
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
