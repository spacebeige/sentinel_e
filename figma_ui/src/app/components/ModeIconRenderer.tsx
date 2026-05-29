import React from "react";
import { Swords, Gem, FileSearch, Sparkles, GitMerge, Brain } from "lucide-react";

interface ModeIconRendererProps {
  iconName?: string;
  className?: string;
}

export const ModeIconRenderer: React.FC<ModeIconRendererProps> = ({ iconName, className = "w-3.5 h-3.5" }) => {
  switch (iconName) {
    case "swords":
      return <Swords className={className} />;
    case "gem":
      return <Gem className={className} />;
    case "file-search":
      return <FileSearch className={className} />;
    case "sparkles":
      return <Sparkles className={className} />;
    case "git-merge":
      return <GitMerge className={className} />;
    default:
      return <Brain className={className} />;
  }
};
