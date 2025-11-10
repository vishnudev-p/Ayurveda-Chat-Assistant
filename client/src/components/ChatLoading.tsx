import { Bot } from "lucide-react";

export default function ChatLoading() {
  return (
    <div className="flex gap-4 py-6 px-4 md:px-6 bg-muted/30" data-testid="loading-indicator">
      <div className="flex-shrink-0">
        <div className="w-8 h-8 rounded-md flex items-center justify-center bg-muted text-muted-foreground">
          <Bot className="w-5 h-5" />
        </div>
      </div>
      <div className="flex-1">
        <div className="flex gap-1 items-center">
          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
        </div>
      </div>
    </div>
  );
}
