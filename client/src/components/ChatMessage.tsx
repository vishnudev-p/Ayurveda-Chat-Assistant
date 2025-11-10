import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";

export interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
}

export default function ChatMessage({ role, content }: ChatMessageProps) {
  const isUser = role === "user";

  return (
    <div
      className={`flex gap-4 py-6 px-4 md:px-6 ${
        isUser ? "bg-background" : "bg-muted/30"
      }`}
      data-testid={`message-${role}`}
    >
      <div className="flex-shrink-0">
        <div
          className={`w-8 h-8 rounded-md flex items-center justify-center ${
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground"
          }`}
          data-testid={`avatar-${role}`}
        >
          {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
        </div>
      </div>
      <div className="flex-1 overflow-hidden">
        <div
          className="prose prose-sm md:prose-base dark:prose-invert max-w-none"
          data-testid={`content-${role}`}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              code({ node, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || "");
                const isInline = !match;
                
                if (isInline) {
                  return (
                    <code
                      className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono"
                      {...props}
                    >
                      {children}
                    </code>
                  );
                }
                
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
              pre({ children, ...props }) {
                return (
                  <pre
                    className="bg-muted rounded-lg p-4 overflow-x-auto"
                    {...props}
                  >
                    {children}
                  </pre>
                );
              },
              ul({ children, ...props }) {
                return (
                  <ul className="list-disc pl-6 space-y-2" {...props}>
                    {children}
                  </ul>
                );
              },
              ol({ children, ...props }) {
                return (
                  <ol className="list-decimal pl-6 space-y-2" {...props}>
                    {children}
                  </ol>
                );
              },
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
