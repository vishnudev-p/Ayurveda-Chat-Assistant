import { Sparkles, Code, Lightbulb, MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface ChatEmptyStateProps {
  onExampleClick: (example: string) => void;
}

export default function ChatEmptyState({ onExampleClick }: ChatEmptyStateProps) {
  const examples = [
    {
      icon: Code,
      text: "Explain how async/await works in JavaScript",
      prompt: "Can you explain how async/await works in JavaScript with examples?",
    },
    {
      icon: Lightbulb,
      text: "Give me creative ideas for a blog post",
      prompt: "I need creative ideas for a blog post about web development trends",
    },
    {
      icon: MessageCircle,
      text: "Help me write a professional email",
      prompt: "Can you help me write a professional email to a client about a project delay?",
    },
    {
      icon: Sparkles,
      text: "Summarize a complex topic simply",
      prompt: "Can you explain quantum computing in simple terms?",
    },
  ];

  return (
    <div className="flex-1 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full space-y-8">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
            <MessageCircle className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-2xl font-semibold" data-testid="text-welcome">
            Welcome to ChatAI
          </h2>
          <p className="text-muted-foreground">
            Start a conversation or try one of these examples
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {examples.map((example, index) => {
            const Icon = example.icon;
            return (
              <Button
                key={index}
                variant="outline"
                className="h-auto p-4 justify-start text-left hover-elevate active-elevate-2"
                onClick={() => onExampleClick(example.prompt)}
                data-testid={`button-example-${index}`}
              >
                <div className="flex gap-3 items-start w-full">
                  <Icon className="w-5 h-5 flex-shrink-0 mt-0.5 text-primary" />
                  <span className="text-sm">{example.text}</span>
                </div>
              </Button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
