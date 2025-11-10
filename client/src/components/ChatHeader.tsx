import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import ThemeToggle from "./ThemeToggle";

export interface ChatHeaderProps {
  onToggleSidebar?: () => void;
}

export default function ChatHeader({ onToggleSidebar }: ChatHeaderProps) {
  return (
    <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex items-center justify-between h-14 px-4">
        <div className="flex items-center gap-2">
          {onToggleSidebar && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggleSidebar}
              className="md:hidden"
              data-testid="button-toggle-sidebar"
            >
              <Menu className="w-5 h-5" />
            </Button>
          )}
          <h1 className="text-lg font-semibold" data-testid="text-title">
            ChatAI
          </h1>
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
}
