import { MessageSquare, MessageSquarePlus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { formatDistanceToNow } from "date-fns";

export interface ChatSession {
  id: string;
  title: string;
  lastMessageAt: Date;
  messageCount: number;
}

export interface ChatSidebarProps {
  sessions: ChatSession[];
  activeSessionId: string;
  onSessionSelect: (sessionId: string) => void;
  onNewChat: () => void;
  onDeleteSession: (sessionId: string) => void;
}

export default function ChatSidebar({
  sessions,
  activeSessionId,
  onSessionSelect,
  onNewChat,
  onDeleteSession,
}: ChatSidebarProps) {
  return (
    <div className="flex flex-col h-full bg-sidebar border-r">
      <div className="p-3 border-b">
        <Button
          onClick={onNewChat}
          className="w-full gap-2"
          data-testid="button-sidebar-new-chat"
        >
          <MessageSquarePlus className="w-4 h-4" />
          New Chat
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {sessions.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted-foreground">
              No conversations yet
            </div>
          ) : (
            sessions.map((session) => (
              <div
                key={session.id}
                className={`group relative rounded-md hover-elevate ${
                  session.id === activeSessionId
                    ? "bg-sidebar-accent"
                    : ""
                }`}
              >
                <button
                  onClick={() => onSessionSelect(session.id)}
                  className="w-full text-left p-3 pr-10 rounded-md"
                  data-testid={`button-session-${session.id}`}
                >
                  <div className="flex items-start gap-2">
                    <MessageSquare className="w-4 h-4 flex-shrink-0 mt-0.5 text-sidebar-foreground" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate text-sidebar-foreground">
                        {session.title}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {formatDistanceToNow(session.lastMessageAt, {
                          addSuffix: true,
                        })}
                      </p>
                    </div>
                  </div>
                </button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                  data-testid={`button-delete-${session.id}`}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
