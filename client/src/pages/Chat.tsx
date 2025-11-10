import { useState, useRef, useEffect } from "react";
import ChatHeader from "@/components/ChatHeader";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import ChatEmptyState from "@/components/ChatEmptyState";
import ChatLoading from "@/components/ChatLoading";
import ChatSidebar, { ChatSession } from "@/components/ChatSidebar";
import { useToast } from "@/hooks/use-toast";
import { Sheet, SheetContent } from "@/components/ui/sheet";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface ChatSessionWithMessages extends ChatSession {
  messages: Message[];
}

export default function Chat() {
  const [sessions, setSessions] = useState<ChatSessionWithMessages[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const messages = activeSession?.messages || [];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const createNewSession = (): string => {
    const newSessionId = Date.now().toString();
    const newSession: ChatSessionWithMessages = {
      id: newSessionId,
      title: "New Chat",
      lastMessageAt: new Date(),
      messageCount: 0,
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSessionId);
    setIsSidebarOpen(false);
    return newSessionId;
  };

  const handleNewChat = () => {
    createNewSession();
    toast({
      title: "New chat started",
      description: "Your previous conversations are saved in the sidebar",
    });
  };

  const handleSessionSelect = (sessionId: string) => {
    setActiveSessionId(sessionId);
    setIsSidebarOpen(false);
  };

  const handleDeleteSession = (sessionId: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    if (activeSessionId === sessionId) {
      const remainingSessions = sessions.filter((s) => s.id !== sessionId);
      if (remainingSessions.length > 0) {
        setActiveSessionId(remainingSessions[0].id);
      } else {
        setActiveSessionId("");
      }
    }
    toast({
      title: "Chat deleted",
      description: "The conversation has been removed",
    });
  };

  const updateSessionTitle = (sessionId: string, firstMessage: string) => {
    setSessions((prev) =>
      prev.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              title: firstMessage.slice(0, 50) + (firstMessage.length > 50 ? "..." : ""),
            }
          : session
      )
    );
  };

  const handleSendMessage = async (content: string) => {
    let currentSessionId = activeSessionId;

    if (!currentSessionId) {
      currentSessionId = createNewSession();
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
    };

    setSessions((prev) =>
      prev.map((session) =>
        session.id === currentSessionId
          ? {
              ...session,
              messages: [...session.messages, userMessage],
              lastMessageAt: new Date(),
              messageCount: session.messageCount + 1,
            }
          : session
      )
    );

    if (activeSession?.messageCount === 0) {
      updateSessionTitle(currentSessionId, content);
    }

    setIsLoading(true);

    try {
      const response = await fetch(
        "https://machineless-saran-torquate.ngrok-free.dev/generate",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ question: content }),
        }
      );

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response || data.message || "I received your message!",
      };

      setSessions((prev) =>
        prev.map((session) =>
          session.id === currentSessionId
            ? {
                ...session,
                messages: [...session.messages, assistantMessage],
                lastMessageAt: new Date(),
                messageCount: session.messageCount + 1,
              }
            : session
        )
      );
    } catch (error) {
      console.error("Error sending message:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to send message. Please try again.",
      });

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "I apologize, but I'm having trouble connecting to the server right now. Please try again later.",
      };

      setSessions((prev) =>
        prev.map((session) =>
          session.id === currentSessionId
            ? {
                ...session,
                messages: [...session.messages, errorMessage],
                lastMessageAt: new Date(),
                messageCount: session.messageCount + 1,
              }
            : session
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (example: string) => {
    handleSendMessage(example);
  };

  const chatSessions: ChatSession[] = sessions.map(({ messages, ...session }) => session);

  return (
    <div className="flex h-screen">
      {/* Desktop Sidebar */}
      <div className="hidden md:block w-64 flex-shrink-0">
        <ChatSidebar
          sessions={chatSessions}
          activeSessionId={activeSessionId}
          onSessionSelect={handleSessionSelect}
          onNewChat={handleNewChat}
          onDeleteSession={handleDeleteSession}
        />
      </div>

      {/* Mobile Sidebar */}
      <Sheet open={isSidebarOpen} onOpenChange={setIsSidebarOpen}>
        <SheetContent side="left" className="p-0 w-64">
          <ChatSidebar
            sessions={chatSessions}
            activeSessionId={activeSessionId}
            onSessionSelect={handleSessionSelect}
            onNewChat={handleNewChat}
            onDeleteSession={handleDeleteSession}
          />
        </SheetContent>
      </Sheet>

      {/* Main Chat Area */}
      <div className="flex flex-col flex-1 min-w-0">
        <ChatHeader onToggleSidebar={() => setIsSidebarOpen(true)} />

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <ChatEmptyState onExampleClick={handleExampleClick} />
          ) : (
            <div>
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  role={message.role}
                  content={message.content}
                />
              ))}
              {isLoading && <ChatLoading />}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <ChatInput onSend={handleSendMessage} disabled={isLoading} />
      </div>
    </div>
  );
}
