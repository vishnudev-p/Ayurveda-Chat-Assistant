import ChatSidebar, { ChatSession } from "../ChatSidebar";

export default function ChatSidebarExample() {
  const mockSessions: ChatSession[] = [
    {
      id: "1",
      title: "How to use React hooks",
      lastMessageAt: new Date(Date.now() - 1000 * 60 * 5),
      messageCount: 4,
    },
    {
      id: "2",
      title: "Explain async/await in JavaScript",
      lastMessageAt: new Date(Date.now() - 1000 * 60 * 30),
      messageCount: 6,
    },
    {
      id: "3",
      title: "Best practices for TypeScript",
      lastMessageAt: new Date(Date.now() - 1000 * 60 * 60 * 2),
      messageCount: 8,
    },
  ];

  const handleSessionSelect = (sessionId: string) => {
    console.log("Selected session:", sessionId);
  };

  const handleNewChat = () => {
    console.log("New chat clicked");
  };

  const handleDeleteSession = (sessionId: string) => {
    console.log("Delete session:", sessionId);
  };

  return (
    <div className="h-screen w-64">
      <ChatSidebar
        sessions={mockSessions}
        activeSessionId="1"
        onSessionSelect={handleSessionSelect}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
      />
    </div>
  );
}
