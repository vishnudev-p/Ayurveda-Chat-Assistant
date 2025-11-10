import ChatInput from "../ChatInput";

export default function ChatInputExample() {
  const handleSend = (message: string) => {
    console.log("Message sent:", message);
  };

  return (
    <div className="h-screen flex flex-col">
      <div className="flex-1 flex items-center justify-center p-4">
        <p className="text-muted-foreground text-center">
          Type a message below and press Enter or click the send button
        </p>
      </div>
      <ChatInput onSend={handleSend} />
    </div>
  );
}
