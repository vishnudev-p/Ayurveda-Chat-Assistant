import ChatLoading from "../ChatLoading";

export default function ChatLoadingExample() {
  return (
    <div className="space-y-0">
      <div className="bg-background py-6 px-4">
        <p className="text-sm text-muted-foreground">User message would appear here...</p>
      </div>
      <ChatLoading />
    </div>
  );
}
