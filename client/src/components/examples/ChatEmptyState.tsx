import ChatEmptyState from "../ChatEmptyState";

export default function ChatEmptyStateExample() {
  const handleExampleClick = (example: string) => {
    console.log("Example clicked:", example);
  };

  return (
    <div className="h-screen flex flex-col">
      <ChatEmptyState onExampleClick={handleExampleClick} />
    </div>
  );
}
