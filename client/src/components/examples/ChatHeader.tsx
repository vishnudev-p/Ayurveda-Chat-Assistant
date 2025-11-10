import ChatHeader from "../ChatHeader";
import { ThemeProvider } from "../ThemeProvider";

export default function ChatHeaderExample() {
  const handleToggleSidebar = () => {
    console.log("Toggle sidebar clicked");
  };

  return (
    <ThemeProvider>
      <ChatHeader onToggleSidebar={handleToggleSidebar} />
    </ThemeProvider>
  );
}
