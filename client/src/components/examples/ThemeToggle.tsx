import ThemeToggle from "../ThemeToggle";
import { ThemeProvider } from "../ThemeProvider";

export default function ThemeToggleExample() {
  return (
    <ThemeProvider>
      <div className="flex items-center justify-center h-screen gap-4">
        <p className="text-foreground">Click to toggle theme:</p>
        <ThemeToggle />
      </div>
    </ThemeProvider>
  );
}
