import ChatMessage from "../ChatMessage";

export default function ChatMessageExample() {
  return (
    <div className="space-y-0">
      <ChatMessage
        role="user"
        content="Can you help me understand how to use markdown? Show me some examples."
      />
      <ChatMessage
        role="assistant"
        content={`# Markdown Formatting Guide

I'd be happy to help! Here are some common markdown formatting options:

## Text Formatting
- **Bold text** using double asterisks
- *Italic text* using single asterisks
- ***Bold and italic*** using triple asterisks

## Lists
You can create bullet points:
- First item
- Second item
- Third item

Or numbered lists:
1. Step one
2. Step two
3. Step three

## Code
For inline code, use backticks: \`const example = "hello"\`

For code blocks, use triple backticks:

\`\`\`javascript
function greet(name) {
  console.log(\`Hello, \${name}!\`);
}

greet("World");
\`\`\`

That's the basics! Try any of these in your messages.`}
      />
    </div>
  );
}
