# Design Guidelines: ChatGPT-like Chat Application

## Design Approach
**Selected Approach:** Design System (Utility-Focused)  
**References:** ChatGPT, Claude AI, Google Gemini  
**Rationale:** Chat interfaces prioritize clarity, readability, and efficiency. Users need to focus on conversation content without visual distractions.

## Core Design Principles
1. **Conversation-First:** Maximum reading comfort with optimal line lengths and spacing
2. **Clarity:** Clear visual distinction between user and AI messages
3. **Efficiency:** Minimal UI chrome to maximize content area
4. **Responsiveness:** Seamless experience across all device sizes

## Typography System
**Font Families:**
- Primary: Inter or system-ui for UI elements
- Content: Georgia or Charter for message text (improved readability)
- Code: JetBrains Mono or Fira Code for code blocks

**Hierarchy:**
- Message text: 16px (base), line-height 1.6
- Timestamps: 12px, weight 400
- Input placeholder: 15px, weight 400
- Code inline: 14px
- Headings in messages: Use semantic sizing (1.5em, 1.3em, 1.1em)

## Layout System
**Spacing Primitives:** Tailwind units of 2, 3, 4, 6, 8, 12, 16

**Main Layout:**
- Full viewport height container
- Left sidebar (280px desktop, collapsible mobile): New chat button, conversation history
- Main chat area: Flexible width with max-w-3xl centered content column
- Fixed bottom input area with consistent padding

**Message Container:**
- Alternating background treatment for user vs AI messages
- User messages: Align right, max-w-2xl
- AI messages: Align left, max-w-3xl for optimal reading
- Padding: py-6 px-4 (mobile), py-8 px-6 (desktop)
- Message spacing: gap-4 between messages

## Component Library

**Sidebar:**
- Fixed position on desktop, slide-out drawer on mobile
- "New Chat" button at top: Full width, prominent placement, p-3
- Previous chats list below (if implementing history display)
- Collapse/expand toggle (mobile hamburger icon)

**Chat Messages:**
- User message bubble: Compact container, right-aligned
- AI message container: Full width with avatar/icon on left
- Avatar size: w-8 h-8
- Message content area: Prose formatting with proper spacing
- Timestamp: Bottom right, subtle, text-xs

**Markdown Rendering:**
- Headings: Bold, increased size (1.5em, 1.3em, 1.1em)
- Bold/Italic: Standard semantic formatting
- Lists: pl-6, proper bullet/number spacing (space-y-2)
- Code blocks: Rounded corners (rounded-lg), p-4, syntax highlighting via library
- Inline code: Subtle background, px-1.5, rounded

**Input Area:**
- Fixed bottom position with backdrop blur
- Container: max-w-3xl centered, p-4
- Textarea: Auto-expanding, min-h-12, max-h-32, rounded-xl
- Submit button: Icon-only (send arrow), positioned inside textarea (bottom right), rounded-full
- Border: Subtle outline on focus
- Responsive: Full width on mobile with p-3

**Loading States:**
- AI thinking indicator: Animated dots or pulse effect
- Message streaming: Character-by-character reveal
- Disabled input during API calls

**Empty State:**
- Centered in viewport when no messages
- Welcome message with suggested prompts (3-4 example questions)
- Example cards: Grid layout (grid-cols-1 md:grid-cols-2), gap-3, clickable

## Navigation & Actions
- "New Chat" button: Top of sidebar, clear icon + text
- Mobile menu: Hamburger icon (top left) to reveal sidebar
- Clear conversation action triggers new session

## Responsive Breakpoints
- Mobile (<768px): Single column, collapsible sidebar, full-width input
- Tablet (768px-1024px): Sidebar visible, max-w-2xl messages
- Desktop (>1024px): Full layout, max-w-3xl messages, sidebar always visible

## Animation Guidelines
**Minimal, Purposeful Only:**
- Message appearance: Simple fade-in (150ms)
- Sidebar toggle: Slide transition (200ms)
- No scroll animations or excessive motion
- Loading states: Subtle pulse or dots only

## Accessibility
- Semantic HTML for all messages (article/section tags)
- ARIA labels for icon-only buttons
- Keyboard navigation: Tab through interactive elements, Enter to send
- Focus indicators on all interactive elements
- Screen reader announcements for new messages

## Images
**No hero images required.** This is a utility application focused entirely on conversation. The only visual elements are:
- Optional: Small brand logo in sidebar header (h-8)
- AI avatar icon next to each response (w-8 h-8)
- User avatar/initial in user messages (w-6 h-6)