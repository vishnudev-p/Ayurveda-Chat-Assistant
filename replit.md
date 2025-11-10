# ChatAI - Intelligent Conversation Assistant

## Overview

ChatAI is a ChatGPT-like conversational AI application built with a modern web stack. The application provides an intelligent chat interface with markdown support, conversation history management, and a clean, responsive design inspired by popular AI chat platforms like ChatGPT, Claude, and Google Gemini.

The system follows a conversation-first design philosophy, prioritizing readability, clarity, and efficiency. The UI is built using shadcn/ui components with Tailwind CSS for styling, ensuring a consistent and polished user experience across all device sizes.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework & Build System:**
- React 18 with TypeScript for type-safe component development
- Vite as the build tool and development server
- Wouter for lightweight client-side routing
- React Query (@tanstack/react-query) for server state management

**UI Component System:**
- shadcn/ui component library (New York style variant) providing pre-built, accessible components
- Radix UI primitives for low-level accessibility and behavior
- Tailwind CSS for utility-first styling with custom design tokens
- Class Variance Authority (CVA) for type-safe component variants

**Design System:**
- Custom color system using HSL values with CSS custom properties for theme support
- Light and dark theme support via ThemeProvider context
- Typography system with specific font stacks: Inter/system-ui for UI, Georgia/Charter for message content, JetBrains Mono/Fira Code for code blocks
- Spacing primitives based on Tailwind's spacing scale (2, 3, 4, 6, 8, 12, 16)

**State Management Strategy:**
- Local React state for UI interactions and form handling
- React Query for server data caching and synchronization
- Context API for cross-cutting concerns (theme)

**Key Layout Decisions:**
- Full-height viewport container with fixed sidebar on desktop (280px)
- Collapsible mobile sidebar using Sheet component
- Centered content column with max-w-3xl constraint for optimal reading
- Fixed bottom input area with auto-expanding textarea
- Alternating backgrounds for user vs AI messages for visual clarity

### Backend Architecture

**Server Framework:**
- Express.js as the HTTP server
- TypeScript for type safety across the stack
- ESM module system throughout

**Session Management:**
- In-memory storage implementation (MemStorage class)
- Interface-based storage abstraction (IStorage) allowing easy swapping of persistence layers
- User authentication schema defined but not yet implemented in routes

**Development Tools:**
- Custom Vite middleware for HMR in development
- Request/response logging middleware
- Runtime error overlay via Replit plugins

**Build & Deployment:**
- Separate client and server builds
- Client builds to `dist/public` via Vite
- Server bundles with esbuild to `dist/index.js`
- Production mode serves static files from build output

### Data Storage Solutions

**Database Configuration:**
- Drizzle ORM configured for PostgreSQL via `@neondatabase/serverless`
- Schema-first approach with TypeScript types generated from Drizzle schema
- Migrations managed via Drizzle Kit (output to `./migrations`)

**Current Schema:**
- Users table with UUID primary keys, username (unique), and password fields
- Zod validation schemas generated via drizzle-zod for runtime type safety
- Schema location: `shared/schema.ts` for sharing between client and server

**Storage Pattern:**
- Repository pattern via IStorage interface
- Current implementation uses in-memory Map for rapid prototyping
- Designed for easy migration to PostgreSQL-backed implementation

### Chat Features Architecture

**Message Handling:**
- Message interface with id, role (user/assistant), and content
- Chat sessions with title, timestamp, message count, and message array
- Session management entirely client-side (no persistence yet)

**Markdown Rendering:**
- ReactMarkdown with GitHub Flavored Markdown (remark-gfm)
- Syntax highlighting for code blocks via rehype-highlight
- Custom code component styling for inline vs block code
- Prose styling via Tailwind typography classes

**UI Components:**
- ChatMessage: Displays individual messages with role-based styling
- ChatInput: Auto-expanding textarea with keyboard shortcuts (Enter to send, Shift+Enter for newline)
- ChatEmptyState: Welcome screen with example prompts
- ChatLoading: Animated loading indicator for AI responses
- ChatSidebar: Conversation history with delete functionality
- ChatHeader: App title with theme toggle and mobile menu button

## External Dependencies

### UI & Component Libraries
- **Radix UI**: Complete suite of accessible, unstyled UI primitives (accordion, alert-dialog, avatar, checkbox, collapsible, context-menu, dialog, dropdown-menu, hover-card, label, menubar, navigation-menu, popover, progress, radio-group, scroll-area, select, separator, slider, slot, switch, tabs, toast, toggle, tooltip)
- **shadcn/ui**: Pre-configured component system built on Radix UI
- **cmdk**: Command palette component
- **Lucide React**: Icon library for consistent iconography
- **embla-carousel-react**: Carousel/slider component

### Form & Validation
- **React Hook Form**: Form state management
- **@hookform/resolvers**: Validation resolver for React Hook Form
- **Zod**: Runtime type validation and schema declaration
- **drizzle-zod**: Zod schema generation from Drizzle ORM schemas

### Markdown & Syntax Highlighting
- **react-markdown**: Markdown rendering in React
- **remark-gfm**: GitHub Flavored Markdown support
- **rehype-highlight**: Syntax highlighting for code blocks
- **highlight.js**: Syntax highlighting engine

### Styling
- **Tailwind CSS**: Utility-first CSS framework
- **tailwind-merge**: Utility for merging Tailwind classes
- **clsx**: Conditional class name utility
- **class-variance-authority**: Type-safe variant API for components
- **autoprefixer**: CSS vendor prefix automation
- **PostCSS**: CSS transformation tool

### Database & ORM
- **Drizzle ORM**: TypeScript ORM for SQL databases
- **@neondatabase/serverless**: Serverless PostgreSQL driver for Neon
- **Drizzle Kit**: Schema management and migration tool
- **connect-pg-simple**: PostgreSQL session store (configured but not actively used)

### Development Tools
- **Vite**: Fast build tool and dev server
- **@vitejs/plugin-react**: React support for Vite
- **@replit/vite-plugin-runtime-error-modal**: Runtime error overlay
- **@replit/vite-plugin-cartographer**: Code navigation (development only)
- **@replit/vite-plugin-dev-banner**: Development banner (development only)
- **tsx**: TypeScript execution for Node.js
- **esbuild**: Fast JavaScript bundler for production builds

### Date & Time
- **date-fns**: Modern date utility library for formatting and manipulation

### Routing
- **wouter**: Minimal routing library (1KB) as lightweight alternative to React Router