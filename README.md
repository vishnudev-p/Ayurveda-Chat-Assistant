#  Ayurveda Chat Assistant – Pure Knowledge Meets AI Wisdom 🌿

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![MongoDB](https://img.shields.io/badge/MongoDB-Vector_Storage-brightgreen)
![RAG](https://img.shields.io/badge/RAG-Powered-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A **Retrieval-Augmented Generation (RAG)** powered **Ayurveda Chat Assistant** that answers only **Ayurveda-related questions**, sourced exclusively from verified Ayurvedic texts.  
> No random data. No hallucinations. Just **authentic Ayurvedic wisdom**, intelligently delivered.

---

## 🌿 Overview

The **Ayurveda Chat Assistant** bridges the gap between **ancient Ayurvedic wisdom** and **modern AI technology**.  
It uses **RAG architecture** to retrieve relevant information, rerank responses, and generate insightful answers — all derived solely from an internal Ayurveda dataset stored in **MongoDB Vector Storage**.

🧠 Built with:
- **Sentence Transformers** for embeddings  
- **CrossEncoder** for reranking  
- **Gemma** and **Mistral** for contextual generation  
- **FastAPI** for backend  
- **MongoDB** for secure and efficient vector storage  

---
## 🎥 Demo Interface

Here’s a quick look at the Ayurveda Chat Assistant in action 👇  

![Ayurveda Chat Assistant Demo](assets/ayurveda-chat-demo%20-%20Copy.gif)

*(The animated preview above is stored in `/assets/`. For better resolution, you can also see the full `.mp4` demo from there.)*  



---

## 🌟 Key Features

✅ **Domain-Specific Knowledge** — Answers only Ayurveda questions  
✅ **RAG-Based Retrieval** — Uses hybrid (BM25 + Embedding) search  
✅ **MongoDB Vector Storage** — Efficient storage and retrieval of embeddings  
✅ **CrossEncoder Reranking** — Improves relevance and accuracy  
✅ **Gemma/Mistral for Generation** — Produces well-structured Ayurvedic responses  
✅ **FastAPI + Ngrok** — Lightweight API-based architecture  
✅ **Zero Hallucination** — Uses only verified internal Ayurvedic content  

---

## ⚙️ System Architecture

```text
 ┌──────────────────────────┐
 │        User Query        │
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   SentenceTransformer    │  → Embedding model
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   BM25 + Vector Search   │  → Hybrid retrieval
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   CrossEncoder Reranker  │  → Rank top chunks
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   Generator (Gemma/Mistral) │ → Generate Ayurvedic answer


 └──────────────────────────┘

## Windows Installation Guide

### Prerequisites

Before you begin, ensure you have the following installed on your Windows machine:

1. **Node.js** (v18 or higher)
   - Download from: https://nodejs.org/
   - Choose the LTS (Long Term Support) version
   - During installation, make sure to check "Add to PATH"
   - Verify installation:
     ```cmd
     node --version
     npm --version
     ```

2. **Git** (Optional, for cloning)
   - Download from: https://git-scm.com/download/win
   - Or download the project as a ZIP file

### Installation Steps

#### Step 1: Get the Project Files

**Option A - Using Git:**
```cmd
git clone <your-repository-url>
cd <project-folder>
```

**Option B - Using ZIP:**
1. Download the project ZIP file
2. Extract to a folder of your choice
3. Open Command Prompt or PowerShell
4. Navigate to the project folder:
   ```cmd
   cd path\to\project\folder
   ```

#### Step 2: Install Dependencies

Run the following command in the project root directory:

```cmd
npm install
```

This will install all required packages listed in `package.json`, including:
- React and React DOM
- Vite (build tool)
- Express (backend server)
- TailwindCSS (styling)
- Shadcn UI components
- React Markdown
- And many more dependencies

**Note:** This may take a few minutes depending on your internet connection.

#### Step 3: Run the Application

Start the development server:

```cmd
npm run dev
```

The application will start on **http://localhost:5000**

You should see output similar to:
```
> dev
> tsx server/index.ts

Server running on http://localhost:5000
```

#### Step 4: Access the Application

1. Open your web browser (Chrome, Edge, Firefox, etc.)
2. Navigate to: **http://localhost:5000**
3. The chat interface should load

### Troubleshooting

#### Issue: "npm is not recognized"
**Solution:** Node.js is not properly installed or not in PATH. Reinstall Node.js and ensure "Add to PATH" is checked.

#### Issue: Port 5000 is already in use
**Solution:** Another application is using port 5000. Either:
- Close the other application
- Or modify the port in `server/index.ts`

#### Issue: Dependencies installation fails
**Solution:** 
1. Delete `node_modules` folder and `package-lock.json`
2. Clear npm cache: `npm cache clean --force`
3. Run `npm install` again

#### Issue: Application won't start
**Solution:**
1. Check if all dependencies installed successfully
2. Make sure you're in the project root directory
3. Try deleting `node_modules` and running `npm install` again

### Project Structure

```
project/
├── client/                 # Frontend React application
│   └── src/
│       ├── components/     # Reusable UI components
│       ├── pages/          # Page components
│       └── App.tsx         # Main app component
├── server/                 # Backend Express server
│   ├── index.ts            # Server entry point
│   └── routes.ts           # API routes
├── shared/                 # Shared types and schemas
├── package.json            # Dependencies and scripts
└── README.md              # This file
```

## Backend Setup (Python RAG System)

The chat application connects to a separate Python backend that runs the RAG system. This backend:
- Uses MongoDB for vector storage
- Implements semantic search with embeddings
- Uses Mistral-7B for response generation
- Runs on Jupyter Notebook with ngrok

### Backend Requirements

If you want to run the backend locally on Windows:

1. **Python 3.8+**
   - Download from: https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

2. **Install Python Libraries:**
   ```cmd
   pip install fastapi uvicorn pyngrok nest_asyncio sentence-transformers rank-bm25 pymongo transformers torch
   ```

3. **Jupyter Notebook:**
   ```cmd
   pip install jupyter
   ```

4. **CUDA (Optional, for GPU acceleration):**
   - If you have an NVIDIA GPU: https://developer.nvidia.com/cuda-downloads

**Note:** The backend code is designed to run in Jupyter Notebook and requires significant computational resources for the LLM models.

## Configuration

### Backend API Endpoint

The frontend connects to the backend at:
```
https://your-ngrok-link.ngrok-free.app
```

To change the backend URL, edit `client/src/pages/Chat.tsx` (line 134):
```typescript
const response = await fetch(
  "YOUR_BACKEND_URL_HERE/generate",
  // ...
);
```

### Environment Variables

No environment variables are required for the frontend. The application uses:
- In-memory session storage (no database needed)
- Client-side state management
- Direct API calls to the backend

## Available Scripts

- `npm run dev` - Start development server (frontend + backend)
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Technologies Used

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TailwindCSS** - Styling
- **Shadcn UI** - Component library
- **Wouter** - Routing
- **React Markdown** - Markdown rendering
- **Lucide React** - Icons

### Backend (Node.js)
- **Express** - Server framework
- **TypeScript** - Type safety

### Backend (Python RAG System)
- **FastAPI** - API framework
- **Sentence Transformers** - Embeddings
- **Mistral-7B** - Language model
- **MongoDB** - Vector database
- **PyTorch** - ML framework

## Browser Support

- Chrome (recommended)
- Edge
- Firefox
- Safari

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure your backend is running and accessible
3. Check browser console for error messages (F12)

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
>>>>>>> 92b65625319f72f3bf452a3372bde9b44ab931a8


