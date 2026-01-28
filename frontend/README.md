# Document Processing Frontend

Minimalistic React frontend for the Document Processing system.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Configuration

The API base URL defaults to `http://localhost:8000`. You can override this by setting the `VITE_API_BASE_URL` environment variable:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Features

- **Home Page**: Lists all processed documents with their status
- **Upload Page**: Upload PDF files and automatically process them (markdown conversion → vectorization)
- **Chat Interface**: Query processed documents using the agentic chat system

## Requirements

- Node.js 16+ 
- Backend API running on port 8000 (or configure via `VITE_API_BASE_URL`)
