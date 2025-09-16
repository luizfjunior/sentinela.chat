# Sentinela - AI Chat Assistant

Sentinela is an intelligent chat application that provides AI-powered conversations with support for both text and audio interactions. Built with modern web technologies, it features real-time messaging, user authentication, and seamless integration with AI services.

## Features

- 🤖 AI-powered chat conversations
- 🎤 Voice message recording and transcription
- 🔐 Secure user authentication with Supabase
- 💬 Real-time messaging interface
- 📱 Responsive design for desktop and mobile
- 🎨 Modern UI with dark/light mode support

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager

### Installation

1. Clone the repository:
```bash
git clone <YOUR_GIT_URL>
cd sentinela
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Fill in your Supabase credentials and other required environment variables.

4. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:8080`

### Building for Production

```bash
npm run build
```

## Technology Stack

- **Frontend Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with shadcn/ui components
- **Backend**: Supabase (Authentication, Database, Real-time)
- **State Management**: React Context API
- **Routing**: React Router DOM
- **Audio Processing**: Web Audio API for voice recordings

## Project Structure

```
src/
├── components/          # Reusable UI components
├── contexts/           # React context providers
├── hooks/              # Custom React hooks
├── integrations/       # Third-party service integrations
├── lib/                # Utility functions
├── pages/              # Page components
└── main.tsx           # Application entry point
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License.
