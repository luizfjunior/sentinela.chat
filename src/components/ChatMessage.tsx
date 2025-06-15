
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { UserRound } from "lucide-react";
import ReactMarkdown from 'react-markdown';

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
  isTyping?: boolean;
}

interface ChatMessageProps {
  message: Message;
  isNewMessage?: boolean;
}

const ChatMessage = ({
  message,
  isNewMessage = false
}: ChatMessageProps) => {
  const [displayedContent, setDisplayedContent] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isThinking, setIsThinking] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const isUser = message.role === "user";

  // Process content to handle line breaks
  const processContent = (content: string) => {
    return content.replace(/\\n\\n/g, '\n\n').replace(/\\n/g, '\n');
  };
  const processedContent = processContent(message.content);

  // Handle typing animation for new AI messages
  useEffect(() => {
    if (isUser) {
      setDisplayedContent(processedContent);
      return;
    }

    // If this is a typing message (empty content), show thinking
    if (message.isTyping && !message.content) {
      setIsThinking(true);
      setIsTyping(false);
      setDisplayedContent("");
      setCurrentIndex(0);
      return;
    }

    // If this is a new message with content, start typing animation
    if (isNewMessage && message.content && !message.isTyping) {
      setIsThinking(false);
      setIsTyping(true);
      setDisplayedContent("");
      setCurrentIndex(0);
      return;
    }

    // For existing messages, show full content immediately
    if (!isNewMessage) {
      setDisplayedContent(processedContent);
      setIsThinking(false);
      setIsTyping(false);
      return;
    }
  }, [message, isUser, processedContent, isNewMessage]);

  // Handle typing animation
  useEffect(() => {
    if (!isTyping || isUser || !processedContent) return;
    
    if (currentIndex < processedContent.length) {
      const char = processedContent[currentIndex];
      // Variable speed: slower for punctuation, faster for spaces
      const delay = char === '.' || char === '!' || char === '?' ? 100 : char === ',' || char === ';' ? 80 : char === ' ' ? 10 : 20;
      
      const timeoutId = setTimeout(() => {
        setDisplayedContent(processedContent.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, delay);
      
      return () => clearTimeout(timeoutId);
    } else {
      setIsTyping(false);
    }
  }, [currentIndex, isTyping, isUser, processedContent]);

  const renderThinkingDots = () => (
    <div className="flex items-center space-x-1">
      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0ms' }}></div>
      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '150ms' }}></div>
      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '300ms' }}></div>
    </div>
  );

  return (
    <div className={cn("group relative", isUser ? "ml-auto max-w-[80%]" : "mr-auto max-w-full")}>
      <div className={cn("flex gap-4", isUser ? "justify-end" : "justify-start")}>
        {!isUser && (
          <div className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center flex-shrink-0">
            <img src="/lovable-uploads/iconeIA.jpg" alt="IA" className="w-full h-full object-cover" />
          </div>
        )}
        
        <div className="max-w-full break-words">
          {isUser ? (
            <p className="whitespace-pre-wrap leading-relaxed text-white">{displayedContent}</p>
          ) : (
            <div className="markdown-content text-white">
              {isThinking ? (
                renderThinkingDots()
              ) : (
                <ReactMarkdown components={{
                  p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed text-white">{children}</p>,
                  h1: ({ children }) => <h1 className="text-xl font-bold mb-3 text-white">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-lg font-bold mb-2 text-white">{children}</h2>,
                  h3: ({ children }) => <h3 className="text-base font-bold mb-2 text-white">{children}</h3>,
                  strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
                  em: ({ children }) => <em className="italic text-white">{children}</em>,
                  ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1 ml-4 text-white">{children}</ol>,
                  ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1 ml-4 text-white">{children}</ul>,
                  li: ({ children }) => <li className="leading-relaxed text-white">{children}</li>,
                  code: ({ children }) => <code className="bg-gray-700 px-1 py-0.5 rounded text-sm font-mono text-white">{children}</code>,
                  pre: ({ children }) => <pre className="bg-gray-700 p-3 rounded-lg overflow-x-auto mb-3 text-white">{children}</pre>,
                  blockquote: ({ children }) => <blockquote className="border-l-4 border-gray-600 pl-4 italic mb-3 text-white">{children}</blockquote>
                }}>
                  {displayedContent}
                </ReactMarkdown>
              )}
            </div>
          )}
        </div>
        
        {isUser && (
          <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-zinc-600">
            <UserRound className="h-4 w-4 text-white" />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
